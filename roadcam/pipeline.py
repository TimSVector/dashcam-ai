import logging
import time
import cv2
from .interfaces import (VideoSource, VehicleDetector, PlateDetector, PlateReader,
                         VehicleDatabase, EventLogger)
from .models import Box, DetectionEvent, PlateCandidate
from .plates import normalize_plate

LOG = logging.getLogger(__name__)


def annotate(image, box, text, color=(0, 220, 0)):
    cv2.rectangle(image, (box.x1, box.y1), (box.x2 - 1, box.y2 - 1), color, 2)
    cv2.putText(image, text, (box.x1, max(18, box.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def annotate_plate_candidate(image, candidate: PlateCandidate, debug: bool):
    if candidate.accepted:
        annotate(image, candidate.box, f"plate {candidate.confidence:.2f}", (0, 200, 255))
    elif debug:
        annotate(image, candidate.box, f"reject: {candidate.rejection_reason}", (0, 0, 255))


class Pipeline:
    def __init__(self, detector: VehicleDetector, plate_detector: PlateDetector | None = None,
                 plate_reader: PlateReader | None = None, database: VehicleDatabase | None = None,
                 event_logger: EventLogger | None = None, ocr_confidence=0.5, source_name="",
                 native_core=None, plate_debug=False):
        self.detector = detector
        stages = (plate_detector, plate_reader, database, event_logger)
        if any(s is not None for s in stages) and not all(s is not None for s in stages):
            raise ValueError("Plate processing requires detector, reader, database, and logger")
        self.plate_detector, self.plate_reader = plate_detector, plate_reader
        self.database, self.event_logger = database, event_logger
        self.ocr_confidence, self.source_name = ocr_confidence, source_name
        self.native_core = native_core
        self.plate_debug = plate_debug

    def process(self, frame):
        image = frame.image.copy()
        for vehicle_index, vehicle in enumerate(self.detector.detect(frame.image)):
            box = vehicle.box.clipped(image.shape[1], image.shape[0])
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            if self.native_core and not self.native_core.decide_vehicle(
                    vehicle.label, vehicle.confidence, box, frame.index, frame.timestamp).accepted:
                continue
            annotate(image, box, f"{vehicle.label} {vehicle.confidence:.2f}")
            if self.plate_detector is None:
                continue
            crop = box.crop(frame.image)  # Never OCR the drawn annotations.
            seen = set()
            for proposal in self.plate_detector.detect_candidates(crop):
                local = proposal.box.clipped(crop.shape[1], crop.shape[0])
                absolute = Box(box.x1 + local.x1, box.y1 + local.y1,
                               box.x1 + local.x2, box.y1 + local.y2)
                candidate = PlateCandidate(absolute, proposal.confidence, vehicle_index,
                                           vehicle.label, proposal.accepted, proposal.rejection_reason)
                annotate_plate_candidate(image, candidate, self.plate_debug)
                if not candidate.accepted:
                    continue
                plate_image = local.crop(crop)
                if plate_image.size == 0:
                    continue
                read = self.plate_reader.read(plate_image)
                if read is None or not self.ocr_confidence <= read.confidence <= 1:
                    continue
                plate = normalize_plate(read.text)
                if not plate or plate in seen:
                    continue
                seen.add(plate)
                known = self.database.lookup(plate)
                text = f"{plate} {read.confidence:.2f}"
                if known:
                    text += f" KNOWN VEHICLE: {known.description}"
                annotate(image, absolute, text, (255, 220, 0))
                event = DetectionEvent(frame.timestamp, plate, read.confidence, vehicle.label,
                                       known.id if known else None, self.source_name, frame.index)
                self.event_logger.log(event)
                LOG.info("Plate %s at %.2fs%s", plate, frame.timestamp,
                         f"; KNOWN VEHICLE: {known.description}" if known else "")
        return image

    def run(self, source: VideoSource, writer, display=False, max_frames=None, viewer=None):
        if viewer is None and display:
            from .viewer import BeforeAfterViewer
            viewer = BeforeAfterViewer()
        count = 0
        processing_fps = 0.0
        for frame in source:
            started = time.perf_counter()
            image = self.process(frame)
            elapsed = time.perf_counter() - started
            instant_fps = 1.0 / max(elapsed, 1e-9)
            processing_fps = instant_fps if count == 0 else 0.15 * instant_fps + 0.85 * processing_fps
            writer.write(image)
            count += 1
            if count % 100 == 0:
                LOG.info("Processed %d frames (video %.2fs)", count, frame.timestamp)
            if display:
                # self.process creates the annotation copy; inference is never repeated here.
                if not viewer.show(frame.image, image, frame.index, processing_fps):
                    break
            if max_frames is not None and count >= max_frames:
                break
        LOG.info("Finished: %d frames", count)
        return count
