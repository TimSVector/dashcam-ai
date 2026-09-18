import sqlite3
import numpy as np
import pytest
import cv2
from roadcam.models import Box, Detection, DetectionEvent, Frame, LocalPlateCandidate, PlateRead
from roadcam.plates import normalize_plate, OpenCVPlateDetector, TesseractPlateReader
from roadcam.pipeline import Pipeline
from roadcam.storage import SQLiteVehicleDatabase, SQLiteEventLogger


@pytest.mark.parametrize("raw,expected", [(" ab-c 1234!", "ABC1234"), ("...", ""), ("o0-i1", "O0I1")])
def test_normalize(raw, expected):
    assert normalize_plate(raw) == expected


def test_database_normalization_upsert_and_events(tmp_path):
    path = tmp_path / "vehicles.sqlite3"
    db = SQLiteVehicleDatabase(path)
    logger = SQLiteEventLogger(path)
    try:
        vehicle = db.add("abc-1234", "Sister's car", "blue")
        assert db.lookup(" ABC 1234 ") == vehicle
        assert db.lookup("UNKNOWN") is None
        assert db.add("ABC1234", "Updated").id == vehicle.id
        assert len(db.list()) == 1
        with pytest.raises(ValueError):
            db.add("---", "Invalid")
        logger.log(DetectionEvent(1.25, "abc-1234", .9, "car", vehicle.id, "test.mp4", 30))
        logger.log(DetectionEvent(2, "OTHER", .8, "truck", None))
    finally:
        logger.close()
        db.close()
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT timestamp, detected_plate, ocr_confidence, vehicle_type, known_vehicle_id FROM detection_events ORDER BY id").fetchall()
        assert rows == [(1.25, "ABC1234", .9, "car", vehicle.id), (2, "OTHER", .8, "truck", None)]


class Vehicles:
    def detect(self, image):
        return [Detection(Box(10, 20, 190, 100), "car", .9)]


class Plates:
    def detect_candidates(self, image):
        assert image.shape == (80, 180, 3)
        return [LocalPlateCandidate(Box(20, 20, 140, 60), .8, True)] * 2


class Reader:
    def __init__(self, result):
        self.result = result

    def read(self, image):
        assert image.shape == (40, 120, 3)
        assert not image.any()  # annotations never contaminate OCR input
        return self.result


@pytest.mark.parametrize("read,expected", [(PlateRead("abc-1234", .9), 1),
                                         (PlateRead("abc", .1), 0), (None, 0),
                                         (PlateRead("--", .9), 0)])
def test_pipeline_lookup_and_confidence(tmp_path, read, expected, monkeypatch):
    path = tmp_path / "test.sqlite3"
    db, logger = SQLiteVehicleDatabase(path), SQLiteEventLogger(path)
    texts = []
    from roadcam import pipeline
    original = pipeline.annotate
    def capture(image, box, text, color=(0, 220, 0)):
        texts.append(text)
        original(image, box, text, color)
    monkeypatch.setattr(pipeline, "annotate", capture)
    try:
        known = db.add("ABC1234", "Sister's car")
        p = Pipeline(Vehicles(), Plates(), Reader(read), db, logger, source_name="drive.mp4")
        raw = np.zeros((120, 200, 3), np.uint8)
        image = p.process(Frame(raw, 25, 2.5))
        rows = db.connection.execute("SELECT timestamp, known_vehicle_id, frame_index, source FROM detection_events").fetchall()
        assert len(rows) == expected
        if expected:
            assert rows[0] == (2.5, known.id, 25, "drive.mp4")
            assert any("KNOWN VEHICLE: Sister's car" in t for t in texts)
            assert image[40, 30].any()  # crop-relative box translated to frame
        assert not raw.any()
    finally:
        logger.close()
        db.close()


def test_plate_candidates():
    image = np.zeros((180, 400, 3), np.uint8)
    cv2.rectangle(image, (100, 100), (280, 145), (255, 255, 255), -1)
    candidates = OpenCVPlateDetector().detect_candidates(image)
    accepted = [candidate for candidate in candidates if candidate.accepted]
    assert accepted
    assert any(d.box.x1 <= 100 and d.box.x2 >= 280 for d in accepted)
    assert OpenCVPlateDetector().detect_candidates(np.zeros_like(image)) == []


def test_rejected_candidates_are_not_sent_to_ocr_and_can_be_debugged(tmp_path, monkeypatch):
    class Candidates:
        def detect_candidates(self, image):
            return [
                LocalPlateCandidate(Box(10, 10, 60, 25), .7, True),
                LocalPlateCandidate(Box(70, 10, 90, 70), .2, False, "aspect"),
            ]

    class NeverCalledReader:
        def __init__(self):
            self.calls = 0

        def read(self, image):
            self.calls += 1
            return None

    annotated = []
    associated = []
    from roadcam import pipeline
    original = pipeline.annotate
    original_candidate = pipeline.annotate_plate_candidate
    monkeypatch.setattr(pipeline, "annotate", lambda image, box, text, color=(0, 220, 0):
                        (annotated.append(text), original(image, box, text, color)))
    monkeypatch.setattr(pipeline, "annotate_plate_candidate",
                        lambda image, candidate, debug: (associated.append(candidate),
                                                         original_candidate(image, candidate, debug)))
    path = tmp_path / "test.sqlite3"
    db, logger, reader = SQLiteVehicleDatabase(path), SQLiteEventLogger(path), NeverCalledReader()
    try:
        Pipeline(Vehicles(), Candidates(), reader, db, logger, plate_debug=True).process(
            Frame(np.zeros((120, 200, 3), np.uint8), 1, .1))
        assert reader.calls == 1
        assert "plate 0.70" in annotated
        assert "reject: aspect" in annotated
        assert [candidate.parent_vehicle_index for candidate in associated] == [0, 0]
        assert all(candidate.parent_vehicle_type == "car" for candidate in associated)
    finally:
        logger.close()
        db.close()


def test_reader_word_confidence():
    # Avoid requiring the binary in hardware/model-free tests.
    reader = TesseractPlateReader.__new__(TesseractPlateReader)
    class Engine:
        class Output:
            DICT = "dict"
        def image_to_data(self, *args, **kwargs):
            return {"text": ["", "ABC", "1234"], "conf": [-1, 80, 90]}
    reader.engine = Engine()
    result = reader.read(np.full((30, 150, 3), 255, np.uint8))
    assert result.text == "ABC1234"
    assert result.confidence == pytest.approx((.8 * 3 + .9 * 4) / 7)


def test_partial_pipeline_rejected():
    with pytest.raises(ValueError):
        Pipeline(Vehicles(), plate_detector=Plates())
