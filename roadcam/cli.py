import argparse
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import cv2
from .detection import TorchvisionVehicleDetector
from .pipeline import Pipeline
from .video import OpenCVVideoWriter, VideoFileSource
from .plates import OpenCVContourPlateDetector, TesseractPlateReader
from .storage import SQLiteEventLogger, SQLiteVehicleDatabase
from .native import NativeCore


def probability(value):
    result = float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU vehicle detection on MP4 video")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plates", action="store_true", help="Enable experimental plate detection and OCR")
    parser.add_argument("--plate-debug", action="store_true",
                        help="Draw accepted and rejected contour plate candidates with scores/reasons")
    parser.add_argument("--database", type=Path, default=Path("roadcam.sqlite3"))
    parser.add_argument("--ocr-confidence", type=probability, default=0.5)
    parser.add_argument("--add-known", nargs=2, metavar=("PLATE", "DESCRIPTION"))
    parser.add_argument("--notes", default="")
    parser.add_argument("--list-known", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--confidence", type=probability, default=0.5)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--native-library", type=Path, help="Path to the built roadcam_core shared library")
    parser.add_argument("--max-frames", type=positive)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    management = args.add_known is not None or args.list_known
    if management and (args.input or args.output):
        parser.error("Use database management separately from video processing")
    if not management and (args.input is None or args.output is None):
        parser.error("Video processing requires --input and --output")
    if args.plate_debug and not args.plates:
        parser.error("--plate-debug requires --plates")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if management:
            with ExitStack() as stack:
                database = SQLiteVehicleDatabase(args.database)
                stack.callback(database.close)
                if args.add_known:
                    vehicle = database.add(*args.add_known, notes=args.notes)
                    print(f"Saved {vehicle.plate}: {vehicle.description} (id={vehicle.id})")
                if args.list_known:
                    for vehicle in database.list():
                        print(f"{vehicle.id}\t{vehicle.plate}\t{vehicle.description}\t{vehicle.notes}")
            return 0
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        if args.plates and args.database.resolve() in {args.input.resolve(), args.output.resolve()}:
            raise ValueError("Database must differ from input and output video")
        if args.display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise ValueError("--display needs WSLg or an X server; omit it for headless processing")
        with ExitStack() as stack:
            source = VideoFileSource(args.input)
            stack.callback(source.close)
            stages = {}
            if args.plates:
                reader = TesseractPlateReader()
                database = SQLiteVehicleDatabase(args.database)
                stack.callback(database.close)
                event_logger = SQLiteEventLogger(args.database)
                stack.callback(event_logger.close)
                stages = dict(plate_detector=OpenCVContourPlateDetector(), plate_reader=reader,
                              database=database, event_logger=event_logger,
                              ocr_confidence=args.ocr_confidence, source_name=str(args.input.resolve()),
                              plate_debug=args.plate_debug)
            detector = TorchvisionVehicleDetector(args.confidence, args.threads)
            native_core = NativeCore(args.confidence, args.native_library)
            stack.callback(native_core.close)
            writer = OpenCVVideoWriter(args.output, source.fps, source.width, source.height)
            stack.callback(writer.close)
            if args.display:
                # Pipeline owns the viewer; retain this broad fallback for an interrupted GUI.
                stack.callback(cv2.destroyAllWindows)
            Pipeline(detector, native_core=native_core, **stages).run(source, writer, args.display, args.max_frames)
        return 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted; output may contain only processed frames")
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc, exc_info=args.verbose)
        return 1
