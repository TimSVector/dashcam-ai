"""Real contour detection + Tesseract + SQLite, using a synthetic vehicle crop."""
from pathlib import Path
import sqlite3
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roadcam.models import Box, Detection, Frame
from roadcam.pipeline import Pipeline
from roadcam.plates import OpenCVPlateDetector, TesseractPlateReader
from roadcam.storage import SQLiteVehicleDatabase, SQLiteEventLogger


def main():
    image = np.zeros((240, 640, 3), np.uint8)
    cv2.rectangle(image, (160, 120), (480, 200), (255, 255, 255), -1)
    canvas = Image.fromarray(image)
    font = ImageFont.truetype("DejaVuSansMono.ttf", 60)
    ImageDraw.Draw(canvas).text((185, 120), "ABC1234", font=font, fill=(0, 0, 0))
    image = np.array(canvas)
    class VehicleFixture:
        def detect(self, image):
            return [Detection(Box(0, 0, 640, 240), "car", 1.0)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.sqlite3"
        db, logger = SQLiteVehicleDatabase(path), SQLiteEventLogger(path)
        try:
            known = db.add("ABC1234", "Sister's car")
            pipeline = Pipeline(VehicleFixture(), OpenCVPlateDetector(), TesseractPlateReader(), db, logger)
            annotated = pipeline.process(Frame(image, 5, 0.5))
            with sqlite3.connect(path) as conn:
                rows = conn.execute("SELECT detected_plate, ocr_confidence, known_vehicle_id FROM detection_events").fetchall()
            assert any(plate == "ABC1234" and identity == known.id for plate, _, identity in rows), rows
            Path("artifacts").mkdir(exist_ok=True)
            cv2.imwrite("artifacts/plate-smoke.jpg", annotated)
            print(f"PASS: real plate detection/OCR, known lookup and event logging: {rows}")
        finally:
            logger.close()
            db.close()


if __name__ == "__main__":
    main()
