"""Baseline plate candidates and interchangeable Tesseract OCR adapter."""
import re
import shutil
import cv2
from .models import Box, LocalPlateCandidate, PlateRead


def normalize_plate(text: str) -> str:
    """Phase 0 supports Latin A-Z / 0-9 plates, without guessing O/0 or I/1."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class OpenCVContourPlateDetector:
    """Experimental contour baseline, not a trained plate detector.

    It scores wide rectangular contours inside a vehicle crop. Its score is a
    heuristic, not a model confidence. Rejected candidates are deliberately
    retained for debug rendering, helping compare this baseline with a future
    model-backed detector.
    """
    def detect_candidates(self, vehicle_image):
        if vehicle_image.size == 0:
            return []
        gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[LocalPlateCandidate] = []
        total = gray.shape[0] * gray.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 12 or h < 5:
                continue
            ratio, area = w / h, w * h
            rectangularity = min(1.0, cv2.contourArea(contour) / area)
            vertical_position = (y + h / 2) / gray.shape[0]
            reason = ""
            # Broad vehicle-relative limits: these reject obvious trim while
            # allowing distant, angled, small and front/rear plate locations.
            if not 1.5 <= ratio <= 8.0:
                reason = "aspect"
            elif not 0.0004 <= area / total <= 0.30:
                reason = "area"
            elif not 0.12 <= vertical_position <= 0.98:
                reason = "location"
            elif rectangularity < 0.35:
                reason = "shape"
            aspect_score = max(0.0, 1.0 - abs(ratio - 4.0) / 4.0)
            location_score = max(0.0, 1.0 - abs(vertical_position - 0.65) / 0.65)
            score = 0.55 * rectangularity + 0.30 * aspect_score + 0.15 * location_score
            candidates.append(LocalPlateCandidate(Box(x, y, x + w, y + h), score,
                                                   not reason, reason))
        accepted = sorted((c for c in candidates if c.accepted), key=lambda c: c.confidence, reverse=True)
        rejected = sorted((c for c in candidates if not c.accepted), key=lambda c: c.confidence, reverse=True)
        # Limit accepted OCR work while retaining useful debug evidence.
        return accepted[:3] + rejected[:8]


# Compatibility alias for callers of the original Phase 0 implementation.
OpenCVPlateDetector = OpenCVContourPlateDetector


class TesseractPlateReader:
    def __init__(self):
        if not shutil.which("tesseract"):
            raise RuntimeError("OCR requires Tesseract: sudo apt install tesseract-ocr tesseract-ocr-eng")
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Install OCR support: python -m pip install -e '.[ocr]'") from exc
        self.engine = pytesseract

    def read(self, plate_image):
        if plate_image.size == 0:
            return None
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, 80 / gray.shape[0])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        data = self.engine.image_to_data(
            binary, lang="eng", config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            output_type=self.engine.Output.DICT, timeout=5,
        )
        words, scores = [], []
        for text, confidence in zip(data["text"], data["conf"]):
            normalized = normalize_plate(text)
            score = float(confidence)
            if normalized and score >= 0:
                words.append(normalized)
                scores.append((max(0, min(1, score / 100)), len(normalized)))
        if not words:
            return None
        return PlateRead("".join(words), sum(s * n for s, n in scores) / sum(n for _, n in scores))
