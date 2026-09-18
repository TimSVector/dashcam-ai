"""OpenCV presentation adapter; replaceable by a future browser/PWA viewer."""
from __future__ import annotations

import cv2
import numpy as np


class BeforeAfterViewer:
    """Side-by-side original/processed frames with non-destructive status text."""

    def __init__(self, window_name: str = "roadcam-ai — q/Esc quit, Space pause"):
        self.window_name = window_name
        self.paused = False

    @staticmethod
    def compose(original: np.ndarray, processed: np.ndarray, frame_number: int,
                processing_fps: float) -> np.ndarray:
        """Build a presentation canvas without modifying either supplied frame."""
        if original.shape != processed.shape:
            raise ValueError("Original and processed frames must have identical dimensions")
        height, width = original.shape[:2]
        header = np.full((42, width * 2, 3), 28, dtype=np.uint8)
        cv2.putText(header, "ORIGINAL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(header, "PROCESSED", (width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 230, 80), 2, cv2.LINE_AA)
        text = f"Frame: {frame_number}    Processing FPS: {processing_fps:.1f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(header, text, (width * 2 - text_size[0] - 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return np.vstack((header, np.hstack((original, processed))))

    @staticmethod
    def _key_action(key: int) -> str | None:
        key &= 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord(" "):
            return "pause"
        return None

    def show(self, original: np.ndarray, processed: np.ndarray, frame_number: int,
             processing_fps: float) -> bool:
        """Show the current pair. Returns False when the user chooses to quit."""
        canvas = self.compose(original, processed, frame_number, processing_fps)
        cv2.imshow(self.window_name, canvas)
        while True:
            action = self._key_action(cv2.waitKey(30 if self.paused else 1))
            if action == "quit":
                return False
            if action == "pause":
                self.paused = not self.paused
            if not self.paused:
                return True

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
