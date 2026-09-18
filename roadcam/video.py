from pathlib import Path
import math
import cv2
from .models import Frame

class VideoFileSource:
    def __init__(self, path: str | Path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {path}")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.close()
            raise ValueError(f"Cannot open video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(self.fps) or self.fps <= 0 or min(self.width, self.height) <= 0:
            self.close()
            raise ValueError("Video has invalid FPS or frame dimensions")
        self.expected_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        index, previous = 0, -1.0
        while True:
            ok, image = self.capture.read()
            if not ok:
                if index == 0:
                    raise ValueError("Video contains no decodable frames")
                if self.expected_frames > 0 and index < self.expected_frames:
                    raise RuntimeError(f"Video decoding stopped early at frame {index}/{self.expected_frames}")
                break
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                timestamp = max(index / self.fps, previous + 1 / self.fps)
            yield Frame(image, index, timestamp)
            previous = timestamp
            index += 1

    def close(self):
        self.capture.release()


class OpenCVVideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        # mp4v silently truncates odd dimensions in common FFmpeg builds.
        if width % 2 or height % 2:
            raise ValueError("MP4 output requires even frame dimensions")
        self.size = (width, height)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        if not self.writer.isOpened():
            self.close()
            raise RuntimeError(f"Cannot create output video: {path} (check directory and codec support)")

    def write(self, image):
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError("Frame dimensions changed during processing")
        self.writer.write(image)

    def close(self):
        self.writer.release()
