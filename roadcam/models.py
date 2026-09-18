"""Shared records. Images are OpenCV BGR arrays; boxes use exclusive x2/y2."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def clipped(self, width: int, height: int) -> "Box":
        return Box(max(0, min(self.x1, width)), max(0, min(self.y1, height)),
                   max(0, min(self.x2, width)), max(0, min(self.y2, height)))

    def crop(self, image: np.ndarray) -> np.ndarray:
        b = self.clipped(image.shape[1], image.shape[0])
        return image[b.y1:b.y2, b.x1:b.x2]


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    index: int
    timestamp: float  # seconds from start of video


@dataclass(frozen=True)
class Detection:
    box: Box
    label: str
    confidence: float


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float  # 0..1, engine score (not calibrated probability)


@dataclass(frozen=True)
class LocalPlateCandidate:
    """A plate proposal relative to its parent vehicle crop."""
    box: Box
    confidence: float
    accepted: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class PlateCandidate:
    """A plate proposal in frame coordinates, explicitly tied to a vehicle."""
    box: Box
    confidence: float
    parent_vehicle_index: int
    parent_vehicle_type: str
    accepted: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class KnownVehicle:
    id: int
    plate: str
    description: str
    notes: str


@dataclass(frozen=True)
class DetectionEvent:
    timestamp: float
    detected_plate: str
    ocr_confidence: float
    vehicle_type: str
    known_vehicle_id: int | None
    source: str = ""
    frame_index: int = 0
