"""Structural interfaces; adapters need no hardware-specific base class."""
from typing import Iterator, Protocol
import numpy as np
from .models import (Detection, DetectionEvent, Frame, KnownVehicle,
                     LocalPlateCandidate, PlateRead)


class VideoSource(Protocol):
    fps: float
    width: int
    height: int
    def __iter__(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class VehicleDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class PlateDetector(Protocol):
    """Find plate candidates in one vehicle crop; boxes are crop-relative."""
    def detect_candidates(self, vehicle_image: np.ndarray) -> list[LocalPlateCandidate]: ...


class TrainedPlateDetector(PlateDetector, Protocol):
    """Marker interface for a model-backed CPU/ONNX or future Hailo adapter."""


class PlateReader(Protocol):
    def read(self, plate_image: np.ndarray) -> PlateRead | None: ...


class VehicleDatabase(Protocol):
    def lookup(self, plate: str) -> KnownVehicle | None: ...
    def close(self) -> None: ...


class EventLogger(Protocol):
    def log(self, event: DetectionEvent) -> None: ...
    def close(self) -> None: ...
