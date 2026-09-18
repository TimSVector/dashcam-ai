"""ctypes wrapper around roadcam_core; no OpenCV or inference dependencies."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

from roadcam.models import Box

_PLATE_CAPACITY = 64


class _BoundingBox(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int)]


class _VehicleDetection(ctypes.Structure):
    _fields_ = [
        ("vehicle_type", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("bounding_box", _BoundingBox),
        ("frame_number", ctypes.c_ulonglong),
        ("timestamp_seconds", ctypes.c_double),
    ]


class _Result(ctypes.Structure):
    _fields_ = [("accepted", ctypes.c_int), ("known_vehicle_id", ctypes.c_int),
                ("normalized_plate", ctypes.c_char * _PLATE_CAPACITY)]


_VEHICLE_TYPES = {"car": 1, "truck": 2, "bus": 3, "motorcycle": 4}


@dataclass(frozen=True)
class NativeDecision:
    accepted: bool
    normalized_plate: str
    known_vehicle_id: int | None


def default_library_path() -> Path:
    """Find the normal local CMake output or honor an explicit override."""
    override = os.environ.get("ROADCAM_CORE_LIBRARY")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]
    names = ("libroadcam_core.so", "libroadcam_core.dylib", "roadcam_core.dll")
    for name in names:
        candidate = root / "build" / "native" / name
        if candidate.is_file():
            return candidate
    return root / "build" / "native" / names[0]


class NativeCore:
    def __init__(self, minimum_confidence: float, library_path: str | Path | None = None):
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("Native confidence must be between 0 and 1")
        path = Path(library_path) if library_path else default_library_path()
        if not path.is_file():
            raise RuntimeError(f"Native core library not found: {path}. Build it with: cmake -S . -B build && cmake --build build")
        self.library = ctypes.CDLL(str(path))
        self.library.roadcam_create.argtypes = [ctypes.c_float]
        self.library.roadcam_create.restype = ctypes.c_void_p
        self.library.roadcam_destroy.argtypes = [ctypes.c_void_p]
        self.library.roadcam_destroy.restype = None
        self.library.roadcam_process_detection.argtypes = [ctypes.c_void_p, ctypes.POINTER(_VehicleDetection),
                                                            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint]
        self.library.roadcam_process_detection.restype = _Result
        self.handle = self.library.roadcam_create(minimum_confidence)
        if not self.handle:
            raise RuntimeError("Native core rejected the confidence threshold")

    def decide_vehicle(self, label: str, confidence: float, box: Box, frame_number: int, timestamp: float) -> NativeDecision:
        vehicle_type = _VEHICLE_TYPES.get(label, 0)
        detection = _VehicleDetection(vehicle_type, confidence,
                                      _BoundingBox(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                                      frame_number, timestamp)
        result = self.library.roadcam_process_detection(self.handle, ctypes.byref(detection), None, None, 0)
        return NativeDecision(bool(result.accepted), result.normalized_plate.decode("ascii"),
                              result.known_vehicle_id or None)

    def close(self) -> None:
        if self.handle:
            self.library.roadcam_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
