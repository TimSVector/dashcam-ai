from roadcam.models import Box
from roadcam.native import NativeCore


def test_native_core_accepts_valid_vehicle_and_rejects_invalid_box():
    with NativeCore(0.5) as core:
        accepted = core.decide_vehicle("car", 0.9, Box(10, 20, 110, 90), 12, 1.2)
        assert accepted.accepted
        rejected = core.decide_vehicle("car", 0.9, Box(10, 20, 10, 90), 12, 1.2)
        assert not rejected.accepted


def test_native_core_applies_confidence_threshold():
    with NativeCore(0.9) as core:
        assert not core.decide_vehicle("truck", 0.89, Box(0, 0, 10, 10), 0, 0).accepted
