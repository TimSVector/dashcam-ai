import numpy as np

from roadcam.viewer import BeforeAfterViewer
from roadcam.models import Box, Detection, Frame
from roadcam.pipeline import Pipeline


def test_compose_preserves_original_and_processed_frames():
    original = np.zeros((80, 120, 3), dtype=np.uint8)
    processed = original.copy()
    processed[30, 40] = (0, 220, 0)
    canvas = BeforeAfterViewer.compose(original, processed, 7, 12.3)
    assert canvas.shape == (122, 240, 3)
    assert not original.any()
    assert canvas[42 + 30, 40].tolist() == [0, 0, 0]
    assert canvas[42 + 30, 120 + 40].tolist() == [0, 220, 0]


def test_keyboard_controls():
    assert BeforeAfterViewer._key_action(ord("q")) == "quit"
    assert BeforeAfterViewer._key_action(27) == "quit"
    assert BeforeAfterViewer._key_action(ord(" ")) == "pause"
    assert BeforeAfterViewer._key_action(ord("x")) is None


def test_display_pipeline_runs_inference_once_and_shows_original_and_annotation():
    class Detector:
        calls = 0

        def detect(self, image):
            self.calls += 1
            return [Detection(Box(10, 10, 40, 40), "car", 0.9)]

    class Writer:
        def __init__(self):
            self.frames = []

        def write(self, image):
            self.frames.append(image)

    class Viewer:
        def __init__(self):
            self.calls = []

        def show(self, original, processed, frame_number, processing_fps):
            self.calls.append((original.copy(), processed.copy(), frame_number, processing_fps))
            return True

    detector, writer, viewer = Detector(), Writer(), Viewer()
    raw = np.zeros((60, 80, 3), dtype=np.uint8)
    processed = Pipeline(detector).run([Frame(raw, 4, 0.4)], writer, display=True, viewer=viewer)
    assert processed == 1
    assert detector.calls == 1
    original, annotated, frame_number, processing_fps = viewer.calls[0]
    assert not original.any()
    assert annotated[10, 10].any()
    assert frame_number == 4
    assert processing_fps > 0
