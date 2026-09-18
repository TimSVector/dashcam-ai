import numpy as np
import pytest
from roadcam.models import Box, Detection, Frame
from roadcam.pipeline import Pipeline
from roadcam.video import OpenCVVideoWriter, VideoFileSource
from roadcam.cli import main


class FakeDetector:
    def detect(self, image):
        return [Detection(Box(20, 30, 90, 70), "car", 0.9)]


def test_annotation_preserves_input():
    raw = np.zeros((100, 160, 3), np.uint8)
    output = Pipeline(FakeDetector()).process(Frame(raw, 0, 0))
    assert not raw.any()
    assert output[30, 20].any()


def test_video_roundtrip(tmp_path):
    path = tmp_path / "input.mp4"
    writer = OpenCVVideoWriter(path, 10, 160, 100)
    for _ in range(3):
        writer.write(np.zeros((100, 160, 3), np.uint8))
    writer.close()
    source = VideoFileSource(path)
    try:
        frames = list(source)
        assert len(frames) == 3
        assert [f.timestamp for f in frames] == pytest.approx([0, .1, .2])
        output = OpenCVVideoWriter(tmp_path / "out.mp4", 10, 160, 100)
        try:
            assert Pipeline(FakeDetector()).run(frames, output) == 3
        finally:
            output.close()
    finally:
        source.close()
    check = VideoFileSource(tmp_path / "out.mp4")
    try:
        assert len(list(check)) == 3
    finally:
        check.close()


def test_invalid_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_text("invalid")
    with pytest.raises(ValueError):
        VideoFileSource(bad)


def test_cli_prevents_using_input_as_output(tmp_path):
    path = tmp_path / "existing.mp4"
    path.write_bytes(b"keep")
    assert main(["--input", str(path), "--output", str(path)]) == 1
    assert path.read_bytes() == b"keep"


def test_clip():
    image = np.zeros((10, 20, 3), np.uint8)
    assert Box(-5, -4, 30, 40).crop(image).shape == image.shape


def test_cli_releases_resources_after_failure(tmp_path, monkeypatch):
    from roadcam import cli
    closed = []
    class Source:
        fps, width, height = 10, 160, 100
        def __init__(self, path):
            pass
        def __iter__(self):
            yield Frame(np.zeros((100, 160, 3), np.uint8), 0, 0)
        def close(self):
            closed.append("source")
    class Writer:
        def __init__(self, *args):
            pass
        def write(self, image):
            raise RuntimeError("write failed")
        def close(self):
            closed.append("writer")
    monkeypatch.setattr(cli, "VideoFileSource", Source)
    monkeypatch.setattr(cli, "OpenCVVideoWriter", Writer)
    monkeypatch.setattr(cli, "TorchvisionVehicleDetector", lambda *args: FakeDetector())
    assert main(["--input", str(tmp_path / "in.mp4"), "--output", str(tmp_path / "out.mp4")]) == 1
    assert closed == ["writer", "source"]
