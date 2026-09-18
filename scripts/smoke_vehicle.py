"""Exercise real CPU inference and MP4 I/O. Optionally supply a vehicle photo."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from roadcam.detection import TorchvisionVehicleDetector
from roadcam.pipeline import Pipeline
from roadcam.video import VideoFileSource, OpenCVVideoWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/vehicle-smoke"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.image)) if args.image else np.zeros((320, 480, 3), np.uint8)
    if image is None:
        raise ValueError("Could not load smoke-test image")
    image = cv2.resize(image, (640, 480))
    input_path = args.directory / "input.mp4"
    output_path = args.directory / "annotated.mp4"
    writer = OpenCVVideoWriter(input_path, 5, 640, 480)
    try:
        for _ in range(3):
            writer.write(image)
    finally:
        writer.close()
    detector = TorchvisionVehicleDetector()
    source = VideoFileSource(input_path)
    output = OpenCVVideoWriter(output_path, source.fps, source.width, source.height)
    try:
        assert Pipeline(detector).run(source, output) == 3
    finally:
        source.close()
        output.close()
    check = VideoFileSource(output_path)
    try:
        frames = list(check)
        assert len(frames) == 3
        assert frames[0].image.shape == image.shape
    finally:
        check.close()
    detections = detector.detect(image)
    if args.image:
        assert detections, "Provided image must contain a detectable vehicle"
    print(f"PASS: CPU inference, 3 MP4 frames; detections: {detections}; output: {output_path}")


if __name__ == "__main__":
    main()
