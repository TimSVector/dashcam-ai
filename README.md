# roadcam-ai

Phase 0: Python 3 CPU processing of prerecorded MP4 video on Ubuntu/WSL2.
No Raspberry Pi or Hailo dependencies.

## Milestone 1 architecture

The project keeps probabilistic perception separate from deterministic policy:

```text
OpenCV video + TorchVision AI (Python) → roadcam_core (portable C++17) → OpenCV output (Python)
```

Python captures frames and obtains vehicle candidates. `roadcam_core` receives
plain C data only: a vehicle type, confidence, frame/timestamp, and an `x/y/
width/height` bounding box. It deterministically validates the box, applies the
confidence threshold, normalizes a supplied plate, and can exactly match that
plate against supplied known-vehicle records. It has no dependency on Python,
OpenCV, models, SQLite, hardware, network access, or Linux APIs. The C ABI is
declared in `native/include/roadcam_core.h`; `roadcam/native/core.py` is the
only ctypes integration point.

The existing optional plate/OCR/SQLite prototype is retained for later work. It
is outside Milestone 1 and is not part of the native demonstration.

## Build the native core

Install a C++17 compiler, CMake, and Python dependencies as described below,
then build and test the portable core:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The default Python wrapper loads `build/native/libroadcam_core.so` on Linux. Set
`ROADCAM_CORE_LIBRARY` or pass `--native-library /path/to/library` to override
that location. The core is intentionally simple enough to compile and unit-test
in a VectorCAST environment without adding VectorCAST dependencies.

## Architecture

`VideoSource → VehicleDetector → PlateDetector → PlateReader → VehicleDatabase`

The pipeline draws annotated frames and sends recognition records to `EventLogger`.
`interfaces.py` defines structural interfaces; `models.py` contains plain records.
Images are BGR NumPy arrays. Vehicle boxes are frame-relative; plate boxes are
vehicle-crop-relative. Coordinates use exclusive right/bottom edges. Confidence
scores are 0–1. The pipeline owns no model or hardware implementation.

- `video.py`: OpenCV file input and MP4 output.
- `detection.py`: CPU TorchVision SSDLite320 MobileNetV3, COCO pretrained.
- `pipeline.py`: orchestration and annotation.
- `cli.py`: constructs adapters and owns resource lifetime.
- `plates.py`: separate contour-based plate detector and Tesseract OCR adapter.
- `storage.py`: SQLite known-vehicle database and event logger.

Future PiCameraSource, HailoVehicleDetector, and HailoPlateDetector can implement
the same contracts. GPS, hotspot, browser/PWA, and loop recording are deferred.
A future web consumer can use the event records without coupling inference to HTTP.

## Model licensing

TorchVision code is BSD-3-Clause, but pretrained models may have additional terms
derived from training data. This prototype uses COCO-trained SSDLite weights,
downloaded separately, not bundled. Review model/dataset terms before redistributing
weights or a product; the library license alone is not clearance for the weights.
See https://github.com/pytorch/vision/blob/main/LICENSE and
https://github.com/pytorch/vision#pre-trained-model-license.

## Ubuntu / WSL2 installation

Use Python 3.10+ (Ubuntu 24.04 Python 3.12 is suitable). Keep the project on the
WSL Linux filesystem for better I/O performance. In Ubuntu:

```bash
sudo apt update
sudo apt install python3-venv libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install this first so the project uses CPU wheels, not CUDA dependencies.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,ocr]'
```

Tesseract and the `ocr` extra are only needed for OCR. Do not install
`opencv-python-headless` alongside `opencv-python`. `--display` requires working
WSLg or an X server. Without it, processing works headlessly.

```bash
python -m roadcam --input test_drive.mp4 --output annotated.mp4
python -m roadcam --input test_drive.mp4 --output preview.mp4 --display --max-frames 100
python -m pytest
python scripts/smoke_vehicle.py
```

Build the native core first. The regular MP4 command now calls it once for each
AI vehicle candidate before annotations are drawn:

```bash
cmake -S . -B build && cmake --build build
python -m roadcam --input test_drive.mp4 --output annotated.mp4 --max-frames 100
python -m pytest -q
```

SSDLite's first run downloads approximately 14 MB of weights into Torch's cache.
For a project-local cache use `export TORCH_HOME="$PWD/.cache/torch"`. Subsequent
runs can be offline. The smoke script builds a three-frame MP4 and runs actual CPU
inference; `--image /path/to/vehicle.jpg` additionally requires a positive detection.
Unit tests use fake inference and do not download models.

`--confidence 0.5` controls vehicle filtering and `--threads 4` limits CPU threads.
Existing output files are overwritten; parent directories must already exist. MP4 output uses
`mp4v`, requires even dimensions, and has no audio. Display runs at processing
speed in a side-by-side viewer: the left panel is the original unmodified frame
and the right panel is the same frame with detections. A header shows frame number
and processing/inference FPS. Press **q** or **Esc** to stop; press **Space** to
pause or resume. The viewer receives the source frame and the annotation copy
after a single inference pass, and is isolated in `roadcam/viewer.py` so it can
later be replaced by a browser/PWA interface. Decoded timestamps are used when available, with an
index/FPS fallback. Output uses the input nominal FPS, so variable-frame-rate
sources are not timing-perfect. A failed/interrupted run may leave partial output.

## Plate recognition and known vehicles

Vehicle-only processing is the default. Enable the full experimental pipeline with
`--plates`. Register known vehicles before processing; no example rows are seeded
automatically. All commands default to `roadcam.sqlite3` in the current directory.

```bash
python -m roadcam --add-known ABC1234 "Sister's car" --notes "Blue sedan"
python -m roadcam --list-known
python -m roadcam --input test_drive.mp4 --output recognized.mp4 --plates
# A separate database and a stricter OCR threshold:
python -m roadcam --input test_drive.mp4 --output strict.mp4 --plates \
  --database vehicles.sqlite3 --ocr-confidence 0.8
```

`--add-known` updates description/notes if the normalized plate already exists,
retaining its ID. Normalization uppercases and keeps ASCII A–Z/0–9 only. Spaces
and punctuation are removed. No fuzzy matching or automatic O/0 substitutions
are used. Non-Latin plates need a future normalization/OCR adapter.

`plates.py` implements two independent adapters: `OpenCVContourPlateDetector` finds wide
rectangular contours within each vehicle crop, and `TesseractPlateReader` enlarges
and thresholds plate crops before OCR. **This is a baseline plate heuristic, not a
trained plate model.** Small, skewed, obscured and two-line plates can be missed;
grilles and signs can be false positives. OCR confidence is an engine score, not
a calibrated probability. Even a high score can be wrong. Validate on your own
footage before relying on matches. Vehicle detection itself can also miss small
or occluded vehicles.

Plate localization is vehicle-aware: the pipeline runs the plate detector only
inside each vehicle box, then creates a `PlateCandidate` with its frame-relative
plate box, detector score, parent vehicle index, and parent vehicle type. OCR only
receives accepted candidates. The present contour detector is retained as a debug
baseline; it scores contour rectangularity, aspect ratio, and broad
vehicle-relative location. That score is not a learned probability.

Use `--plate-debug` together with `--plates` to show accepted candidates in yellow
with their score and rejected candidates in red with the rule that rejected them:

```bash
python -m roadcam --input test_drive.mp4 --output plate-debug.mp4 --plates --plate-debug
```

The next detector should be a model-backed implementation of the `TrainedPlateDetector`
interface, preferably a CPU ONNX model for Phase 0 and a `HailoPlateDetector` using
the same interface later. Hailo lists its single-class `tiny_yolov4_license_plates`
model as a Hailo-10H-targeted option, but its listed in-house model weights require
separate redistribution approval; the MIT license of the Model Zoo source code does
not itself grant that approval. No trained plate-model weights are bundled or loaded
by this project yet.

Accepted OCR reads are logged to SQLite `detection_events`, including `timestamp`
(seconds within input video), normalized `detected_plate`, `ocr_confidence`,
`vehicle_type`, nullable `known_vehicle_id`, `source`, `frame_index`, and UTC
`created_at`. Unmatched reads are logged too; unreadable or below-threshold
candidates are not. One plate is logged once per vehicle per frame; there is no
tracking or cross-frame deduplication yet, so repeated observations are expected.
Each event is committed immediately. Known matches show
`KNOWN VEHICLE: <description>` on the annotated video.

For example, inspect events using Python's built-in SQLite support:

```bash
python - <<'PY'
import sqlite3
with sqlite3.connect('roadcam.sqlite3') as db:
    for row in db.execute('SELECT timestamp, detected_plate, vehicle_type, known_vehicle_id FROM detection_events ORDER BY id'):
        print(row)
PY
```

No browser interface, Pi capture, Hailo inference, GPS, hotspot or loop recording
is implemented. SQLite event records and the `EventLogger` interface provide a
boundary for future consumers.

## Validation and limitations

```bash
python -m pytest -q
python scripts/smoke_vehicle.py --image /path/to/vehicle.jpg
# Requires Tesseract plus DejaVu fonts (Ubuntu: sudo apt install fonts-dejavu-core):
python scripts/smoke_plate.py
```

The initial vehicle-only milestone passed five tests before plate/OCR work began.
A real-model smoke run detected a bus at 0.974 confidence and processed/decoded a
three-frame MP4. The local smoke photo came from
https://github.com/ultralytics/assets/blob/main/im/bus.jpg and is not bundled as a
project asset. No Ultralytics library or model is used.

The full pipeline passed unit/integration tests and a separate real Tesseract
smoke test: generated `ABC1234` → known vehicle ID → SQLite event (OCR score 0.92).
The OCR smoke uses a fixture vehicle detector, while the vehicle smoke uses real
SSDLite inference. The CLI with `--plates` also processed the bus MP4. These are
integration checks, not road-video accuracy benchmarks. An initial synthetic
OpenCV-font plate misread `3` as `5`; a conventional DejaVu-font fixture is used
for the repeatable OCR smoke. GUI display has not been verified in this session.

Tesseract and pytesseract use Apache-2.0 licenses; preserve the relevant notices
when redistributing them:
https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE and
https://github.com/madmaze/pytesseract/blob/master/LICENSE.
System language data and other transitive dependencies retain their own terms.
See `requirements-tested.txt` for the Python environment used during validation;
the CPU PyTorch install command above remains important when recreating it.

## Real moving-traffic sample

A 60-second dashcam sample is available locally as `test_drive.mp4`, with a
processed H.264 preview in `traffic-annotated.mp4`. It shows an actual drive on
I-270 in Maryland, at 960×540 and 15 fps for manageable CPU processing. See
[SAMPLE_VIDEO.md](SAMPLE_VIDEO.md) for source attribution, CC BY-SA 4.0 media
licensing, and commands. The large media files are ignored by Git.

If Windows Media Player fails to open videos from the WSL network path, copy
them to a Windows folder first (for example Downloads).

A second local sample, `city-drive.mp4`, contains 60 seconds of city driving
from Pexels (German Korb). Its processed preview is `city-annotated.mp4`. Source
and the separate Pexels media license are recorded in [SAMPLE_VIDEO.md](SAMPLE_VIDEO.md).
