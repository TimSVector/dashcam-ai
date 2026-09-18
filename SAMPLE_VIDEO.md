# Moving traffic sample

Source: **Driving southbound on I-270 from Shady Grove Road to the I-270 Spur
(1 June 2026)**, by **Illegitimate Barrister**, Wikimedia Commons.

Source page:
https://commons.wikimedia.org/wiki/File:Driving_southbound_on_I-270_from_Shady_Grove_Road_to_the_I-270_Spur_(1_June_2026).webm

License: **Creative Commons Attribution-ShareAlike 4.0 International**
https://creativecommons.org/licenses/by-sa/4.0/

The source video and derived sample videos retain this license. Include this
attribution and license when sharing them. This license notice concerns the
sample media, not the project's Python code. No endorsement is implied.

Local files:

- `artifacts/traffic/source.webm`: original download.
- `test_drive.mp4`: first 60 seconds, converted to H.264, 960×540, 15 fps, no audio.
- `traffic-annotated.mp4`: the same excerpt with CPU vehicle detection boxes,
  class names and confidence scores, re-encoded to H.264 for Windows playback.

The reduced resolution/frame rate keeps CPU testing manageable. This is actual
moving-camera footage, unlike the three-frame bus smoke fixture. Plate recognition
is not enabled in the supplied annotated preview.

To process it again (choose an unused output filename):

```bash
source .venv/bin/activate
export TORCH_HOME="$PWD/.cache/torch"
python -m roadcam --input test_drive.mp4 --output traffic-new.mp4
```

Copy MP4 files to a Windows folder before opening them in Windows Media Player
if opening them directly through the WSL path fails.

## City-driving sample (Pexels)

**Dash Cam Footage in City Driving**, by **German Korb**:
https://www.pexels.com/video/dash-cam-footage-in-city-driving-4644521/

License: **Pexels License**, https://www.pexels.com/license/ . This is separate
from the CC BY-SA license for the highway footage above. Pexels permits free use
and modification without mandatory attribution; restrictions include selling
unaltered copies, implying endorsement, and redistribution on stock/wallpaper
platforms. Consult the linked terms before redistribution.

- `artifacts/city/pexels-4644521.mp4`: downloaded original rendition.
- `city-drive.mp4`: first 60 seconds, resized/padded to 960×540, 15 fps, H.264,
  without audio, for CPU testing.
- `city-annotated.mp4`: vehicle-only detection results, encoded as H.264.

```bash
python -m roadcam --input city-drive.mp4 --output city-new.mp4
```

### User-supplied local city clip

The downloaded city clip is also available at
`artifacts/city-traffic/city-traffic.mp4` (2562×1440, approximately 62.7 seconds).
The original is preserved. Processing this full clip produces:

- `artifacts/city-traffic/city-traffic-cpu.mp4`: full-duration 960×540, 15 fps input.
- `artifacts/city-traffic/city-traffic-annotated.mp4`: H.264 vehicle-detection preview.

To run detection again:

```bash
python -m roadcam \
  --input artifacts/city-traffic/city-traffic-cpu.mp4 \
  --output artifacts/city-traffic/city-traffic-new.mp4
```

These derivatives omit audio. The Pexels source attribution and license above
apply to this copy too.
