# gs-frames

Extract sharp, overlap-aware stills from video for 3D Gaussian Splatting and COLMAP.

Spec: [`docs/specs/0001-gs-frames-overlap-extraction.md`](docs/specs/0001-gs-frames-overlap-extraction.md).

## Install

```bash
uv venv .venv
uv pip install -e ".[dev]" --python .venv/bin/python
```

If your OpenCV build can't decode HEVC (`.mov` from iPhone), also install a
system `ffmpeg` binary. `decode.py` detects this at video-open time and falls
back to a subprocess pipe through `ffmpeg` automatically when
`cv2.VideoCapture` can't open or read the file; that fallback path uses
`index/fps` timestamps (not true container timestamps) since a raw video pipe
carries no per-frame PTS.

## Quick start

```bash
gs-frames extract take.mp4 out/ --preview --overlap 70-80
```

This is phase 1: it writes `out/analysis.csv` (one row per analyzed frame,
with Tenengrad/Laplacian/combined sharpness) and `out/manifest.json`
(video metadata, resolved config, stats). No images are exported yet —
image export lands in phase 2; running without `--preview` prints that and
still exits 0 having written the CSV and manifest.

## iPhone footage

Rotation is auto-detected from container metadata (`CAP_PROP_ORIENTATION_META`,
falling back to `ffprobe` when that's unavailable or reports no rotation);
override with `--rotate auto|none|90|180|270` if detection is wrong. The
detected/applied rotation and its source (`container_meta`, `ffprobe`,
`user_override`, or `none_detected`) are logged at startup and recorded in
`manifest.json` under `video_meta`.

Timestamps use the container's per-frame values (`CAP_PROP_POS_MSEC`) by
default, which handles variable frame rate; use `--no-container-timestamps`
to force `index/fps` timestamps for debugging.

## Known limitations

- No auto-exposure/white-balance compensation for iPhone auto-exposure hunting.
- Overlap (from phase 4 onward) is a 2D image-plane proxy, not true 3D
  covisibility.
- VFR timestamp accuracy depends on `CAP_PROP_POS_MSEC` support in the local
  OpenCV build; when unavailable, timestamps fall back to `index/fps` and a
  warning is logged.
