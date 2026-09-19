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
gs-frames extract take.mp4 out/ --overlap 70-80
```

This is phase 2: it writes `out/analysis.csv` (one row per analyzed frame,
with Tenengrad/Laplacian/combined sharpness and a `selected` flag) and
`out/manifest.json` (video metadata, resolved config, stats, per-selected-
frame records), then exports the selected frames at full resolution to
`out/images/frame_XXXXXX.jpg` (zero-padded original frame index). Pass
`--preview` to skip image export and only write the CSV/manifest.

Selection is currently `--mode time` (the only mode implemented so far;
`overlap-greedy` becomes the default once phase 4 lands): the sharpest frame
per `--chunk-frames N` or `--every-seconds N` window (default: one window per
second), skipping frames below `--min-sharpness` (or the
`--min-sharpness-percentile`, default 5th percentile, when unset). Cap the
result with `--max-frames N`.

Exported images are protected from accidental overwrite; pass `--force` to
replace a prior export. Output format/quality: `--format jpg|png` and
`--quality N` (JPEG only).

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

Apple's "HDR Video" capture mode (10-bit HEVC, HLG/Rec.2020) is detected via
`ffprobe`'s `color_transfer` tag and tone-mapped down to SDR through an
ffmpeg-subprocess `zscale`/`tonemap` pipeline before any scoring/export --
`cv2.VideoCapture` has no HDR-aware decode path of its own and reads such
clips out far too bright otherwise. This forces the ffmpeg-subprocess backend
(so timestamps become `index/fps` approximations, per above) and requires a
working `ffmpeg`+`ffprobe` on PATH; if either is missing, extraction falls
back to the raw (too-bright) OpenCV decode with a logged warning rather than
failing outright. Whether tone-mapping was applied is recorded in
`manifest.json` under `video_meta.hdr_tonemapped`.

## Known limitations

- No auto-exposure/white-balance compensation for iPhone auto-exposure hunting.
- Overlap (from phase 4 onward) is a 2D image-plane proxy, not true 3D
  covisibility.
- VFR timestamp accuracy depends on `CAP_PROP_POS_MSEC` support in the local
  OpenCV build; when unavailable, timestamps fall back to `index/fps` and a
  warning is logged.
