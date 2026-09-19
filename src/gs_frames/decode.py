"""Video decode boundary: the only module that talks to OpenCV/ffmpeg directly.

Everything downstream (sharpness, motion, overlap, selection, export) consumes
plain (index, timestamp_s, bgr_frame) tuples and never touches cv2.VideoCapture
or a subprocess directly. See docs/specs/0001-gs-frames-overlap-extraction.md
section 8a for why rotation, codec reliability, and VFR timestamps each need
explicit handling on iPhone-recorded input.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import cv2
import numpy as np

from gs_frames.types import RotationMode, RotationSource

logger = logging.getLogger("gs_frames.decode")

_ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class VideoInfo:
    fps: float
    frame_count: Optional[int]
    width: int
    height: int
    rotation_applied: int
    rotation_source: RotationSource
    backend: str
    timestamps_source: str = "container"


@dataclass
class AnalysisFrame:
    index: int
    timestamp_s: float
    frame: np.ndarray


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    flag = _ROTATE_FLAGS.get(rotation)
    if flag is None:
        return frame
    return cv2.rotate(frame, flag)


def _probe_rotation_ffprobe(path: Path) -> Optional[int]:
    """Returns the rotation in degrees, or None if ffprobe itself could not be
    run/parsed. A successful run with no rotate tag/side-data is a valid "0"
    answer, not a failure -- callers must not conflate the two.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            return int(round(float(tags["rotate"]))) % 360
        except ValueError:
            pass

    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                return int(round(abs(float(side_data["rotation"])))) % 360
            except ValueError:
                continue
    # ffprobe ran and parsed fine, it just found no rotation tag/side-data:
    # a successful "no rotation" answer, distinct from ffprobe failing outright.
    return 0


def _probe_rotation_cv2(cap: cv2.VideoCapture) -> int:
    try:
        value = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    except cv2.error:
        return 0
    if value and not np.isnan(value):
        return int(round(value)) % 360
    return 0


def _detect_rotation(
    path: Path, cap: Optional[cv2.VideoCapture], rotation_mode: RotationMode
) -> tuple[int, RotationSource]:
    if rotation_mode != "auto":
        return int(rotation_mode) if rotation_mode != "none" else 0, "user_override"

    if cap is not None:
        cv2_rotation = _probe_rotation_cv2(cap)
        if cv2_rotation:
            return cv2_rotation, "container_meta"

    ffprobe_rotation = _probe_rotation_ffprobe(path)
    if ffprobe_rotation is not None:
        return ffprobe_rotation, "ffprobe"

    logger.warning(
        "Could not verify video rotation (no container metadata, ffprobe unavailable or "
        "inconclusive); proceeding with rotation=0."
    )
    return 0, "none_detected"


class _DecodeBackend(Protocol):
    """Common shape of the two decode backends (OpenCV, ffmpeg-subprocess fallback)."""

    def usable(self) -> bool: ...
    def raw_info(self) -> tuple[float, Optional[int], int, int]: ...
    def iter_frames(self, start_frame: int = 0) -> Iterator[tuple[int, float, np.ndarray]]: ...
    def close(self) -> None: ...


class _OpenCvBackend:
    """Decodes via cv2.VideoCapture."""

    def __init__(self, path: Path, rotation: int, use_container_timestamps: bool):
        self._path = path
        self._rotation = rotation
        self._use_container_timestamps = use_container_timestamps
        self.cap = cv2.VideoCapture(str(path))
        # Some OpenCV/FFmpeg builds auto-apply container rotation metadata to
        # decoded frames themselves (and to FRAME_WIDTH/HEIGHT); others don't.
        # Disabling this explicitly makes rotation handling deterministic
        # across builds -- we always get raw frames here and apply the
        # rotation ourselves exactly once in _apply_rotation.
        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def usable(self) -> bool:
        if not self.cap.isOpened():
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    def raw_info(self) -> tuple[float, Optional[int], int, int]:
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return fps, (frame_count or None), width, height

    def iter_frames(self, start_frame: int = 0) -> Iterator[tuple[int, float, np.ndarray]]:
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame
        nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        warned_bad_timestamps = False
        last_valid_msec: Optional[float] = None
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return
            timestamp_s: float
            if self._use_container_timestamps:
                msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                # msec == 0 is a legitimate reading for the very first frame,
                # but for any later frame it (and any non-increasing value)
                # signals the backend isn't reporting real per-frame PTS.
                is_valid = (
                    msec is not None
                    and not np.isnan(msec)
                    and (msec > 0 or index == start_frame)
                    and (last_valid_msec is None or msec > last_valid_msec)
                )
                if is_valid:
                    timestamp_s = msec / 1000.0
                    last_valid_msec = msec
                else:
                    if not warned_bad_timestamps:
                        logger.warning(
                            "CAP_PROP_POS_MSEC unavailable/non-monotonic; falling back to "
                            "index/fps timestamps (may be approximate for VFR sources)."
                        )
                        warned_bad_timestamps = True
                    timestamp_s = index / nominal_fps
            else:
                timestamp_s = index / nominal_fps
            yield index, timestamp_s, _apply_rotation(frame, self._rotation)
            index += 1

    def close(self) -> None:
        self.cap.release()


class _FfmpegBackend:
    """Fallback decoder: pipes raw BGR24 frames from a system ffmpeg process.

    Used only when the installed opencv-python build cannot open/read the
    input (e.g. HEVC support missing from its bundled ffmpeg). Timestamps here
    are index/fps, since a raw video pipe does not carry per-frame PTS: this
    fallback path is approximate and a warning is logged accordingly.
    """

    def __init__(self, path: Path, rotation: int):
        self._path = path
        self._rotation = rotation
        info = self._probe()
        self.fps, self.frame_count, self.width, self.height = info

    def _probe(self) -> tuple[float, Optional[int], int, int]:
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_frames",
                    "-of",
                    "json",
                    str(self._path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            stream = json.loads(proc.stdout)["streams"][0]
            num, den = (stream.get("r_frame_rate") or "30/1").split("/")
            fps = float(num) / float(den) if float(den) else 30.0
            frame_count = (
                int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None
            )
            width = int(stream["width"])
            height = int(stream["height"])
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"ffprobe fallback probe failed for {self._path}: {exc}") from exc
        return fps, frame_count, width, height

    def usable(self) -> bool:
        return self.width > 0 and self.height > 0

    def raw_info(self) -> tuple[float, Optional[int], int, int]:
        return self.fps, self.frame_count, self.width, self.height

    def iter_frames(self, start_frame: int = 0) -> Iterator[tuple[int, float, np.ndarray]]:
        logger.warning(
            "Using ffmpeg-subprocess decode fallback; per-frame timestamps are "
            "index/fps approximations, not true container timestamps."
        )
        frame_bytes = self.width * self.height * 3
        cmd = ["ffmpeg", "-v", "error", "-i", str(self._path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            index = 0
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    return
                if index >= start_frame:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
                    yield index, index / self.fps, _apply_rotation(frame, self._rotation)
                index += 1
        finally:
            proc.stdout.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def close(self) -> None:
        pass


class DecodedVideo:
    """Open video handle exposing rotated frames and resolved VideoInfo."""

    def __init__(self, path: Path, rotation_mode: RotationMode, use_container_timestamps: bool):
        self.path = path
        cv2_backend = _OpenCvBackend(path, rotation=0, use_container_timestamps=use_container_timestamps)
        cv2_usable = cv2_backend.usable()
        probe_cap = cv2_backend.cap if cv2_usable else None
        rotation, rotation_source = _detect_rotation(path, probe_cap, rotation_mode)

        self._backend: _DecodeBackend
        if cv2_usable:
            cv2_backend._rotation = rotation
            self._backend = cv2_backend
            backend_name = "opencv"
        else:
            cv2_backend.close()
            ffmpeg_backend = _FfmpegBackend(path, rotation)
            if not ffmpeg_backend.usable():
                raise RuntimeError(
                    f"Could not decode video with either OpenCV or the ffmpeg fallback: {path}"
                )
            self._backend = ffmpeg_backend
            backend_name = "ffmpeg"

        fps, frame_count, raw_w, raw_h = self._backend.raw_info()
        width, height = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)
        self.info = VideoInfo(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            rotation_applied=rotation,
            rotation_source=rotation_source,
            backend=backend_name,
        )
        logger.info(
            "video fps=%.3f size=%dx%d frame_count=%s rotation=%d (%s) backend=%s",
            fps,
            width,
            height,
            frame_count,
            rotation,
            rotation_source,
            backend_name,
        )

    def iter_analysis_frames(
        self,
        *,
        analysis_scale: float,
        analysis_max_width: int,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> Iterator[AnalysisFrame]:
        """Yield rotated frames downscaled per the configured analysis size."""
        target_w: Optional[int] = None
        target_h: Optional[int] = None
        for index, timestamp_s, frame in self._backend.iter_frames(start_frame=start_frame):
            if end_frame is not None and index >= end_frame:
                return
            if start_s is not None and timestamp_s < start_s:
                continue
            if end_s is not None and timestamp_s > end_s:
                return
            if target_w is None:
                src_h, src_w = frame.shape[:2]
                target_w = max(1, min(round(src_w * analysis_scale), analysis_max_width))
                target_h = max(1, round(src_h * target_w / src_w))
            assert target_w is not None and target_h is not None
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            yield AnalysisFrame(index=index, timestamp_s=timestamp_s, frame=resized)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "DecodedVideo":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_video(
    path: Path, rotation_mode: RotationMode = "auto", use_container_timestamps: bool = True
) -> DecodedVideo:
    return DecodedVideo(path, rotation_mode, use_container_timestamps)
