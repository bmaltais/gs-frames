from pathlib import Path

import cv2
import numpy as np
import pytest


def make_synthetic_video(path: Path, *, n_frames: int = 30, width: int = 320, height: int = 240, fps: float = 10.0) -> None:
    """30 BGR frames: first half sharp rectangles, second half heavily blurred, CFR, unrotated."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter (mp4v) unavailable on this platform")
    try:
        rng = np.random.default_rng(42)
        half = n_frames // 2
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = rng.integers(0, 40, size=3)
            x0, y0 = (i * 5) % (width - 60), (i * 3) % (height - 60)
            cv2.rectangle(frame, (x0, y0), (x0 + 60, y0 + 60), (255, 255, 255), -1)
            cv2.rectangle(frame, (x0 + 10, y0 + 10), (x0 + 50, y0 + 50), (0, 0, 0), -1)
            if i >= half:
                frame = cv2.GaussianBlur(frame, (21, 21), 8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    path = tmp_path / "synthetic.mp4"
    make_synthetic_video(path)
    return path


@pytest.fixture
def iphone_fixture() -> Path:
    """A real iPhone-recorded clip to exercise 8a (rotation/HEVC/VFR) against.

    Checked in two places: the committable `tests/fixtures/` path, and the
    gitignored `videos/` dev folder (real footage kept local-only, not
    committed -- see .gitignore). Neither is required; the test skips with a
    clear reason if no real clip is present in this checkout.
    """
    committed = Path(__file__).parent / "fixtures" / "iphone_sample_portrait_hevc.mov"
    if committed.exists():
        return committed

    local_dir = Path(__file__).parent.parent / "videos"
    if local_dir.is_dir():
        for ext in ("*.mov", "*.MOV", "*.mp4", "*.MP4"):
            matches = sorted(local_dir.glob(ext))
            if matches:
                return matches[0]

    pytest.skip(
        "no real iPhone clip found (checked tests/fixtures/iphone_sample_portrait_hevc.mov "
        "and videos/*.mov); 8a rotation/HEVC/VFR handling is implemented in decode.py but "
        "cannot be exercised against real iPhone footage without one."
    )
