from pathlib import Path

from gs_frames.cli import _estimate_analysis_frame_count
from gs_frames.decode import VideoInfo
from gs_frames.types import ExtractConfig


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def _video_info(**overrides) -> VideoInfo:
    defaults = dict(
        fps=30.0,
        frame_count=900,
        width=1080,
        height=1920,
        rotation_applied=0,
        rotation_source="none_detected",
        backend="opencv",
    )
    defaults.update(overrides)
    return VideoInfo(**defaults)


def test_no_bounds_uses_full_frame_count():
    assert _estimate_analysis_frame_count(_config(), _video_info()) == 900


def test_end_seconds_bounds_the_total():
    total = _estimate_analysis_frame_count(_config(end_s=5.0), _video_info(fps=30.0))
    assert total == 150


def test_start_and_end_seconds_bound_the_total():
    total = _estimate_analysis_frame_count(
        _config(start_s=2.0, end_s=5.0), _video_info(fps=30.0)
    )
    assert total == 90


def test_end_frame_takes_precedence_over_end_seconds():
    total = _estimate_analysis_frame_count(
        _config(end_frame=10, end_s=100.0), _video_info(fps=30.0)
    )
    assert total == 10


def test_unknown_frame_count_and_no_end_bound_is_indeterminate():
    assert _estimate_analysis_frame_count(_config(), _video_info(frame_count=None)) is None
