from pathlib import Path

import pytest

from gs_frames.select import resolve_min_sharpness, select_frames, select_time
from gs_frames.types import ConfigError, ExtractConfig, FrameScore


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def _score(index: int, timestamp_s: float, sharpness: float) -> FrameScore:
    return FrameScore(
        index=index,
        timestamp_s=timestamp_s,
        sharpness=sharpness,
        sharpness_tenengrad=sharpness,
        sharpness_laplacian=sharpness,
    )


def test_select_time_empty_input_returns_empty():
    assert select_time([], _config()) == []


def test_select_time_by_chunk_frames_picks_sharpest_per_window():
    scores = [_score(i, i * 0.1, sharpness=float(i % 5)) for i in range(10)]
    selections = select_time(scores, _config(chunk_frames=5))
    assert [s.index for s in selections] == [4, 9]
    assert all(s.reason == "time-window" for s in selections)
    assert all(s.overlap_with_prev is None and s.overlap_metric == "none" for s in selections)


def test_select_time_by_every_seconds_groups_by_timestamp():
    scores = [_score(i, i * 0.5, sharpness=float(i)) for i in range(6)]  # t=0,0.5,...,2.5
    selections = select_time(scores, _config(every_seconds=1.0))
    # windows: [0,0.5)->0..wait floor(t//1.0): t=0,0.5->0; t=1,1.5->1; t=2,2.5->2
    assert [s.index for s in selections] == [1, 3, 5]


def test_select_time_defaults_to_one_second_windows_when_unset():
    scores = [_score(i, i * 0.5, sharpness=float(i)) for i in range(6)]
    default_selections = select_time(scores, _config())
    explicit_selections = select_time(scores, _config(every_seconds=1.0))
    assert [s.index for s in default_selections] == [s.index for s in explicit_selections]


def test_select_time_excludes_frames_below_min_sharpness():
    scores = [_score(i, i * 0.1, sharpness=10.0 if i < 3 else 1.0) for i in range(5)]
    selections = select_time(scores, _config(chunk_frames=5, min_sharpness=5.0))
    assert [s.index for s in selections] == [0]  # sharpest among the >=5.0 frames


def test_select_time_window_with_no_frame_above_floor_is_a_gap():
    scores = [_score(i, i * 0.1, sharpness=1.0) for i in range(5)] + [
        _score(i, i * 0.1, sharpness=10.0) for i in range(5, 10)
    ]
    selections = select_time(scores, _config(chunk_frames=5, min_sharpness=5.0))
    assert [s.index for s in selections] == [5]  # equal-sharpness ties keep the earliest


def test_select_time_respects_max_frames_cap():
    scores = [_score(i, i * 0.1, sharpness=float(i)) for i in range(20)]
    selections = select_time(scores, _config(chunk_frames=2, max_frames=3))
    assert len(selections) == 3


def test_resolve_min_sharpness_prefers_explicit_value():
    scores = [_score(i, 0.0, sharpness=float(i)) for i in range(10)]
    assert resolve_min_sharpness(scores, _config(min_sharpness=42.0)) == 42.0


def test_resolve_min_sharpness_falls_back_to_percentile():
    scores = [_score(i, 0.0, sharpness=float(i)) for i in range(101)]  # 0..100
    floor = resolve_min_sharpness(scores, _config(min_sharpness_percentile=10.0))
    assert floor == pytest.approx(10.0, abs=0.5)


def test_select_frames_dispatches_time_mode():
    scores = [_score(i, i * 0.1, sharpness=float(i)) for i in range(5)]
    assert select_frames(scores, _config(mode="time", chunk_frames=5)) == select_time(
        scores, _config(chunk_frames=5)
    )


def test_select_frames_raises_for_unimplemented_modes():
    scores = [_score(0, 0.0, 1.0)]
    with pytest.raises(NotImplementedError):
        select_frames(scores, _config(mode="flow"))


def test_chunk_frames_and_every_seconds_are_mutually_exclusive():
    with pytest.raises(ConfigError):
        _config(chunk_frames=5, every_seconds=1.0)
