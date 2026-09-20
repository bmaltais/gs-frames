from pathlib import Path
from typing import Optional

import pytest

from gs_frames.select import (
    resolve_min_sharpness,
    select_flow,
    select_frames,
    select_overlap_greedy,
    select_time,
)
from gs_frames.types import ConfigError, ExtractConfig, FrameScore


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def _score(
    index: int, timestamp_s: float, sharpness: float, flow_median: float = 0.0
) -> FrameScore:
    return FrameScore(
        index=index,
        timestamp_s=timestamp_s,
        sharpness=sharpness,
        sharpness_tenengrad=sharpness,
        sharpness_laplacian=sharpness,
        flow_median=flow_median,
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
        select_frames(scores, _config(mode="overlap-beam"))


def test_chunk_frames_and_every_seconds_are_mutually_exclusive():
    with pytest.raises(ConfigError):
        _config(chunk_frames=5, every_seconds=1.0)


def test_flow_trigger_must_be_positive():
    with pytest.raises(ConfigError):
        _config(flow_trigger=0.0)


def test_select_flow_empty_input_returns_empty():
    assert select_flow([], _config()) == []


def test_select_flow_closes_a_window_once_trigger_is_reached():
    # flow_median 0,3,3,3,3,3 (index 0 has no predecessor); trigger=5 closes
    # a window every time the running sum crosses it: [0,1,2]->6, [3,4]->6,
    # [5]->flushed unclosed at the end. Sharpest per window wins (sharpness
    # == index here).
    scores = [_score(i, i * 0.1, sharpness=float(i), flow_median=0.0 if i == 0 else 3.0) for i in range(6)]
    selections = select_flow(scores, _config(flow_trigger=5.0))
    assert all(s.reason == "flow-window" for s in selections)
    assert all(s.overlap_with_prev is None and s.overlap_metric == "none" for s in selections)
    assert [s.index for s in selections] == [2, 4, 5]


def test_select_flow_static_sequence_yields_a_single_selection():
    scores = [_score(i, i * 0.1, sharpness=float(i), flow_median=0.0) for i in range(10)]
    selections = select_flow(scores, _config(flow_trigger=5.0))
    assert len(selections) == 1
    assert selections[0].index == 9  # sharpest frame in the single unclosed window


def test_select_flow_motion_produces_more_selections_than_stillness():
    still = [_score(i, i * 0.1, sharpness=float(i % 3), flow_median=0.0) for i in range(30)]
    moving = [_score(i, i * 0.1, sharpness=float(i % 3), flow_median=2.0) for i in range(30)]
    still_selections = select_flow(still, _config(flow_trigger=5.0))
    moving_selections = select_flow(moving, _config(flow_trigger=5.0))
    assert len(moving_selections) > len(still_selections)


def test_select_flow_excludes_frames_below_min_sharpness():
    scores = [
        _score(i, i * 0.1, sharpness=10.0 if i < 3 else 1.0, flow_median=3.0) for i in range(5)
    ]
    # trigger high enough that all 5 frames sit in one (never-closed) window.
    selections = select_flow(scores, _config(flow_trigger=100.0, min_sharpness=5.0))
    assert [s.index for s in selections] == [0]  # sharpest among the >=5.0 frames


def test_select_flow_respects_max_frames_cap():
    scores = [_score(i, i * 0.1, sharpness=float(i), flow_median=3.0) for i in range(20)]
    selections = select_flow(scores, _config(flow_trigger=5.0, max_frames=2))
    assert len(selections) == 2


def test_select_frames_dispatches_flow_mode():
    scores = [_score(i, i * 0.1, sharpness=float(i), flow_median=3.0) for i in range(10)]
    assert select_frames(scores, _config(mode="flow", flow_trigger=5.0)) == select_flow(
        scores, _config(flow_trigger=5.0)
    )


def _fake_overlap_fn(overlaps: dict[tuple[int, int], float]):
    def fn(last_kept_index: int, candidate_index: int) -> Optional[float]:
        return overlaps.get((last_kept_index, candidate_index))

    return fn


def test_select_overlap_greedy_empty_input_returns_empty():
    assert select_overlap_greedy([], _config()) == []


def test_select_overlap_greedy_max_frames_le_zero_returns_empty():
    scores = [_score(0, 0.0, sharpness=5.0)]
    assert select_overlap_greedy(scores, _config(max_frames=0)) == []


def test_select_overlap_greedy_seed_is_first_frame_at_or_above_floor():
    scores = [_score(i, i * 0.1, sharpness=1.0 if i < 2 else 10.0) for i in range(5)]
    selections = select_overlap_greedy(
        scores, _config(min_sharpness=5.0, max_frames=1), overlap_fn=lambda a, b: None
    )
    assert [s.index for s in selections] == [2]
    assert selections[0].reason == "seed"
    assert selections[0].overlap_with_prev is None


def test_select_overlap_greedy_prefers_sharpest_in_range_candidate():
    scores = [
        _score(0, 0.0, sharpness=5.0),
        _score(1, 0.1, sharpness=9.0),
        _score(2, 0.2, sharpness=7.0),
        _score(3, 0.3, sharpness=10.0),
    ]
    overlaps = {(0, 1): 0.5, (0, 2): 0.75, (0, 3): 0.95}
    selections = select_overlap_greedy(
        scores,
        _config(overlap_min=0.7, overlap_max=0.8, min_sharpness=0.0, max_frames=2),
        overlap_fn=_fake_overlap_fn(overlaps),
    )
    assert [s.index for s in selections] == [0, 2]
    assert selections[1].reason == "in-range"
    assert selections[1].overlap_with_prev == 0.75


def test_select_overlap_greedy_falls_back_to_closest_to_target_with_sharpness_tiebreak():
    scores = [
        _score(0, 0.0, sharpness=5.0),
        _score(1, 0.1, sharpness=8.0),
        _score(2, 0.2, sharpness=9.0),
    ]
    overlaps = {(0, 1): 0.6, (0, 2): 0.9}  # both 0.15 away from target_overlap == 0.75
    selections = select_overlap_greedy(
        scores,
        _config(overlap_min=0.7, overlap_max=0.8, min_sharpness=0.0, max_frames=2),
        overlap_fn=_fake_overlap_fn(overlaps),
    )
    assert [s.index for s in selections] == [0, 2]  # tie broken by higher sharpness
    assert selections[1].reason == "fallback-closest"
    assert selections[1].overlap_with_prev == 0.9


def test_select_overlap_greedy_falls_back_to_sharpest_when_all_overlap_measurements_fail():
    scores = [
        _score(0, 0.0, sharpness=5.0),
        _score(1, 0.1, sharpness=8.0),
        _score(2, 0.2, sharpness=12.0),
    ]
    selections = select_overlap_greedy(
        scores, _config(min_sharpness=0.0, max_frames=2), overlap_fn=lambda a, b: None
    )
    assert [s.index for s in selections] == [0, 2]
    assert selections[1].reason == "fallback-closest"
    assert selections[1].overlap_with_prev is None


def test_select_overlap_greedy_bounds_pool_once_index_distance_and_flow_both_exceed_threshold():
    scores = [
        _score(0, 0.0, sharpness=5.0, flow_median=0.0),
        _score(1, 0.1, sharpness=6.0, flow_median=10.0),
        _score(2, 0.2, sharpness=7.0, flow_median=10.0),
        _score(3, 0.3, sharpness=20.0, flow_median=10.0),  # would be a perfect in-range pick...
        _score(4, 0.4, sharpness=20.0, flow_median=10.0),
    ]
    # ...but idx3/idx4 sit outside the search window once BOTH index distance
    # (> search_expand_frames=2) and accumulated flow (> 3 * flow_trigger=24)
    # have exceeded their bounds, so their overlap is never even measured.
    overlaps = {(0, 1): 0.5, (0, 2): 0.55, (0, 3): 0.75, (0, 4): 0.75}
    selections = select_overlap_greedy(
        scores,
        _config(
            overlap_min=0.7,
            overlap_max=0.8,
            min_sharpness=0.0,
            search_expand_frames=2,
            flow_trigger=8.0,
            max_frames=2,
        ),
        overlap_fn=_fake_overlap_fn(overlaps),
    )
    assert [s.index for s in selections] == [0, 2]  # closest-to-target among the reachable {1, 2}
    assert selections[1].reason == "fallback-closest"


def test_select_overlap_greedy_stops_pool_growth_after_low_overlap_streak():
    # 8 consecutive candidates measured well below overlap_min - 0.15 trip
    # the early-stop; a 9th candidate with a great overlap is never reached.
    scores = [_score(0, 0.0, sharpness=5.0)] + [
        _score(i, i * 0.1, sharpness=float(i)) for i in range(1, 10)
    ]
    overlaps = {(0, i): 0.3 for i in range(1, 9)}
    overlaps[(0, 9)] = 0.75
    selections = select_overlap_greedy(
        scores,
        _config(overlap_min=0.7, overlap_max=0.8, min_sharpness=0.0, max_frames=2),
        overlap_fn=_fake_overlap_fn(overlaps),
    )
    assert selections[1].index != 9
    assert selections[1].index == 8  # sharpest among the 8 equally-far-off candidates
    assert selections[1].reason == "fallback-closest"


def test_select_overlap_greedy_respects_max_frames_cap():
    scores = [_score(i, i * 0.1, sharpness=float(i)) for i in range(10)]
    overlaps = {(i, i + 1): 0.75 for i in range(9)}
    selections = select_overlap_greedy(
        scores,
        _config(overlap_min=0.7, overlap_max=0.8, min_sharpness=0.0, max_frames=3),
        overlap_fn=_fake_overlap_fn(overlaps),
    )
    assert len(selections) == 3


def test_select_overlap_greedy_flow_metric_computes_overlap_without_overlap_fn():
    scores = [
        _score(0, 0.0, sharpness=5.0, flow_median=0.0),
        _score(1, 0.1, sharpness=6.0, flow_median=1.0),
        _score(2, 0.2, sharpness=7.0, flow_median=20.0),
    ]
    selections = select_overlap_greedy(
        scores,
        _config(
            overlap_metric="flow",
            flow_trigger=8.0,
            overlap_min=0.9,
            overlap_max=1.0,
            min_sharpness=0.0,
            max_frames=2,
        ),
    )  # no overlap_fn passed: "flow" never needs one
    assert [s.index for s in selections] == [0, 1]
    assert selections[1].reason == "in-range"
    assert selections[1].overlap_with_prev == pytest.approx(1 - 1 / 24, abs=1e-6)


def test_select_overlap_greedy_none_metric_always_falls_back():
    scores = [_score(i, i * 0.1, sharpness=float(i)) for i in range(4)]
    selections = select_overlap_greedy(
        scores, _config(overlap_metric="none", min_sharpness=0.0, max_frames=2)
    )
    assert [s.index for s in selections] == [0, 3]  # sharpest remaining; overlap never measured
    assert selections[1].reason == "fallback-closest"
    assert selections[1].overlap_with_prev is None


def test_select_overlap_greedy_requires_overlap_fn_for_orb_metric():
    scores = [_score(0, 0.0, sharpness=5.0), _score(1, 0.1, sharpness=6.0)]
    with pytest.raises(ValueError):
        select_overlap_greedy(scores, _config(min_sharpness=0.0))


def test_select_frames_dispatches_overlap_greedy_mode():
    scores = [_score(0, 0.0, sharpness=5.0), _score(1, 0.1, sharpness=6.0)]
    overlaps = {(0, 1): 0.75}
    config = _config(mode="overlap-greedy", overlap_min=0.7, overlap_max=0.8, min_sharpness=0.0)
    assert select_frames(
        scores, config, overlap_fn=_fake_overlap_fn(overlaps)
    ) == select_overlap_greedy(scores, config, overlap_fn=_fake_overlap_fn(overlaps))
