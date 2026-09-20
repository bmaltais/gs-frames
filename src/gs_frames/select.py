"""Selection logic: pure functions over FrameScore lists and ExtractConfig.

No OpenCV objects or video I/O here -- callers own decoding; these functions
take plain in-memory data (and, from phase 4 on, an injected overlap_fn) so
selection behavior is unit-testable without a real video. See "Selection" in
docs/specs/0001-gs-frames-overlap-extraction.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from gs_frames.overlap import OverlapFn, flow_proxy_overlap
from gs_frames.types import ExtractConfig, FrameScore, Selection, SelectMode

# Applied when neither --chunk-frames nor --every-seconds is given: one
# window per second of video, a reasonable default sampling rate.
_DEFAULT_EVERY_SECONDS = 1.0

# Single source of truth for which --mode values select_frames can actually
# run; overlap-beam lands in phase 5. cli.py checks this up front (before
# decoding a whole video) rather than relying solely on the
# NotImplementedError below.
IMPLEMENTED_MODES: frozenset[SelectMode] = frozenset({"time", "flow", "overlap-greedy"})

# select_overlap_greedy's candidate-pool bound: a candidate stays in the pool
# while it is within --search-expand-frames index distance of the last-kept
# frame OR accumulated flow_median since the last-kept frame is still below
# this multiple of --flow-trigger (whichever bound is more permissive keeps
# it in the pool -- slow motion gets a farther-reaching search).
_SEARCH_FLOW_MULTIPLIER = 3

# Once a candidate's measured overlap falls this far below overlap_min, it is
# considered "clearly too low" for the purpose of early-stopping pool growth.
_LOW_OVERLAP_MARGIN = 0.15

# Consecutive "clearly too low" measured overlaps that stop pool growth early
# (the reference frame is assumed to have "moved on" for good, not just
# noisily dipped below range once).
_LOW_OVERLAP_STREAK_LIMIT = 8


def resolve_min_sharpness(scores: list[FrameScore], config: ExtractConfig) -> float:
    """The sharpness floor below which a frame can never win a window/seed.

    An explicit --min-sharpness wins; otherwise the floor is the
    --min-sharpness-percentile of the analyzed population.
    """
    if config.min_sharpness is not None:
        return config.min_sharpness
    if not scores:
        return float("-inf")
    values = np.array([s.sharpness for s in scores], dtype=np.float64)
    return float(np.percentile(values, config.min_sharpness_percentile))


def _time_window_key(score: FrameScore, config: ExtractConfig) -> int:
    if config.chunk_frames is not None:
        return score.index // config.chunk_frames
    every_seconds = config.every_seconds or _DEFAULT_EVERY_SECONDS
    return int(score.timestamp_s // every_seconds)


def _window_selection(score: FrameScore, reason: str) -> Selection:
    """A Selection for a window-based mode (time/flow): no overlap metric is
    computed in these modes, so overlap_with_prev/overlap_metric are always
    None/"none"."""
    return Selection(
        index=score.index,
        timestamp_s=score.timestamp_s,
        sharpness=score.sharpness,
        overlap_with_prev=None,
        overlap_metric="none",
        reason=reason,
    )


def select_time(scores: list[FrameScore], config: ExtractConfig) -> list[Selection]:
    """Sharpest frame per `--chunk-frames`- or `--every-seconds`-wide window.

    Frames below the resolved min-sharpness floor can never win a window; a
    window with no frame above the floor contributes no selection (a gap in
    coverage, not a forced bad pick).
    """
    if not scores:
        return []

    floor = resolve_min_sharpness(scores, config)

    windows: dict[int, FrameScore] = {}
    order: list[int] = []
    for score in scores:
        if score.sharpness < floor:
            continue
        key = _time_window_key(score, config)
        current = windows.get(key)
        if current is None:
            order.append(key)
            windows[key] = score
        elif score.sharpness > current.sharpness:
            windows[key] = score

    selections = [_window_selection(windows[key], "time-window") for key in order]

    if config.max_frames is not None:
        selections = selections[: config.max_frames]

    return selections


def select_flow(scores: list[FrameScore], config: ExtractConfig) -> list[Selection]:
    """Sharpest frame per motion-adaptive window: a window closes once
    accumulated `flow_median` since the last window boundary reaches
    `--flow-trigger`, so static segments produce fewer, wider windows and
    fast motion produces more, narrower ones. Accumulation is frame-index
    driven (a running sum over the FrameScore sequence), not timestamp
    driven, so VFR does not skew window sizing.
    """
    if not scores:
        return []

    floor = resolve_min_sharpness(scores, config)
    selections: list[Selection] = []
    window: list[FrameScore] = []
    accum = 0.0

    def flush(frames: list[FrameScore]) -> None:
        candidates = [s for s in frames if s.sharpness >= floor]
        if not candidates:
            return
        best = max(candidates, key=lambda s: s.sharpness)
        selections.append(_window_selection(best, "flow-window"))

    for score in scores:
        window.append(score)
        accum += score.flow_median
        if accum >= config.flow_trigger:
            flush(window)
            window = []
            accum = 0.0
    flush(window)

    if config.max_frames is not None:
        selections = selections[: config.max_frames]

    return selections


def _overlap_selection(score: FrameScore, overlap_value: Optional[float], metric: str, reason: str) -> Selection:
    """A Selection for select_overlap_greedy, mirroring _window_selection's
    role for the window-based modes -- the one place a Selection gets built
    from a FrameScore, so seed/in-range/fallback-closest picks can't drift
    out of sync with each other."""
    return Selection(
        index=score.index,
        timestamp_s=score.timestamp_s,
        sharpness=score.sharpness,
        overlap_with_prev=overlap_value,
        overlap_metric=metric,
        reason=reason,
    )


@dataclass
class _OverlapCandidate:
    """One entry in select_overlap_greedy's candidate pool: `pos` is the
    candidate's position in the caller's `scores` list (so a pick can seed
    the next pool's search without re-scanning for it)."""

    pos: int
    score: FrameScore
    overlap: Optional[float]


def _measure_overlap(
    last_kept: FrameScore,
    candidate: FrameScore,
    flow_accum: float,
    config: ExtractConfig,
    overlap_fn: Optional[OverlapFn],
) -> Optional[float]:
    """Routes to the right overlap computation for config.overlap_metric.
    "none" always fails the measurement (matches phase 2/3 modes' lack of
    geometric overlap); "flow" is pure arithmetic on the flow already
    accumulated while building the candidate pool, so it never needs
    overlap_fn/image data at all.
    """
    if config.overlap_metric == "none":
        return None
    if config.overlap_metric == "flow":
        return flow_proxy_overlap(flow_accum, config.flow_trigger)
    if overlap_fn is None:
        raise ValueError(
            f"select_overlap_greedy requires overlap_fn for --overlap-metric {config.overlap_metric!r}"
        )
    return overlap_fn(last_kept.index, candidate.index)


def _overlap_candidate_pool(
    scores: list[FrameScore],
    pos: int,
    last_kept: FrameScore,
    floor: float,
    config: ExtractConfig,
    overlap_fn: Optional[OverlapFn],
) -> list[_OverlapCandidate]:
    """Frames after list position `pos` that stay within the search window
    from `last_kept` (see _SEARCH_FLOW_MULTIPLIER above), each paired with
    its measured overlap against `last_kept` (None if the measurement
    failed). Frames below the sharpness floor consume search-window budget
    (they still count toward index distance/flow accumulation) but are never
    added to the pool, matching select_time/select_flow's "never wins"
    treatment of sub-floor frames.
    """
    pool: list[_OverlapCandidate] = []
    flow_accum = 0.0
    low_streak = 0
    for i in range(pos + 1, len(scores)):
        score = scores[i]
        flow_accum += score.flow_median
        index_distance = score.index - last_kept.index
        if (
            index_distance > config.search_expand_frames
            and flow_accum > _SEARCH_FLOW_MULTIPLIER * config.flow_trigger
        ):
            break
        if score.sharpness < floor:
            continue

        measured = _measure_overlap(last_kept, score, flow_accum, config, overlap_fn)
        pool.append(_OverlapCandidate(pos=i, score=score, overlap=measured))

        if measured is not None and measured < config.overlap_min - _LOW_OVERLAP_MARGIN:
            low_streak += 1
            if low_streak >= _LOW_OVERLAP_STREAK_LIMIT:
                break
        else:
            low_streak = 0
    return pool


def _pick_overlap_candidate(
    pool: list[_OverlapCandidate], config: ExtractConfig
) -> tuple[_OverlapCandidate, str]:
    """Prefers the sharpest in-range candidate; else the candidate closest to
    target_overlap (tie-break: higher sharpness); else, if every overlap
    measurement in the pool failed, the sharpest candidate outright -- a
    coverage gap is never left silently (spec user story 10).
    """
    in_range = [
        c for c in pool if c.overlap is not None and config.overlap_min <= c.overlap <= config.overlap_max
    ]
    if in_range:
        return max(in_range, key=lambda c: c.score.sharpness), "in-range"

    measured = [c for c in pool if c.overlap is not None]
    if measured:
        target = config.target_overlap
        if target is None:
            target = (config.overlap_min + config.overlap_max) / 2
        best = min(
            measured,
            key=lambda c: (abs(c.overlap - target) if c.overlap is not None else 0.0, -c.score.sharpness),
        )
        return best, "fallback-closest"

    return max(pool, key=lambda c: c.score.sharpness), "fallback-closest"


def select_overlap_greedy(
    scores: list[FrameScore],
    config: ExtractConfig,
    overlap_fn: Optional[OverlapFn] = None,
) -> list[Selection]:
    """Overlap-constrained greedy selection (phase 4's eventual default
    mode): seeds on the first frame at/above the sharpness floor, then
    repeatedly extends from the last-kept frame via _overlap_candidate_pool
    and _pick_overlap_candidate until the pool runs dry (end of video, or a
    fully-collapsed search window) or --max-frames is reached.
    """
    if not scores:
        return []
    if config.max_frames is not None and config.max_frames <= 0:
        return []

    floor = resolve_min_sharpness(scores, config)
    seed_pos = next((i for i, s in enumerate(scores) if s.sharpness >= floor), None)
    if seed_pos is None:
        return []

    seed = scores[seed_pos]
    selections = [_overlap_selection(seed, None, config.overlap_metric, "seed")]
    last_kept = seed
    pos = seed_pos

    while config.max_frames is None or len(selections) < config.max_frames:
        pool = _overlap_candidate_pool(scores, pos, last_kept, floor, config, overlap_fn)
        if not pool:
            break
        picked, reason = _pick_overlap_candidate(pool, config)
        selections.append(_overlap_selection(picked.score, picked.overlap, config.overlap_metric, reason))
        last_kept = picked.score
        pos = picked.pos

    return selections


_ModeFunc = Callable[[list[FrameScore], ExtractConfig, Optional[OverlapFn]], list[Selection]]

_MODE_FUNCS: dict[SelectMode, _ModeFunc] = {
    "time": lambda scores, config, overlap_fn: select_time(scores, config),
    "flow": lambda scores, config, overlap_fn: select_flow(scores, config),
    "overlap-greedy": select_overlap_greedy,
}


def select_frames(
    scores: list[FrameScore],
    config: ExtractConfig,
    overlap_fn: Optional[OverlapFn] = None,
) -> list[Selection]:
    """Dispatches to the selection function for config.mode -- the single
    place mode -> implementation routing happens."""
    if config.mode not in IMPLEMENTED_MODES:
        raise NotImplementedError(
            f"--mode {config.mode!r} is not implemented yet (lands in a later phase)"
        )
    return _MODE_FUNCS[config.mode](scores, config, overlap_fn)
