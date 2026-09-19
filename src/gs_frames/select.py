"""Selection logic: pure functions over FrameScore lists and ExtractConfig.

No OpenCV objects or video I/O here -- callers own decoding; these functions
take plain in-memory data (and, from phase 4 on, an injected overlap_fn) so
selection behavior is unit-testable without a real video. See "Selection" in
docs/specs/0001-gs-frames-overlap-extraction.md.
"""

from __future__ import annotations

import numpy as np

from gs_frames.types import ExtractConfig, FrameScore, Selection, SelectMode

# Applied when neither --chunk-frames nor --every-seconds is given: one
# window per second of video, a reasonable default sampling rate.
_DEFAULT_EVERY_SECONDS = 1.0

# Single source of truth for which --mode values select_frames can actually
# run; flow/overlap-greedy/overlap-beam land in phases 3-5. cli.py checks
# this up front (before decoding a whole video) rather than relying solely on
# the NotImplementedError below.
IMPLEMENTED_MODES: frozenset[SelectMode] = frozenset({"time"})


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

    selections = [
        Selection(
            index=windows[key].index,
            timestamp_s=windows[key].timestamp_s,
            sharpness=windows[key].sharpness,
            overlap_with_prev=None,
            overlap_metric="none",
            reason="time-window",
        )
        for key in order
    ]

    if config.max_frames is not None:
        selections = selections[: config.max_frames]

    return selections


def select_frames(scores: list[FrameScore], config: ExtractConfig) -> list[Selection]:
    """Dispatches to the selection function for config.mode."""
    if config.mode not in IMPLEMENTED_MODES:
        raise NotImplementedError(
            f"--mode {config.mode!r} is not implemented yet (lands in a later phase)"
        )
    return select_time(scores, config)
