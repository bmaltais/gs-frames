from pathlib import Path

import numpy as np

from gs_frames.motion import flow_median
from gs_frames.select import select_flow
from gs_frames.types import ExtractConfig, FrameScore


def _textured_frame(size: int = 128, seed: int = 0) -> np.ndarray:
    """Random texture (not a periodic checkerboard) so DIS optical flow has
    unambiguous gradients to lock onto everywhere in the frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def test_flow_median_is_near_zero_for_identical_frames():
    frame = _textured_frame()
    assert flow_median(frame, frame) < 0.5


def test_flow_median_increases_with_pan_magnitude():
    # Shifts kept small/within DIS's reliable search range -- large jumps on
    # pure random noise (no coherent structure across scales) can make its
    # hierarchical search undershoot, which would make this comparison flaky.
    base = _textured_frame()
    small_pan = flow_median(base, np.roll(base, shift=2, axis=1))
    large_pan = flow_median(base, np.roll(base, shift=6, axis=1))
    assert large_pan > small_pan


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def _scores_from_sequence(frames: list[np.ndarray]) -> list[FrameScore]:
    scores = []
    prev = None
    for i, frame in enumerate(frames):
        flow = flow_median(prev, frame) if prev is not None else 0.0
        scores.append(
            FrameScore(
                index=i,
                timestamp_s=i * 0.1,
                sharpness=float(i),
                sharpness_tenengrad=float(i),
                sharpness_laplacian=float(i),
                flow_median=flow,
            )
        )
        prev = frame
    return scores


def test_select_flow_selects_more_frames_during_motion_than_stillness():
    base = _textured_frame()
    static_sequence = [base.copy() for _ in range(20)]
    panned_sequence = [np.roll(base, shift=i * 4, axis=1) for i in range(20)]

    still_scores = _scores_from_sequence(static_sequence)
    moving_scores = _scores_from_sequence(panned_sequence)

    config = _config(flow_trigger=5.0)
    still_selections = select_flow(still_scores, config)
    moving_selections = select_flow(moving_scores, config)

    assert len(still_selections) == 1  # only the seed frame when static
    assert len(moving_selections) > len(still_selections)
