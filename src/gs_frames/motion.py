"""Motion scoring: dense optical flow magnitude between consecutive analysis frames.

Drives `--mode flow`'s motion-adaptive sampling: DIS optical flow (fast
preset) is cheap enough to run on every analysis-resolution frame pair
without a GPU. Reference: morishuz/adaptive-frame-extractor, AliceVision
Meshroom keyframeSelection (see spec's Further Notes for the full list).
"""

from functools import cache
from typing import cast

import cv2
import numpy as np


@cache
def _flow_estimator() -> cv2.DISOpticalFlow:
    """Built once and reused: constructing a DISOpticalFlow is not free, and
    flow_median runs once per analyzed frame pair -- thousands of times on a
    long clip (see spec user story 6)."""
    return cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)


def flow_median(prev_gray: np.ndarray, curr_gray: np.ndarray, *, cell_size: int = 16) -> float:
    """Median of per-cell mean flow magnitude between two grayscale frames.

    Cell-based aggregation damps outlier pixels (e.g. a moving subject in an
    otherwise-static scene) while still capturing camera-motion-scale
    magnitude, per the spec's Motion section.
    """
    # calc's third argument is an optional output array; cv2's stub requires
    # an ndarray rather than None even though None is the normal "allocate
    # for me" call at runtime.
    flow = _flow_estimator().calc(prev_gray, curr_gray, cast(np.ndarray, None))
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])

    h, w = magnitude.shape[:2]
    cell_means = [
        float(magnitude[y : y + cell_size, x : x + cell_size].mean())
        for y in range(0, h, cell_size)
        for x in range(0, w, cell_size)
    ]
    return float(np.median(cell_means)) if cell_means else 0.0
