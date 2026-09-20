"""Overlap estimation between two candidate frames.

Backs `--mode overlap-greedy`'s in-range/fallback-closest decision via the
`overlap_fn(last_kept_index, candidate_index) -> Optional[float]` callback
injected into select.select_overlap_greedy. `orb` and `homography` need real
image data (ORB keypoints/descriptors), so their machinery lives here, behind
`make_overlap_fn`; the `flow` metric needs no image data at all (it is a
cheap proxy from accumulated flow_median) and is computed directly by
select.py without going through this module's overlap_fn.

Reference: Asif Sijan, *Robust 3D content generation with acquisition aware
frame selection for Gaussian Splatting* (UMN, 2025) -- see spec's Further
Notes for the full reference list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from gs_frames.types import ExtractConfig, OverlapMetric

# overlap_fn(last_kept_index, candidate_index) -> overlap of the candidate
# frame against the last-kept (reference) frame, or None if the measurement
# could not be made (too few features to attempt matching).
OverlapFn = Callable[[int, int], Optional[float]]

# Minimum good matches required to attempt a homography fit; below this,
# findHomography's RANSAC has too few points to be meaningful.
_MIN_MATCHES_FOR_HOMOGRAPHY = 4

# flow_proxy_overlap reaches 0 once accumulated flow_median hits this many
# multiples of --flow-trigger -- the same accumulated-motion bound
# select_overlap_greedy uses to stop growing its candidate pool, since
# that's already the point past which the reference frame is assumed to
# have "moved on".
_FLOW_PROXY_REFERENCE_MULTIPLIER = 3


@dataclass
class OrbFeatures:
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: Optional[np.ndarray]


def extract_orb_features(gray: np.ndarray, *, orb_nfeatures: int) -> OrbFeatures:
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)  # type: ignore[attr-defined]
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return OrbFeatures(keypoints=tuple(keypoints), descriptors=descriptors)


def _good_matches(
    features_a: OrbFeatures, features_b: OrbFeatures, match_ratio: float
) -> list[cv2.DMatch]:
    """Lowe's-ratio-filtered matches, queryIdx into features_a, trainIdx into
    features_b. Empty (not an error) whenever either side has too few
    descriptors to run a 2-NN match."""
    descriptors_a, descriptors_b = features_a.descriptors, features_b.descriptors
    if descriptors_a is None or descriptors_b is None:
        return []
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < match_ratio * second.distance:
            good.append(best)
    return good


def orb_overlap(
    features_a: OrbFeatures, features_b: OrbFeatures, *, match_ratio: float
) -> Optional[float]:
    """good matches / min(nA, nB); None if either frame has no features to
    match against at all."""
    n_a, n_b = len(features_a.keypoints), len(features_b.keypoints)
    if n_a == 0 or n_b == 0:
        return None
    good = _good_matches(features_a, features_b, match_ratio)
    return len(good) / min(n_a, n_b)


def _polygon_iou(rect_a: np.ndarray, polygon_b: np.ndarray) -> Optional[float]:
    """IoU of two convex polygons (Nx2 float32 point arrays), or None if
    either polygon is degenerate (zero area) -- that's a homography-fit
    failure (a near-singular H collapsing B's corners), not a genuine
    zero-overlap answer, so callers should treat it as a measurement failure
    rather than a confident 0.0. convexHull reorders points into the winding
    cv2.intersectConvexConvex expects, and guards against a near-degenerate
    perspective warp producing a self-intersecting quad.
    """
    hull_a = cv2.convexHull(rect_a.astype(np.float32))
    hull_b = cv2.convexHull(polygon_b.astype(np.float32))
    area_a = cv2.contourArea(hull_a)
    area_b = cv2.contourArea(hull_b)
    if area_a <= 0 or area_b <= 0:
        return None

    intersection_area, _ = cv2.intersectConvexConvex(hull_a, hull_b)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return None
    return float(np.clip(intersection_area / union_area, 0.0, 1.0))


def homography_overlap(
    features_a: OrbFeatures,
    features_b: OrbFeatures,
    frame_shape: tuple[int, int],
    *,
    match_ratio: float,
    ransac_reproj_threshold: float,
) -> Optional[float]:
    """IoU of candidate frame B's footprint (its own corners, warped by a
    homography H into reference frame A's coordinate space) against A's
    rectangle. H is fit as B -> A (findHomography(src=B points, dst=A
    points)) -- a fixed direction convention, not interchangeable with A -> B,
    since the two give different answers for any perspective/scale change
    that isn't a pure translation. Falls back to orb_overlap whenever there
    aren't enough matches to fit a homography, the fit fails, it produces
    non-finite corners, or the resulting footprint is degenerate (zero area).
    """
    def fallback() -> Optional[float]:
        return orb_overlap(features_a, features_b, match_ratio=match_ratio)

    good = _good_matches(features_a, features_b, match_ratio)
    if len(good) < _MIN_MATCHES_FOR_HOMOGRAPHY:
        return fallback()

    points_a = np.array(
        [features_a.keypoints[m.queryIdx].pt for m in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    points_b = np.array(
        [features_b.keypoints[m.trainIdx].pt for m in good], dtype=np.float32
    ).reshape(-1, 1, 2)
    h_mat, _mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, ransac_reproj_threshold)
    if h_mat is None:
        return fallback()

    height, width = frame_shape
    corners_b = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped_b = cv2.perspectiveTransform(corners_b, h_mat)
    if warped_b is None or not np.all(np.isfinite(warped_b)):
        return fallback()

    rect_a = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    iou = _polygon_iou(rect_a, warped_b.reshape(-1, 2))
    return iou if iou is not None else fallback()


def flow_proxy_overlap(flow_accum: float, flow_trigger: float) -> float:
    """Heuristic overlap in [0, 1] for --overlap-metric flow: no image
    matching at all, just a monotonically decreasing function of accumulated
    flow_median since the reference (last-kept) frame.
    """
    reference = _FLOW_PROXY_REFERENCE_MULTIPLIER * flow_trigger
    if reference <= 0:
        return 0.0
    return float(np.clip(1.0 - flow_accum / reference, 0.0, 1.0))


class OrbFeatureCache:
    """Caches OrbFeatures per frame index for the duration of one selection
    pass, so a frame revisited as a later last-kept frame -- or compared
    against several candidates in the same pool -- is only ever ORB-detected
    once."""

    def __init__(self, get_gray: Callable[[int], np.ndarray], orb_nfeatures: int):
        self._get_gray = get_gray
        self._orb_nfeatures = orb_nfeatures
        self._cache: dict[int, OrbFeatures] = {}

    def __call__(self, index: int) -> OrbFeatures:
        features = self._cache.get(index)
        if features is None:
            features = extract_orb_features(self._get_gray(index), orb_nfeatures=self._orb_nfeatures)
            self._cache[index] = features
        return features


METRICS_NEEDING_IMAGES: frozenset[OverlapMetric] = frozenset({"orb", "homography"})


def make_overlap_fn(get_gray: Callable[[int], np.ndarray], config: ExtractConfig) -> OverlapFn:
    """Builds the overlap_fn for select_overlap_greedy when
    config.overlap_metric is "orb" or "homography" (the metrics that need
    real frame data); "flow" and "none" never call this.
    """
    if config.overlap_metric not in METRICS_NEEDING_IMAGES:
        raise ValueError(
            f"make_overlap_fn does not support overlap_metric={config.overlap_metric!r} "
            "(flow/none are computed by select.py without image data)"
        )

    cache = OrbFeatureCache(get_gray, config.orb_nfeatures)

    def overlap_fn(last_kept_index: int, candidate_index: int) -> Optional[float]:
        features_a = cache(last_kept_index)
        features_b = cache(candidate_index)
        if config.overlap_metric == "orb":
            return orb_overlap(features_a, features_b, match_ratio=config.match_ratio)
        frame_shape = get_gray(last_kept_index).shape[:2]
        return homography_overlap(
            features_a,
            features_b,
            (frame_shape[0], frame_shape[1]),
            match_ratio=config.match_ratio,
            ransac_reproj_threshold=config.ransac_reproj_threshold,
        )

    return overlap_fn
