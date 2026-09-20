from pathlib import Path

import numpy as np
import pytest

from gs_frames.overlap import (
    OrbFeatureCache,
    extract_orb_features,
    flow_proxy_overlap,
    homography_overlap,
    make_overlap_fn,
    orb_overlap,
)
from gs_frames.types import ExtractConfig


def _textured_frame(size: int = 256, seed: int = 0) -> np.ndarray:
    """Random texture: dense, unambiguous corners everywhere, so ORB always
    has plenty to detect regardless of translation/scale."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


def _config(**overrides) -> ExtractConfig:
    defaults = dict(video=Path(__file__), output_dir=Path("/tmp/unused"))
    defaults.update(overrides)
    return ExtractConfig(**defaults)


def test_extract_orb_features_finds_keypoints_in_textured_image():
    features = extract_orb_features(_textured_frame(), orb_nfeatures=500)
    assert len(features.keypoints) > 0
    assert features.descriptors is not None
    assert len(features.descriptors) == len(features.keypoints)


def test_extract_orb_features_blank_image_has_no_descriptors():
    blank = np.zeros((256, 256), dtype=np.uint8)
    features = extract_orb_features(blank, orb_nfeatures=500)
    assert features.descriptors is None


def test_orb_overlap_identical_frames_is_high():
    frame = _textured_frame()
    features = extract_orb_features(frame, orb_nfeatures=1000)
    assert orb_overlap(features, features, match_ratio=0.75) > 0.7


def test_orb_overlap_decreases_as_frames_diverge():
    base = _textured_frame()
    features_base = extract_orb_features(base, orb_nfeatures=1000)
    features_small_shift = extract_orb_features(
        np.roll(base, shift=3, axis=1), orb_nfeatures=1000
    )
    features_large_shift = extract_orb_features(
        np.roll(base, shift=120, axis=1), orb_nfeatures=1000
    )
    small_overlap = orb_overlap(features_base, features_small_shift, match_ratio=0.75)
    large_overlap = orb_overlap(features_base, features_large_shift, match_ratio=0.75)
    assert small_overlap is not None and large_overlap is not None
    assert small_overlap > large_overlap


def test_orb_overlap_independent_noise_images_is_much_lower_than_identical():
    """Two unrelated noise images (not a shift of one another) should match
    far less than a frame against itself -- the "noise image pairs" case
    from the spec's Testing Decisions, distinct from the shifted-same-image
    cases above."""
    features_a = extract_orb_features(_textured_frame(seed=1), orb_nfeatures=1000)
    features_b = extract_orb_features(_textured_frame(seed=2), orb_nfeatures=1000)
    identical_overlap = orb_overlap(features_a, features_a, match_ratio=0.75)
    independent_overlap = orb_overlap(features_a, features_b, match_ratio=0.75)
    assert identical_overlap is not None and independent_overlap is not None
    assert independent_overlap < identical_overlap * 0.5


def test_orb_overlap_returns_none_when_a_frame_has_no_features():
    blank = np.zeros((256, 256), dtype=np.uint8)
    features_textured = extract_orb_features(_textured_frame(), orb_nfeatures=500)
    features_blank = extract_orb_features(blank, orb_nfeatures=500)
    assert orb_overlap(features_textured, features_blank, match_ratio=0.75) is None


def test_homography_overlap_identical_frames_is_high():
    frame = _textured_frame()
    features = extract_orb_features(frame, orb_nfeatures=1000)
    overlap = homography_overlap(
        features, features, frame.shape[:2], match_ratio=0.75, ransac_reproj_threshold=3.0
    )
    assert overlap is not None and overlap > 0.7


def test_homography_overlap_falls_back_to_orb_when_too_few_matches():
    blank = np.zeros((256, 256), dtype=np.uint8)
    features_textured = extract_orb_features(_textured_frame(), orb_nfeatures=500)
    features_blank = extract_orb_features(blank, orb_nfeatures=500)
    assert (
        homography_overlap(
            features_textured,
            features_blank,
            (256, 256),
            match_ratio=0.75,
            ransac_reproj_threshold=3.0,
        )
        is None
    )


def test_polygon_iou_returns_none_for_degenerate_polygon():
    from gs_frames.overlap import _polygon_iou

    rect_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    degenerate_b = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
    assert _polygon_iou(rect_a, degenerate_b) is None


def test_homography_overlap_falls_back_to_orb_on_degenerate_footprint(monkeypatch):
    """A degenerate (zero-area) warped footprint is a homography-fit
    failure, not a genuine zero-overlap answer -- it must fall back to
    orb_overlap rather than silently reporting a confident 0.0."""
    import gs_frames.overlap as overlap_module

    frame = _textured_frame()
    features = extract_orb_features(frame, orb_nfeatures=1000)
    monkeypatch.setattr(overlap_module, "_polygon_iou", lambda rect_a, polygon_b: None)

    result = overlap_module.homography_overlap(
        features, features, frame.shape[:2], match_ratio=0.75, ransac_reproj_threshold=3.0
    )
    assert result == orb_overlap(features, features, match_ratio=0.75)


def test_homography_overlap_matches_candidate_into_reference_direction():
    """Asymmetric-shift case (spec's testing decision) to catch a reversed
    homography A/B direction: B is a zoomed-in crop of A's top-left quadrant
    (a real "candidate moved closer" scenario, not a symmetric translation
    that would hide a reversed convention). Warping B's own frame into A's
    coordinate space should land it approximately within A's top-left
    quadrant, well short of full coverage -- generous bounds, since ORB/
    RANSAC introduce real run-to-run variance on synthetic noise.
    """
    size = 256
    full = _textured_frame(size=size, seed=1)
    zoomed_crop = full[: size // 2, : size // 2]
    zoomed = np.repeat(np.repeat(zoomed_crop, 2, axis=0), 2, axis=1)

    features_full = extract_orb_features(full, orb_nfeatures=2000)
    features_zoomed = extract_orb_features(zoomed, orb_nfeatures=2000)

    overlap = homography_overlap(
        features_full,
        features_zoomed,
        full.shape[:2],
        match_ratio=0.8,
        ransac_reproj_threshold=5.0,
    )
    # If the direction were reversed, B's (zoomed's) full frame would appear
    # to cover roughly all of A -- this stays well below "near total".
    assert overlap is None or overlap < 0.7


def test_flow_proxy_overlap_decreases_with_more_accumulated_flow():
    low = flow_proxy_overlap(1.0, flow_trigger=8.0)
    high = flow_proxy_overlap(20.0, flow_trigger=8.0)
    assert 0.0 <= high < low <= 1.0


def test_flow_proxy_overlap_is_clamped_to_zero_and_one():
    assert flow_proxy_overlap(0.0, flow_trigger=8.0) == 1.0
    assert flow_proxy_overlap(1000.0, flow_trigger=8.0) == 0.0


def test_make_overlap_fn_rejects_flow_and_none_metrics():
    with pytest.raises(ValueError):
        make_overlap_fn(lambda i: _textured_frame(), _config(overlap_metric="flow"))
    with pytest.raises(ValueError):
        make_overlap_fn(lambda i: _textured_frame(), _config(overlap_metric="none"))


def test_make_overlap_fn_caches_orb_features_per_index():
    frames = {0: _textured_frame(seed=0), 1: _textured_frame(seed=1), 2: _textured_frame(seed=2)}
    calls: list[int] = []

    def get_gray(index: int) -> np.ndarray:
        calls.append(index)
        return frames[index]

    overlap_fn = make_overlap_fn(get_gray, _config(overlap_metric="orb"))
    overlap_fn(0, 1)
    overlap_fn(0, 2)

    # get_gray is called once per (index, comparison) via the cache miss path
    # plus once more for shape lookups (orb metric doesn't need shape, so
    # only the two comparisons' four feature extractions matter here): index
    # 0's features must only be *extracted* once even though it's compared
    # against twice.
    cache = OrbFeatureCache(get_gray, orb_nfeatures=500)
    calls.clear()
    cache(0)
    cache(0)
    cache(1)
    assert calls == [0, 1]
