import cv2
import numpy as np

from gs_frames.sharpness import combined_scores, laplacian_variance, tenengrad


def _checkerboard(size: int = 128, cell: int = 8) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y : y + cell, x : x + cell] = 255
    return img


def test_tenengrad_ranks_sharp_above_blurry():
    sharp = _checkerboard()
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert tenengrad(sharp) > tenengrad(blurry)


def test_laplacian_variance_ranks_sharp_above_blurry():
    sharp = _checkerboard()
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert laplacian_variance(sharp) > laplacian_variance(blurry)


def test_combined_ranks_sharp_above_blurry():
    sharp = _checkerboard()
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
    t = np.array([tenengrad(sharp), tenengrad(blurry)])
    l = np.array([laplacian_variance(sharp), laplacian_variance(blurry)])
    combined = combined_scores(t, l)
    assert combined[0] > combined[1]


def test_flat_image_scores_near_zero_sharpness():
    # center_weight=False: the Hann window itself introduces spatial gradient
    # on a flat image by design, so isolate the raw operator here.
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert tenengrad(flat, center_weight=False) == 0
    assert laplacian_variance(flat, center_weight=False) == 0
