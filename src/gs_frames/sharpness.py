"""Sharpness scoring: Tenengrad, Laplacian variance, and a per-video combined score.

Reference: Tenengrad is what cansik/sharp-frame-extractor uses; Laplacian
variance is what Kotohibi and many blur papers use; Reflct combines both.
"""

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _center_weighted(gray: np.ndarray) -> np.ndarray:
    """Multiply by a 2D Hann window so peripheral/sky content weighs less."""
    h, w = gray.shape[:2]
    wy = np.hanning(h) if h > 1 else np.ones(h)
    wx = np.hanning(w) if w > 1 else np.ones(w)
    window = np.outer(wy, wx)
    return gray.astype(np.float64) * window


def laplacian_variance(image: np.ndarray, *, center_weight: bool = True) -> float:
    gray = to_gray(image)
    weighted = _center_weighted(gray) if center_weight else gray.astype(np.float64)
    lap = cv2.Laplacian(weighted, cv2.CV_64F, ksize=3)
    return float(np.var(lap))


def tenengrad(image: np.ndarray, *, center_weight: bool = True) -> float:
    gray = to_gray(image)
    weighted = _center_weighted(gray) if center_weight else gray.astype(np.float64)
    gx = cv2.Sobel(weighted, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(weighted, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def combined_scores(tenengrad_values: np.ndarray, laplacian_values: np.ndarray) -> np.ndarray:
    """Per-video z-score blend: 0.6 * z(tenengrad) + 0.4 * z(laplacian)."""
    t = np.asarray(tenengrad_values, dtype=np.float64)
    l = np.asarray(laplacian_values, dtype=np.float64)
    z_t = (t - t.mean()) / (t.std() + 1e-8)
    z_l = (l - l.mean()) / (l.std() + 1e-8)
    return 0.6 * z_t + 0.4 * z_l
