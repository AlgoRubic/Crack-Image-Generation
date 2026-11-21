"""Fixed local-darkness detector shared by image evaluation and mask validation."""
import numpy as np
from scipy.ndimage import binary_opening, uniform_filter


def fissure_mask(rgb, window=23, threshold=24.5, opening=2):
    if window < 1 or window % 2 != 1 or opening < 1 or threshold < 0:
        raise ValueError('Use a positive odd window, nonnegative threshold and positive opening size')
    rgb = np.asarray(rgb, dtype=np.float32)
    gray = .299 * rgb[:, :, 0] + .587 * rgb[:, :, 1] + .114 * rgb[:, :, 2]
    raw = uniform_filter(gray, size=window) - gray > threshold
    return binary_opening(raw, structure=np.ones((opening, opening), dtype=bool))
