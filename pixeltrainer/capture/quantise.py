"""Back to 4 shades (PLAN.md §6A.1, task 8.5): multi-level Otsu thresholds at the histogram valleys.

Shade 0 = white ... 3 = black, as perception expects. A frame that holds fewer than 4 shades (a fade, a
blank text box) still gets 3 thresholds, so its shades can come out wrong; FLAT catches the blank case.
"""
import numpy as np
from skimage.filters import threshold_multiotsu

FLAT = 12  # grey std below which the screen is one shade (blank)


def quantise(grey):
    """(144, 160) grey (any range, bright = white) -> uint8 shades 0-3."""
    g = np.asarray(grey, np.float32)
    if g.std() < FLAT:
        return np.full(g.shape, 0 if g.mean() > 128 else 3, np.uint8)
    t = threshold_multiotsu(g, classes=4, nbins=256)
    t = t + (g.max() - g.min()) / 512  # skimage gives bin centres; a value on its bin centre must stay below
    return (3 - np.digitize(g, t)).astype(np.uint8)
