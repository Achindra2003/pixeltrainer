"""Back to 4 shades (PLAN.md §6A.1, task 8.5): thresholds at the histogram valleys.

Shade 0 = white ... 3 = black, as perception expects. The 4 most prominent peaks of the smoothed
log-histogram are the shades; each threshold is the lowest point between neighbouring peaks. Multi-Otsu
minimises within-class variance, so it merges rare shades and splits a large class that is spread out
(clipped white under uneven light, Poisson noise); the log makes a 200-pixel shade a clear peak, and a
spread class has no valley inside it. Measured in evaluation/eval_capture.py. A frame with fewer than 4
peaks (fewer shades on screen) falls back to fixed thresholds midway between the palette greys: exact for a
clean or light-flattened image, wrong under strong gamma. FLAT catches the blank case.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

FLAT = 12  # grey std below which the screen is one shade (blank)
BINS = 128
PALETTE_MIDS = [42.5, 119.0, 204.0]  # midway between the palette greys 0, 85, 153, 255
SMOOTH = 1.5  # bins: Gaussian smoothing of the log-histogram


def quantise(grey):
    """(144, 160) grey (any range, bright = white) -> uint8 shades 0-3."""
    g = np.asarray(grey, np.float32)
    if g.std() < FLAT:
        return np.full(g.shape, 0 if g.mean() > 128 else 3, np.uint8)
    cnt, edges = np.histogram(g, BINS, (float(g.min()), float(g.max()) + 1e-3))
    h = np.pad(gaussian_filter1d(np.log1p(cnt.astype(np.float64)), SMOOTH), 1)  # padded: end bins can peak
    peaks, props = find_peaks(h, prominence=0)
    if len(peaks) < 4:
        t = PALETTE_MIDS
    else:
        peaks = np.sort(peaks[np.argsort(props["prominences"])[-4:]]) - 1
        h = h[1:-1]
        t = [edges[a + int(np.argmin(h[a:b + 1])) + 1] for a, b in zip(peaks[:-1], peaks[1:])]
    return (3 - np.digitize(g, t)).astype(np.uint8)
