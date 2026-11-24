"""Restoration of the captured image before rectify + quantise (PLAN.md §6A.1, task 8.4).

Model g = h * f + n (degrade.py). All functions take and return float32 grey (or BGR for flatten_light)
in 0-255. The PSNR/SSIM table of every method x degradation type is evaluation/eval_restore.py; restore()
is the chain the agent uses, picked from that table:
  1. flatten_light: divide by the background model of localise.py (a plane per channel fitted to the image
     border). Uneven light and colour cast multiply the whole image, screen and background alike, so this
     cancels both (homomorphic idea: a product becomes removable without a log).
  2. wiener with a Gaussian PSF whose sigma is chosen blind: the sigma whose rectified, quantised result
     is most 4-shade-like (least within-class variance; too little deblur blends shades, too much rings).
Measured and left out of the chain (eval_capture.py, 30 frames, true corners): an impulse filter (a pixel
unlike all 8 neighbours) and adaptive_median erode single screen pixels, which after the warp are ~1 image
px spikes, while rectify's centre sampling already reads salt and pepper strength 2 at 98.8%; notch_moire on
the background region fires on every level-0 frame (the masked screen's box edges leave sinc ripples) and
gains nothing on moiré (96.7% with and without).
"""
import cv2
import numpy as np
import pywt
from skimage.restoration import wiener as _wiener

from pixeltrainer.capture.localise import background

WIENER_SIGMAS = (0.0, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8)  # px: candidate blur widths (0 = no deconvolution)
WIENER_BALANCE = 0.003  # noise-to-signal power; measured best for level 1 (eval_restore.py)
NOTCH_RATIO = 8.0  # a spectral peak this many times its ring's median counts as moiré
NOTCH_MIN_R = 12  # px of spectrum radius: lower frequencies are screen content (8 px tiles at 2x = 0.06)


# --- spatial smoothing (U2-T1) ---
def mean(g, k=3):
    return cv2.blur(g, (k, k))


def gaussian(g, sigma=1.0):
    return cv2.GaussianBlur(g, (0, 0), sigma)


def median(g, k=3):
    return cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), k).astype(np.float32)


def adaptive_median(g, max_k=7):
    """Gonzalez adaptive median: per pixel, grow the window until its median is not an extreme (min/max),
    then keep the pixel unless it is itself an extreme. Only impulses change."""
    g = np.asarray(g, np.float32)
    out = g.copy()
    todo = np.ones(g.shape, bool)
    for k in range(3, max_k + 1, 2):
        pad = np.pad(g, k // 2, mode="reflect")
        win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
        lo, hi, med = win.min((2, 3)), win.max((2, 3)), np.median(win, (2, 3))
        ok = todo & (med > lo) & (med < hi)
        impulse = (g <= lo) | (g >= hi)
        out[ok & impulse] = med[ok & impulse]
        todo &= ~ok
    return out


# --- sharpening (U2-T1) ---
def unsharp(g, sigma=1.0, amount=1.0):
    return g + amount * (g - cv2.GaussianBlur(g, (0, 0), sigma))


def laplacian_sharpen(g, c=1.0):
    return g - c * cv2.Laplacian(g, cv2.CV_32F)


# --- grey-level and histogram (U2-T1) ---
def gamma(g, y):
    return 255 * (np.clip(g, 0, 255) / 255) ** y


def equalise(g):
    return cv2.equalizeHist(np.clip(g, 0, 255).astype(np.uint8)).astype(np.float32)


def clahe(g, clip=2.0, tiles=8):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return c.apply(np.clip(g, 0, 255).astype(np.uint8)).astype(np.float32)


# --- frequency domain (U2-T2) ---
def _radius(shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return yy - h // 2, xx - w // 2


def _apply_spectrum(g, H):
    F = np.fft.fftshift(np.fft.fft2(g))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H))).astype(np.float32)


def gaussian_lowpass(g, d0=60):
    yy, xx = _radius(g.shape)
    return _apply_spectrum(g, np.exp(-(yy ** 2 + xx ** 2) / (2 * d0 ** 2)))


def butterworth_lowpass(g, d0=60, n=2):
    yy, xx = _radius(g.shape)
    return _apply_spectrum(g, 1 / (1 + (np.sqrt(yy ** 2 + xx ** 2) / d0) ** (2 * n)))


def notch_moire(g, screen_mask=None, width=3.0):
    """Gaussian notch pair at the strongest spectral peak beyond NOTCH_MIN_R, if it stands NOTCH_RATIO times
    above the median magnitude of its ring (a periodic overlay is one sharp peak; content is spread). The
    peak is searched in the spectrum of g with screen_mask (True = screen) set to the background mean."""
    src = g if screen_mask is None else np.where(screen_mask, g[~screen_mask].mean(), g)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(src - src.mean())))
    yy, xx = _radius(g.shape)
    r = np.sqrt(yy ** 2 + xx ** 2)
    cand = np.where(r >= NOTCH_MIN_R, mag, 0)
    py, px = np.unravel_index(np.argmax(cand), cand.shape)
    ring = mag[np.abs(r - r[py, px]) < 2]
    if mag[py, px] < NOTCH_RATIO * np.median(ring):
        return g
    dy, dx = yy[py, px], xx[py, px]
    H = 1 - np.exp(-((yy - dy) ** 2 + (xx - dx) ** 2) / (2 * width ** 2))           - np.exp(-((yy + dy) ** 2 + (xx + dx) ** 2) / (2 * width ** 2))
    return _apply_spectrum(g, np.clip(H, 0, 1))


def wavelet_denoise(g, wavelet="haar", level=2):
    """Wavelet shrinkage (U2-T2): 2-level DWT, soft-threshold each detail band (BayesShrink: t = sigma^2 /
    signal sigma of the band), sigma from the median absolute finest diagonal band / 0.6745, inverse DWT.
    Haar: its box basis fits the flat-shaded tiles (on a synthetic tile image, mean error 3.9 vs 10.6 for db2)."""
    c = pywt.wavedec2(g, wavelet, level=level)
    sigma = np.median(np.abs(c[-1][2])) / 0.6745
    if sigma == 0:  # no noise measured: a zero threshold would divide 0 by 0 inside pywt
        return g.astype(np.float32)
    shrink = [tuple(pywt.threshold(d, sigma ** 2 / np.sqrt(max(d.var() - sigma ** 2, 1e-6)), "soft") for d in band)
              for band in c[1:]]
    return pywt.waverec2([c[0]] + shrink, wavelet)[:g.shape[0], :g.shape[1]].astype(np.float32)


def gaussian_psf(sigma):
    k = cv2.getGaussianKernel(2 * int(np.ceil(3 * sigma)) + 1, sigma)
    return (k @ k.T).astype(np.float64)


def wiener(g, sigma, balance=WIENER_BALANCE):
    if sigma <= 0:
        return np.asarray(g, np.float32)
    return (_wiener(np.asarray(g, np.float64) / 255, gaussian_psf(sigma), balance, clip=False) * 255).astype(np.float32)


# --- light and averaging ---
def flatten_light(image):
    """Captured BGR or grey -> grey with uneven light and colour cast divided out (see module doc)."""
    img = image.astype(np.float32)
    bg = np.maximum(background(image).astype(np.float32), 1)
    if img.ndim == 3:
        img = img / bg.reshape(img.shape) * bg.reshape(img.shape).mean((0, 1))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img / bg.reshape(img.shape) * bg.mean()


def frame_average(images):
    """N-frame averaging of a static scene: noise std falls by sqrt(N)."""
    return np.mean([np.asarray(i, np.float32) for i in images], 0)


# --- the chain ---
def _shade_spread(r):
    """Within-class / total variance after quantising: low when the screen reads as 4 clean shades."""
    from pixeltrainer.capture.quantise import quantise
    q = quantise(r)
    total = r.var()
    if total == 0:
        return 0.0
    within = sum(r[q == s].var() * np.mean(q == s) for s in range(4) if (q == s).any())
    return within / total


def restore(image, corners):
    """Captured image + screen corners -> (restored grey image, chosen Wiener sigma)."""
    from pixeltrainer.capture.rectify import rectify
    g = flatten_light(image)
    best = min(((_shade_spread(rectify(w, corners)), s, w) for s in WIENER_SIGMAS for w in [wiener(g, s)]),
               key=lambda x: x[0])
    return best[2], best[1]
