import numpy as np

from evaluation.degrade import LEVELS, degrade
from pixeltrainer.capture.localise import localise
from pixeltrainer.capture.quantise import quantise
from pixeltrainer.capture.rectify import rectify


def test_level0_round_trip_is_exact():
    rng = np.random.default_rng(0)
    frame = np.kron(rng.integers(0, 4, (18, 20)), np.ones((8, 8), np.int64)).astype(np.uint8)
    frame[:8], frame[-8:], frame[:, :8], frame[:, -8:] = 0, 0, 0, 0  # white border, like most screens
    img, meta = degrade(frame, LEVELS[0])
    corners = localise(img)
    assert np.abs(corners - meta["corners"]).max() <= 1.0
    assert (quantise(rectify(img, meta["corners"])) == frame).all()


def test_wavelet_denoise_reduces_noise_on_tiles():
    from pixeltrainer.capture.restore import wavelet_denoise
    rng = np.random.default_rng(0)
    f = np.kron(rng.integers(0, 4, (18, 20)) * 85.0, np.ones((8, 8))).astype(np.float32)
    g = f + rng.normal(0, 20, f.shape).astype(np.float32)
    out = wavelet_denoise(g)
    assert out.shape == f.shape
    assert np.abs(out - f).mean() < 0.5 * np.abs(g - f).mean()
    assert np.array_equal(wavelet_denoise(f), f)  # no noise measured: returned unchanged
