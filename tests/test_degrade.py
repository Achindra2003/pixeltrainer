import cv2
import numpy as np
import pytest

from evaluation.degrade import LEVELS, PALETTE, TYPES, degrade


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return np.kron(rng.integers(0, 4, (9, 10)), np.ones((16, 16))).astype(np.uint8)  # 16x16 blocks of shades


def rectify(img, corners):
    H = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [160, 0], [160, 144], [0, 144]]))
    grey = cv2.cvtColor(cv2.warpPerspective(img, H, (160, 144)), cv2.COLOR_BGR2GRAY)
    return np.abs(grey[..., None].astype(int) - PALETTE.astype(int)).argmin(-1)


def test_seeded(frame):
    a, _ = degrade(frame, LEVELS[4], seed=3)
    b, _ = degrade(frame, LEVELS[4], seed=3)
    c, _ = degrade(frame, LEVELS[4], seed=4)
    assert a.shape == (360, 400, 3) and a.dtype == np.uint8
    assert (a == b).all() and (a != c).any()


@pytest.mark.parametrize("types", [LEVELS[0], {"geometric": 4}])
def test_true_corners_invert_the_warp(frame, types):
    img, meta = degrade(frame, types, seed=1)
    assert (rectify(img, meta["corners"]) == frame).mean() > 0.97  # only block edges may resample wrong


def test_saltpepper_density(frame):
    img, _ = degrade(frame, {"saltpepper": 2}, seed=0)
    clean, _ = degrade(frame, {}, seed=0)
    changed = (img != clean).any(-1).mean()
    assert 0.01 < changed < 0.02  # 2% of pixels set to 0 or 255; some already had that value


def test_every_type_changes_the_image(frame):
    clean, _ = degrade(frame, {}, seed=0)
    for t in TYPES:
        assert (degrade(frame, {t: 1}, seed=0)[0] != clean).any(), t
    with pytest.raises(ValueError):
        degrade(frame, {"fog": 1})
