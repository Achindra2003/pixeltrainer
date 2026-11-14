import numpy as np
import pytest

from pixeltrainer.mapping import registration

# A world of random 16x16 cells; the view is a 144x160 window, the player at cell (4, 4).
WORLD = np.kron(np.random.default_rng(0).integers(0, 4, (40, 40)), np.ones((16, 16), np.uint8)).astype(np.uint8)


def view(col, row):
    return WORLD[16 * row:16 * row + 144, 16 * col:16 * col + 160].copy()


@pytest.mark.parametrize("direction, move", [("right", (1, 0)), ("up", (0, -1)), ("down", (0, 2)), ("left", (0, 0))])
def test_direction_prior(direction, move):
    a = view(10, 10)
    b = view(10 + move[0], 10 + move[1])
    assert registration.step(a, b, direction) == move


def test_phase_correlation_without_prior():
    assert registration.step(view(10, 10), view(11, 10)) == (1, 0)
    assert registration.step(view(10, 10), view(10, 10)) == (0, 0)


def test_unrelated_frames_are_rejected():
    blank = np.zeros((144, 160), np.uint8)
    assert registration.step(view(10, 10), blank, "up") is None
    assert registration.step(view(10, 10), view(30, 30), "up") is None
