import numpy as np

from pixeltrainer.perception import grid


def test_cells_match_pixel_blocks():
    frame = np.zeros((144, 160), np.uint8)
    for r in range(9):
        for c in range(10):
            frame[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = (r + c) % 4
    frame[4 * 16 + 3, 4 * 16 + 5] = 3  # one marked pixel inside the player cell
    cs = grid.cells(frame)
    assert cs.shape == (9, 10, 16, 16)
    for r in range(9):
        for c in range(10):
            assert np.array_equal(cs[r, c], frame[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16])
    assert cs[grid.PLAYER][3, 5] == 3


def test_keys_equal_iff_content_equal():
    frame = np.zeros((144, 160), np.uint8)
    frame[:16, 16:32] = 2  # cell (0,1) differs from cell (0,0)
    k = grid.keys(frame)
    assert k.shape == (9, 10)
    assert k[0, 0] == k[8, 9] and k[0, 0] != k[0, 1]
