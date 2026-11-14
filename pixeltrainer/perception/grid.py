"""Cut a stable overworld frame into the 9x10 grid of 16x16 cells (grid aligned at (0,0), measured in Phase 0)."""
import hashlib

import numpy as np

ROWS, COLS, CELL = 9, 10, 16
PLAYER = (4, 4)  # (row, col) of the player's cell; the sprite is drawn 4 px above it


def cells(frame):
    """(144, 160) shade frame -> (9, 10, 16, 16) array of cells."""
    return frame.reshape(ROWS, CELL, COLS, CELL).swapaxes(1, 2)


def cell_key(cell):
    """Stable short ID for a cell's exact content (shade indices)."""
    return hashlib.blake2b(np.ascontiguousarray(cell, np.uint8).tobytes(), digest_size=8).hexdigest()


def keys(frame):
    """(9, 10) array of cell IDs."""
    c = cells(frame)
    return np.array([[cell_key(c[r, k]) for k in range(COLS)] for r in range(ROWS)])
