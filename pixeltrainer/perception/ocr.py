"""Read on-screen text (PLAN.md §4.7, task 5.2).

Text sits on the 8x8 tile grid in one fixed font (assets/font.npz, tools/build_font.py). Each tile is
matched exactly first, then to the nearest glyph by Hamming distance (pixels that differ), so a few noisy
pixels still read; a tile further than MAX_DIST from every glyph is not text (None).
"""
from pathlib import Path

import numpy as np

_z = np.load(Path(__file__).resolve().parents[2] / "assets" / "font.npz")
CHARS = [str(c) for c in _z["chars"]]
GLYPHS = _z["glyphs"].reshape(len(CHARS), 64)
_EXACT = {g.tobytes(): c for g, c in zip(GLYPHS, CHARS)}
MAX_DIST = 6


def tiles(frame):
    """(144, 160) -> (18, 20, 64) flattened 8x8 tiles."""
    return frame.reshape(18, 8, 20, 8).swapaxes(1, 2).reshape(18, 20, 64)


def read_tile(tile):
    c = _EXACT.get(np.ascontiguousarray(tile, np.uint8).tobytes())
    if c is not None:
        return c
    d = (GLYPHS != tile.ravel()).sum(1)
    k = int(d.argmin())
    return CHARS[k] if d[k] <= MAX_DIST else None


def read_grid(frame):
    """(18, 20) list of characters (None = not text)."""
    t = tiles(frame)
    return [[read_tile(t[r, c]) for c in range(20)] for r in range(18)]


def read(frame, row, col0=0, col1=20):
    """Text of one tile row between two columns, non-text tiles as spaces, trimmed."""
    t = tiles(frame)
    return "".join(read_tile(t[row, c]) or " " for c in range(col0, col1)).strip()


def cursor(frame, rows=range(18), cols=range(20)):
    """(row, col) tiles showing the menu cursor ▶, in reading order."""
    t = tiles(frame)
    return [(r, c) for r in rows for c in cols if read_tile(t[r, c]) == "▶"]
