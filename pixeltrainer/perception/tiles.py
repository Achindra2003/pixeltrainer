"""Tile classes per 16x16 cell by exact lookup (task 2.7).

Classes: walkable, wall, grass, water, door, ledge_s, ledge_e, ledge_w, sign (tools/label_cells.py).
A cell's exact content (grid.cell_key) maps to its label; an unseen cell is None ("unknown", confidence 0).
Labels come from assets/cell_labels.json (checked by hand, task 2.6); until that exists, from
assets/cell_labels_auto.json (the RAM-based guesses, tools/label_cells.py --auto).
"""
import json
from pathlib import Path

import numpy as np

from pixeltrainer.perception import grid

_ASSETS = Path(__file__).resolve().parents[2] / "assets"
_labels = None


def labels():
    global _labels
    if _labels is None:
        path = _ASSETS / "cell_labels.json"
        # shortcut: falls back to unchecked guesses; drop the fallback once task 2.6 is done.
        if not path.exists():
            path = _ASSETS / "cell_labels_auto.json"
        _labels = json.loads(path.read_text())
    return _labels


def lookup(frame):
    """(9, 10) object array of labels (None = unknown) and (9, 10) confidences (1 hit, 0 miss)."""
    table = labels()
    out = np.array([[table.get(k) for k in row] for row in grid.keys(frame)], dtype=object)
    return out, (out != None).astype(np.float32)  # noqa: E711 (elementwise)
