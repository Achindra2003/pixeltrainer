"""Tile classes per 16x16 cell (PLAN.md §4.3): exact lookup (task 2.7) and a small CNN (task 2.8).

Classes: walkable, wall, grass, water, door, ledge_s, ledge_e, ledge_w, sign (tools/label_cells.py).
lookup: a cell's exact content (grid.cell_key) maps to its label; an unseen cell is None ("unknown", confidence 0).
cnn: a label for every cell, also unseen ones, with its softmax confidence (models/tiles.pt, tools/train_tiles.py).
Labels come from assets/cell_labels.json (task 2.6); until that exists, from
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


CLASSES = ["walkable", "wall", "grass", "water", "door", "ledge_s", "ledge_e", "ledge_w", "sign"]
PASSABLE = {"walkable", "grass", "door"}  # the labels RAM collision calls walkable (all 101 labelled cells agree)
MODEL = Path(__file__).resolve().parents[2] / "models" / "tiles.pt"
_net = None


def make_net():
    """Small CNN on one 16x16 cell. Flatten (not global pooling) keeps where the pattern sits,
    e.g. a ledge's edge at the bottom of the cell; shift augmentation covers ±1 px."""
    import torch.nn as nn

    def block(i, o, s):
        return [nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.ReLU()]

    return nn.Sequential(*block(1, 16, 1), *block(16, 32, 2), *block(32, 64, 2),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 16, len(CLASSES)))


def to_input(cells):
    """(N, 16, 16) shade indices -> float tensor (N, 1, 16, 16) in [0, 1]."""
    import torch
    return torch.as_tensor(np.asarray(cells), dtype=torch.float32).unsqueeze(1) / 3


def cnn(frame):
    """(9, 10) object array of labels and (9, 10) softmax confidences, for every cell."""
    global _net
    import torch
    if _net is None:
        _net = make_net()
        _net.load_state_dict(torch.load(MODEL, map_location="cpu"))
        _net.eval()
    with torch.inference_mode():
        p = torch.softmax(_net(to_input(grid.cells(frame).reshape(-1, 16, 16))), 1)
    conf, i = p.max(1)
    return np.array(CLASSES, dtype=object)[i.numpy()].reshape(9, 10), conf.numpy().reshape(9, 10)
