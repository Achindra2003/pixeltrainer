"""NPC detection with facing (PLAN.md §4.4, task 6.2).

Candidates: 8-connected components of the darkest shade (sprite outlines are shade 3), minus specks. A
standing sprite is drawn 4 px above its cell, so candidates are the sprite boxes (16r - 4, 16c) holding part
of a component. Components are not limited to sprite size: outlines touch dark scenery (fences, trees, the
tile under a sprite), which made 69% of NPCs on runs 0-14 part of a larger component.
Each box is scored by normalised cross-correlation with every template (assets/npc_templates.npz, from
tools/capture_npc_templates.py) over the template's mask; the best template gives the facing. Boxes scoring >= THRESHOLD are kept, then duplicates removed by
overlap (IoU). The player's own box is never a candidate.
"""
from pathlib import Path

import cv2
import numpy as np

from pixeltrainer.perception import grid

THRESHOLD = 0.8  # masked NCC
MIN_AREA = 4  # px
IOU = 0.5

_T = np.load(Path(__file__).resolve().parents[2] / "assets" / "npc_templates.npz")
KEYS = [str(k) for k in _T["keys"]]
_TEMPLATES = _T["templates"].astype(np.float32)
_MASKS = _T["masks"]


def _ncc(patch):
    """Masked NCC of a 16x16 patch with every template -> (K,) scores."""
    out = np.full(len(KEYS), -1.0)
    for i, (t, m) in enumerate(zip(_TEMPLATES, _MASKS)):
        a, b = patch[m] - patch[m].mean(), t[m] - t[m].mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        if d > 0:
            out[i] = (a * b).sum() / d
    return out


def candidates(frame):
    """Sprite boxes (top y, left x) holding part of a dark component."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats((frame == 3).astype(np.uint8), connectivity=8)
    dark = (stats[labels, cv2.CC_STAT_AREA] >= MIN_AREA) & (labels > 0)
    return [(16 * r - 4, 16 * c) for r in range(1, grid.ROWS) for c in range(grid.COLS)
            if (r, c) != grid.PLAYER and dark[16 * r - 4:16 * r + 12, 16 * c:16 * c + 16].any()]


def _iou(a, b):
    dy, dx = 16 - abs(a[0] - b[0]), 16 - abs(a[1] - b[1])
    inter = max(dy, 0) * max(dx, 0)
    return inter / (512 - inter)


def detect(frame, threshold=THRESHOLD):
    """Stable overworld frame (144, 160 shades) -> [{"cell": (r, c), "box": (y, x, 16, 16), "facing", "key",
    "score"}], best first."""
    f = frame.astype(np.float32)
    found = []
    for y, x in candidates(frame):
        s = _ncc(f[y:y + 16, x:x + 16])
        i = int(s.argmax())
        if s[i] >= threshold:
            found.append({"cell": ((y + 4) // 16, x // 16), "box": (y, x, 16, 16), "key": KEYS[i],
                          "facing": KEYS[i].split("_")[1], "score": float(s[i])})
    found.sort(key=lambda d: -d["score"])
    kept = []
    for d in found:
        if all(_iou(d["box"], k["box"]) < IOU for k in kept):
            kept.append(d)
    return kept
