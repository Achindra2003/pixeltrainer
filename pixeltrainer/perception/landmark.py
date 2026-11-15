"""Goal landmarks by normalised cross-correlation (PLAN.md §4.6, task 4.3).

The Pokémon Center "POKé" sign (assets/landmarks/poke_sign.png, tools/capture_landmark.py) marks
Viridian City on the way north from Pallet Town.
"""
from pathlib import Path

import cv2
import numpy as np

_DIR = Path(__file__).resolve().parents[2] / "assets" / "landmarks"
THRESHOLD = 0.9  # NCC score above which the landmark counts as seen


def load(name):
    return ((255 - cv2.imread(str(_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)) // 85).astype(np.float32)


_TEMPLATES = {}


def find(frame, name="poke_sign"):
    """(score, (x, y)) of the best match of the landmark in a (144, 160) shade frame."""
    if name not in _TEMPLATES:
        _TEMPLATES[name] = load(name)
    r = cv2.matchTemplate(frame.astype(np.float32), _TEMPLATES[name], cv2.TM_CCOEFF_NORMED)
    _, score, _, xy = cv2.minMaxLoc(r)
    return score, xy


def seen(frame, name="poke_sign"):
    return find(frame, name)[0] >= THRESHOLD
