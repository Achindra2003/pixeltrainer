"""Screen state from pixels: overworld, battle, dialogue, menu, transition (PLAN.md §4.2).

heuristic(): the hand-written baseline (task 2.2). Text, boxes and HUDs sit on the 8x8 tile grid in a
fixed font, so exact template matches (correlation of an 8x8 glyph at known grid positions) find them.
Glyphs come from assets/tile_glyphs.npz (captured from our own recordings, tools/build_glyphs.py).
"""
from pathlib import Path

import numpy as np

CLASSES = ("overworld", "battle", "dialogue", "menu", "transition")
_z = np.load(Path(__file__).resolve().parents[2] / "assets" / "tile_glyphs.npz")
_GLYPHS = dict(zip(_z["ids"].tolist(), _z["glyphs"]))
_BOX_TL, _HP_START = _GLYPHS[0x79], _GLYPHS[0x71]
_FONT = np.stack([g for g in _GLYPHS.values() if not (g == g.flat[0]).all()])  # drop flat glyphs (space)


def tiles(frame):
    """(144, 160) -> (18, 20, 8, 8) grid of 8x8 tiles."""
    return frame.reshape(18, 8, 20, 8).swapaxes(1, 2)


def heuristic(frame):
    """frame: (144, 160) shade indices 0-3."""
    if np.bincount(frame.ravel(), minlength=4).max() >= 0.99 * frame.size:
        return "transition"  # fades are a flat screen
    t = tiles(frame)
    hp = (t == _HP_START).all(axis=(2, 3))
    if hp[1, 4] or hp[3, 11]:
        return "menu"  # party menu / status screen
    if hp[2, 2] or hp[9, 10]:
        return "battle"  # enemy / own HP bar in the battle HUD
    boxes = set(zip(*np.nonzero((t == _BOX_TL).all(axis=(2, 3)))))
    if not boxes:
        return "overworld"
    top_text = (t[:12, :, None] == _FONT).all(axis=(3, 4)).any()
    if boxes == {(12, 0)} and not top_text:
        return "dialogue"
    return "menu"


MODEL = Path(__file__).resolve().parents[2] / "models" / "screen_state.pt"
_net = None


def make_net():
    """Small strided CNN on the full 144x160 frame. Pooling to 3x3 (not global) keeps where things
    are, e.g. the HUD in the corners vs a text box along the bottom."""
    import torch.nn as nn

    def block(i, o, k):
        return [nn.Conv2d(i, o, k, stride=2, padding=k // 2), nn.BatchNorm2d(o), nn.ReLU()]

    return nn.Sequential(*block(1, 16, 5), *block(16, 32, 3), *block(32, 64, 3), *block(64, 64, 3),
                         nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Linear(576, len(CLASSES)))


def to_input(frames):
    """(N, 144, 160) shade indices -> float tensor (N, 1, 144, 160) in [0, 1]."""
    import torch
    return torch.as_tensor(np.asarray(frames), dtype=torch.float32).unsqueeze(1) / 3


def cnn(frame):
    """frame: (144, 160) shade indices 0-3. Needs models/screen_state.pt (tools/train_screen_state.py)."""
    global _net
    import torch
    if _net is None:
        _net = make_net()
        _net.load_state_dict(torch.load(MODEL, map_location="cpu"))
        _net.eval()
    with torch.inference_mode():
        return CLASSES[int(_net(to_input(frame[None])).argmax())]
