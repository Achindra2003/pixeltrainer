"""Screen-state ground truth (task 2.1) from a recorded label line (truth.read + tilemap) and its frame.

Classes: overworld, battle, dialogue, menu, transition. Labels describe what is on screen, so identical
screens get identical labels (the party menu is "menu" in and out of battle).

wTileMap and wFontLoaded can lag the screen (closing the start menu leaves both stale for a frame or
more), so a tile only counts when its pixels match that tile ID's glyph (assets/tile_glyphs.npz,
built by tools/build_glyphs.py). Overworld map tiles use IDs below 0x60; font, box and HUD tiles 0x60+.
Rules, in order:
- one shade covers >= 99% of pixels: transition (door fades, battle fade-in/out)
- in battle: menu if the party menu (HP-bar start 0x71 at (1,4)) or status screen (0x71 at (3,11))
  is visible, else battle (the battle HUD draws 0x71 at (2,2) and (9,10))
- no visible font/box tile: overworld
- the only visible box is the bottom one at (12,0), and rows 0-11 show no font tiles: dialogue
- anything else: menu (start menu, party, Pokedex, status, yes/no)
"""
from pathlib import Path

import numpy as np

CLASSES = ("overworld", "battle", "dialogue", "menu", "transition")
BOX_TL, HP_START = 0x79, 0x71
_z = np.load(Path(__file__).resolve().parent.parent / "assets" / "tile_glyphs.npz")
GLYPHS = dict(zip(_z["ids"].tolist(), _z["glyphs"]))
_BLANK = {i for i, g in GLYPHS.items() if (g == g.flat[0]).all()}  # e.g. space: matches any flat area


def shades(frame):
    """Dataset PNGs store shade s as 255 - 85*s; shade-index frames pass through."""
    f = np.asarray(frame)
    return f if f.max() <= 3 else ((255 - f.astype(int)) // 85).astype(np.uint8)


def visible(tm, s):
    """Bool (18, 20): cell shows its non-blank font/box/HUD glyph on screen."""
    v = np.zeros(tm.shape, bool)
    for r, c in zip(*np.nonzero(tm >= 0x60)):
        g = GLYPHS.get(int(tm[r, c]))
        v[r, c] = g is not None and int(tm[r, c]) not in _BLANK and (s[8 * r:8 * r + 8, 8 * c:8 * c + 8] == g).all()
    return v


def label(t, frame):
    """t: one labels.jsonl record (needs 'tilemap'); frame: (144, 160) shade indices or dataset grey image."""
    s = shades(frame)
    if np.bincount(s.ravel()).max() >= 0.99 * s.size:
        return "transition"
    tm = np.asarray(t["tilemap"])
    v = visible(tm, s)
    if t["in_battle"]:
        party = any(tm[r, c] == HP_START and v[r, c] for r, c in ((1, 4), (3, 11)))
        return "menu" if party else "battle"
    if not t["font_loaded"] or not v.any():
        return "overworld"
    boxes = set(zip(*np.nonzero((tm == BOX_TL) & v)))
    if boxes == {(12, 0)} and not v[:12].any():
        return "dialogue"
    return "menu"


def main():
    """Write data/raw/<run>/screen.json: one label per frame, in frame order."""
    import json

    import cv2
    for d in sorted(Path("data/raw").iterdir()):
        out = []
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            out.append(label(t, cv2.imread(str(d / "frames" / f"{t['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)))
        (d / "screen.json").write_text(json.dumps(out))
        print(d.name, len(out), flush=True)


if __name__ == "__main__":
    main()
