"""Battle screen perception (PLAN.md §4.7, task 5.4): enemy species, HP bars, menu state.

Layout (measured on recorded frames, 8x8 tile rows/cols):
- enemy sprite: 56x56 px at (x 96, y 0), the game's front sprite bottom-aligned and centred;
- HP bars: 48 px of fill, enemy at y 19-20, x 32-79; own at y 75-76, x 96-143 (fill = shade >= 2);
- own HP numbers: row 10, "16/ 21"; enemy name row 0, own name row 7;
- main menu: "▶FIGHT <PK><MN>" / "ITEM RUN" on rows 14 and 16, cursor col 9 or 15;
- fight menu: moves on rows 13-16 from col 6, cursor col 5; "TYPE/" on row 9.
Species: normalised cross-correlation of the sprite box with all 151 front sprites (assets/species.npz,
tools/import_species.py), all at the fixed position, so one matrix product scores them all.
"""
from pathlib import Path

import numpy as np

from pixeltrainer.perception import ocr

_z = np.load(Path(__file__).resolve().parents[2] / "assets" / "species.npz")
SPECIES_IDS = _z["ids"].tolist()
SPECIES_NAMES = [str(n) for n in _z["names"]]


def _normalise(x):
    x = x.reshape(len(x), -1).astype(np.float32)
    x -= x.mean(1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


_TEMPLATES = _normalise(_z["sprites"])
BARS = {"enemy": (19, 32), "own": (75, 96)}  # (y, x0) of the first fill row


def species(frame):
    """(internal species ID, name, NCC score) of the best-matching front sprite."""
    scores = _TEMPLATES @ _normalise(frame[None, 0:56, 96:152])[0]
    k = int(scores.argmax())
    return SPECIES_IDS[k], SPECIES_NAMES[k], float(scores[k])


def hp_fraction(frame, side="enemy"):
    y, x0 = BARS[side]
    return float((frame[y:y + 2, x0:x0 + 48] >= 2).mean())


def own_hp(frame):
    """(hp, max_hp) from the numbers under the own HP bar, or None if not readable."""
    left, _, right = ocr.read(frame, 10, 12, 18).partition("/")
    return (int(left), int(right)) if left.strip().isdigit() and right.strip().isdigit() else None


def menu(frame):
    """'main' (FIGHT/PKMN/ITEM/RUN), 'fight' (move list) or 'text'."""
    if ocr.read(frame, 14, 10, 15) == "FIGHT":
        return "main"
    if ocr.read(frame, 9, 1, 6) == "TYPE/":
        return "fight"
    return "text"


def moves(frame):
    """Move names in the fight menu, top to bottom ('-' slots dropped)."""
    names = [ocr.read(frame, r, 6, 19) for r in range(13, 17)]
    return [n for n in names if n and n != "-"]


def cursor(frame):
    """Main menu: index 0-3 (FIGHT, PKMN, ITEM, RUN); fight menu: move index; None if not found."""
    m = menu(frame)
    if m == "main":
        spots = [(14, 9), (14, 15), (16, 9), (16, 15)]
    elif m == "fight":
        spots = [(r, 5) for r in range(13, 17)]
    else:
        return None
    found = ocr.cursor(frame, rows=range(13, 17), cols=(5, 9, 15))
    return spots.index(found[0]) if found and found[0] in spots else None
