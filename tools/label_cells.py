"""Label every unique overworld cell once (task 2.6, needs a human). Saves assets/cell_labels.json.

Usage: python tools/label_cells.py      (needs data/cells.npz from tools/cell_inventory.py)
       python tools/label_cells.py --auto   write only the guesses to assets/cell_labels_auto.json,
                                            which perception/tiles.py uses until the checked labels exist

Each cell is pre-filled: walkable/wall from RAM collision, then, for cells seen on outdoor maps only
(Pallet, Route 1, Viridian use the overworld tileset), a guess from its tile IDs: grass 0x52, water 0x14,
doors 0x1B/0x58, ledges 0x36/0x37 south, 0x27 west, 0x0D/0x1D east. Indoor tilesets reuse those IDs
(0x14 is the house's exit mat). Check the guess and fix it.

Keys (in the image window):
  1 walkable  2 wall  3 grass  4 water  5 door  6 ledge_s  7 ledge_e  8 ledge_w  9 sign
  Enter/Space  accept the shown label and go to the next cell
  Backspace    previous cell
  q / Esc      save and quit (progress is also saved after every key)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("assets/cell_labels.json")
CLASSES = ["walkable", "wall", "grass", "water", "door", "ledge_s", "ledge_e", "ledge_w", "sign"]
GUESS = [({0x52}, "grass"), ({0x14}, "water"), ({0x1B, 0x58}, "door"),
         ({0x36, 0x37}, "ledge_s"), ({0x0D, 0x1D}, "ledge_e"), ({0x27}, "ledge_w")]


OUTDOOR = {"0", "1", "12"}


def prefill(tiles, walkable, maps):
    ids = set(int(x) for x in tiles.ravel())
    for t, name in GUESS if set(str(maps).split(",")) <= OUTDOOR else ():
        if ids & t:
            return name
    return "walkable" if walkable else "wall"


def show(z, i, label):
    run, frame_i, r, c = z["example"][i]
    frame = cv2.imread(f"data/raw/{run:02d}/frames/{frame_i:06d}.png")
    big = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(big, (48 * c - 1, 48 * r - 1), (48 * c + 48, 48 * r + 48), (0, 0, 255), 2)
    cell = cv2.resize(frame[16 * r:16 * r + 16, 16 * c:16 * c + 16], (192, 192), interpolation=cv2.INTER_NEAREST)
    panel = np.full((432, 300, 3), 255, np.uint8)
    panel[10:202, 54:246] = cell
    walk = "walkable" if z["votes"][i][1] else "blocked"
    lines = [f"cell {i + 1}/{len(z['keys'])}  seen {z['count'][i]}x", f"RAM: {walk}", f"label: {label}", ""]
    lines += [f"{k + 1} {name}" for k, name in enumerate(CLASSES)] + ["Enter next  Bksp back  q quit"]
    for k, text in enumerate(lines):
        cv2.putText(panel, text, (10, 225 + 17 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200) if k == 2 else (0, 0, 0), 1 + (k == 2))
    cv2.imshow("label cells", np.hstack([big, panel]))


def main():
    z = np.load("data/cells.npz")
    keys = [str(k) for k in z["keys"]]
    if "--auto" in sys.argv:
        auto = {k: prefill(z["tiles"][i], z["votes"][i][1] > 0, z["maps"][i]) for i, k in enumerate(keys)}
        OUT.with_name("cell_labels_auto.json").write_text(json.dumps(auto, indent=1))
        print(f"{len(auto)} guessed labels -> {OUT.with_name('cell_labels_auto.json')}")
        return
    labels = json.loads(OUT.read_text()) if OUT.exists() else {}
    i = next((n for n, k in enumerate(keys) if k not in labels), len(keys))
    while i < len(keys):
        k = keys[i]
        current = labels.get(k) or prefill(z["tiles"][i], z["votes"][i][1] > 0, z["maps"][i])
        show(z, i, current)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27) or cv2.getWindowProperty("label cells", cv2.WND_PROP_VISIBLE) < 1:
            break
        if ord("1") <= key <= ord("9"):
            labels[k] = CLASSES[key - ord("1")]
        elif key in (13, 32):
            labels[k] = current
            i += 1
        elif key == 8:
            i = max(0, i - 1)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows()
    print(f"{len(labels)}/{len(keys)} cells labelled -> {OUT}")


if __name__ == "__main__":
    main()
