"""OCR character accuracy against RAM wTileMap (task 5.2).

Usage: python evaluation/eval_ocr.py [--runs 15-19]
Frames: stable frames labelled battle, menu or dialogue (data/raw/<run>/screen.json, which checks that
text is drawn: on other frames wTileMap can still hold a closed text box). Every tile whose wTileMap ID is a character
(assets/font.npz) is read from pixels and compared. Spaces are counted separately (they are easy).
RAM and screen disagree on some tiles: a box drawn before its text, sprites (party balls, the trainer)
over text, a flash. So each error is also split by what the pixels show: exactly another glyph or no
glyph at all (the screen differs from RAM), or a near-glyph read as the wrong character (a real misread).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_registration import frame  # noqa: E402
from pixeltrainer.perception import ocr  # noqa: E402

_z = np.load("assets/font.npz")
ID_CHAR = dict(zip(_z["ids"].tolist(), [str(c) for c in _z["chars"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="15-19")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    n, wrong, examples = collections.Counter(), collections.Counter(), []
    for run in range(int(a), int(b or a) + 1):
        d = Path(f"data/raw/{run:02d}")
        screen = json.loads((d / "screen.json").read_text())
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not t["stable"] or screen[t["i"]] not in ("battle", "menu", "dialogue"):
                continue
            n["frames"] += 1
            f = frame(d, t)
            got = ocr.read_grid(f)
            for r, row in enumerate(t["tilemap"]):
                for c, i in enumerate(row):
                    if i not in ID_CHAR:
                        continue
                    kind = "space" if ID_CHAR[i] == " " else "char"
                    n[kind] += 1
                    if got[r][c] != ID_CHAR[i]:
                        n[kind + " wrong"] += 1
                        wrong[(ID_CHAR[i], got[r][c])] += 1
                        tile = ocr.tiles(f)[r, c].astype(np.uint8).tobytes()
                        n["exact other glyph" if tile in ocr._EXACT else "no glyph" if got[r][c] is None
                          else "fuzzy misread"] += 1
                        if len(examples) < 10:
                            examples.append((run, t["i"], r, c, ID_CHAR[i], got[r][c]))
    for kind in ("char", "space"):
        print(f"{kind}: {n[kind] - n[kind + ' wrong']}/{n[kind]} = {1 - n[kind + ' wrong'] / max(n[kind], 1):.4%}")
    total = n["char"] + n["space"]
    print(f"errors by what the pixels show: exactly another glyph {n['exact other glyph']}, no glyph "
          f"{n['no glyph']}, fuzzy misread {n['fuzzy misread']}; drawn-text accuracy "
          f"{1 - n['fuzzy misread'] / total:.4%}")
    print(f"frames {n['frames']}; most common errors (truth, read): {wrong.most_common(10)}")
    print("first errors (run, frame, row, col, truth, read):", examples)


if __name__ == "__main__":
    main()
