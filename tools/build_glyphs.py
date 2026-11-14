"""Build assets/tile_glyphs.npz: tile ID -> 8x8 shade pattern for font, box and HUD tiles (IDs 0x60+).

Usage: python tools/build_glyphs.py   (reads train runs 0-9 of data/raw)
Each ID's pattern is its most common pixel tile at the positions wTileMap gives it; sprites drawn over
the background make a minority of samples disagree. Also the starting point for the OCR font (task 5.1).
"""
import collections
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    pats = collections.defaultdict(collections.Counter)
    for run in range(10):
        d = Path(f"data/raw/{run:02d}")
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not t["stable"] or not (t["font_loaded"] or t["in_battle"]):
                continue
            s = (255 - cv2.imread(str(d / "frames" / f"{t['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85
            tm = np.asarray(t["tilemap"])
            for r, c in zip(*np.nonzero(tm >= 0x60)):
                pats[int(tm[r, c])][s[8 * r:8 * r + 8, 8 * c:8 * c + 8].tobytes()] += 1
    ids = sorted(pats)
    glyphs = np.stack([np.frombuffer(pats[i].most_common(1)[0][0], np.uint8).reshape(8, 8) for i in ids])
    Path("assets").mkdir(exist_ok=True)
    np.savez_compressed("assets/tile_glyphs.npz", ids=np.array(ids), glyphs=glyphs)
    print(f"saved {len(ids)} glyphs to assets/tile_glyphs.npz")


if __name__ == "__main__":
    main()
