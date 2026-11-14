"""Unique 16x16 overworld cells with RAM walkability (task 2.5).

Usage: python tools/cell_inventory.py [--runs 0-9]
Reads stable frames labelled "overworld" (data/raw/<run>/screen.json). Skips cells under a visible
sprite (player, NPCs): the inventory is background only. A cell is walkable when all 4 of its 8x8
collision tiles are (PyBoy's game_area_collision, recorded per stable frame).
Saves data/cells.npz: keys, cells (N,16,16) shade indices, count, walkable votes, maps seen in,
the cell's 2x2 tile IDs (wTileMap) and one example (run, frame, row, col) for context.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.truth import sprite_cells  # noqa: E402
from pixeltrainer.perception import grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-9")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    pix, count, walk, maps = {}, defaultdict(int), defaultdict(lambda: [0, 0]), defaultdict(set)
    tiles, example = {}, {}
    frames = 0
    for run in range(int(a), int(b or a) + 1):
        d = Path(f"data/raw/{run:02d}")
        screen = json.loads((d / "screen.json").read_text())
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not t["stable"] or screen[t["i"]] != "overworld" or "collision" not in t:
                continue
            frames += 1
            f = (255 - cv2.imread(str(d / "frames" / f"{t['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85
            w = np.asarray(t["collision"]).reshape(9, 2, 10, 2).min(axis=(1, 3)).astype(bool)
            skip = sprite_cells(t)  # includes the player (slot 0)
            cs = grid.cells(f)
            tm = np.asarray(t["tilemap"])
            for r in range(grid.ROWS):
                for c in range(grid.COLS):
                    if (r, c) in skip:
                        continue
                    k = grid.cell_key(cs[r, c])
                    if k not in pix:
                        pix[k] = cs[r, c].copy()
                        tiles[k] = tm[2 * r:2 * r + 2, 2 * c:2 * c + 2].copy()
                        example[k] = (run, t["i"], r, c)
                    count[k] += 1
                    walk[k][bool(w[r, c])] += 1
                    maps[k].add(t["map_id"])
    keys = sorted(pix, key=lambda k: -count[k])
    votes = np.array([walk[k] for k in keys])  # columns: [blocked, walkable]
    np.savez_compressed("data/cells.npz", keys=np.array(keys), cells=np.stack([pix[k] for k in keys]),
                        count=np.array([count[k] for k in keys]), votes=votes,
                        maps=np.array([",".join(map(str, sorted(maps[k]))) for k in keys]),
                        tiles=np.stack([tiles[k] for k in keys]), example=np.array([example[k] for k in keys]))
    mixed = ((votes > 0).all(1)).sum()
    print(f"{frames} frames, {len(keys)} unique cells; {(votes[:, 1] > 0).sum()} ever walkable, "
          f"{mixed} with mixed walkable votes; singletons {(np.array([count[k] for k in keys]) == 1).sum()}")


if __name__ == "__main__":
    main()
