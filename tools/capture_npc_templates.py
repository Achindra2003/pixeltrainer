"""NPC sprite templates labelled from the RAM sprite table (task 6.1).

Usage: python tools/capture_npc_templates.py [--runs 0-14]
Training runs only (15-19 are the test runs for perception/npc.py). For every stable overworld frame,
every on-screen NPC (slot > 0, image_index != 0xFF) standing on a cell (screen_y + 4 and screen_x multiples
of 16) and fully inside the screen gives a 16x16 crop at (screen_y, screen_x), keyed by (picture_id,
facing). A sprite's transparent pixels show whatever is behind it, and tall grass draws over its lower half.
One NPC standing still can give hundreds of crops of the same background, so crops are first reduced to one
per-pixel mode per location (map, map_x, map_y); the template is the mode across locations and the mask
keeps the pixels where >= AGREE of the locations agree. With one location only, crops vote instead (so
background behind a sprite that never moved stays in its template). The per-location mode also outvotes
the frames where RAM facing runs one frame ahead of the drawn sprite.
Writes assets/npc_templates.npz (keys "pic_facing", templates (K,16,16) shades, masks (K,16,16) bool, counts)
and runs/npc_templates.png (each template 8x, masked pixels red).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_registration import frame  # noqa: E402

AGREE = 0.9
MIN_CROPS = 5
MIN_PER_LOCATION = 3  # fewer crops at a place are mostly frames where RAM facing runs ahead


def crops(runs):
    """{(picture_id, facing): {(map_id, map_x, map_y): [16x16 shade crops]}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in runs:
        d = Path(f"data/raw/{run:02d}")
        screen = json.loads((d / "screen.json").read_text())
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not t["stable"] or screen[t["i"]] != "overworld":
                continue
            f = None
            for s in t["sprites"][1:]:
                y, x = s["screen_y"], s["screen_x"]
                if s["image_index"] == 0xFF or (y + 4) % 16 or x % 16 or not (0 <= y <= 128 and 0 <= x <= 144):
                    continue
                f = frame(d, t) if f is None else f
                out[s["picture_id"], s["facing"]][t["map_id"], s["map_x"], s["map_y"]].append(f[y:y + 16, x:x + 16])
    return out


def _votes(cs):
    cs = np.array(cs)
    return np.stack([(cs == s).mean(0) for s in range(4)])  # (4, 16, 16) share of crops per shade


def build(by_key):
    keys, temps, masks, counts = [], [], [], []
    for k, locs in sorted(by_key.items()):
        n = sum(len(cs) for cs in locs.values() if len(cs) >= MIN_PER_LOCATION)
        if n < MIN_CROPS:
            continue
        locs = [cs for cs in locs.values() if len(cs) >= MIN_PER_LOCATION]
        votes = _votes([_votes(cs).argmax(0) for cs in locs]) if len(locs) > 1 else _votes(locs[0])
        keys.append(f"{k[0]}_{k[1]}")
        temps.append(votes.argmax(0).astype(np.uint8))
        masks.append(votes.max(0) >= AGREE)
        counts.append(n)
    return keys, np.array(temps), np.array(masks), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-14")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    keys, temps, masks, counts = build(crops(range(int(a), int(b or a) + 1)))
    np.savez("assets/npc_templates.npz", keys=np.array(keys), templates=temps, masks=masks, counts=counts)
    tiles = []
    for t, m in zip(temps, masks):
        g = cv2.cvtColor((255 - 85 * t).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        g[~m] = (0, 0, 255)
        tiles.append(cv2.copyMakeBorder(cv2.resize(g, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST),
                                        4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    cv2.imwrite("runs/npc_templates.png", np.hstack(tiles))
    for k, n, m in zip(keys, counts, masks):
        print(f"{k}: {n} crops, {m.mean():.0%} of pixels kept")


if __name__ == "__main__":
    main()
