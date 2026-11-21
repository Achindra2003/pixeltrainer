"""NPC detection precision / recall / facing accuracy against the RAM sprite table (task 6.2).

Usage: python evaluation/eval_npc.py [--runs 15-19] [--sweep]
Frames: stable overworld frames. Truth: NPC sprites (slot > 0, image_index != 0xFF) whose 16x16 box at
(screen_y, screen_x) lies fully on screen. A detection matches an unmatched truth box with IoU >= 0.5.
Templates come from runs 0-14, so the default runs are unseen. --sweep prints P/R over thresholds.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_registration import frame  # noqa: E402
from pixeltrainer.perception import npc  # noqa: E402


def truth_boxes(t):
    return [((s["screen_y"], s["screen_x"]), s["facing"]) for s in t["sprites"][1:]
            if s["image_index"] != 0xFF and 0 <= s["screen_y"] <= 128 and 0 <= s["screen_x"] <= 144]


def score(items, threshold):
    c = collections.Counter()
    misses = []
    for d, t, f in items:
        truth = truth_boxes(t)
        dets = npc.detect(f, threshold)
        used = set()
        for det in dets:
            j = next((j for j, (b, _) in enumerate(truth)
                      if j not in used and npc._iou(det["box"][:2], b) >= npc.IOU), None)
            if j is None:
                c["fp"] += 1
                if len(misses) < 10:
                    misses.append(("fp", d.name, t["i"], det["box"][:2], det["key"], round(det["score"], 2)))
            else:
                used.add(j)
                c["tp"] += 1
                c["facing ok"] += det["facing"] == truth[j][1]
        for j, (b, fc) in enumerate(truth):
            if j not in used:
                c["fn"] += 1
                if len(misses) < 20:
                    misses.append(("fn", d.name, t["i"], b, fc))
    return c, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="15-19")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    items = []
    for run in range(int(a), int(b or a) + 1):
        d = Path(f"data/raw/{run:02d}")
        screen = json.loads((d / "screen.json").read_text())
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if t["stable"] and screen[t["i"]] == "overworld":
                items.append((d, t, frame(d, t)))
    for th in (np.arange(0.5, 0.96, 0.05) if args.sweep else [npc.THRESHOLD]):
        c, misses = score(items, th)
        p = c["tp"] / max(c["tp"] + c["fp"], 1)
        r = c["tp"] / max(c["tp"] + c["fn"], 1)
        print(f"threshold {th:.2f}: precision {p:.1%}, recall {r:.1%} (tp {c['tp']}, fp {c['fp']}, fn {c['fn']}), "
              f"facing {c['facing ok'] / max(c['tp'], 1):.1%}; {len(items)} frames", flush=True)
    print("first errors:", *misses, sep="\n  ")


if __name__ == "__main__":
    main()
