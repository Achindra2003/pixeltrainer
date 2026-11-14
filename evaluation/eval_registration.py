"""Check registration.step against RAM coordinates on recorded walk steps (task 3.1).

Usage: python evaluation/eval_registration.py [--runs 0-19]
Pairs = consecutive stable frames, both "overworld" (data/raw/<run>/screen.json), same map, after a
walk command. Truth step = change in wXCoord/wYCoord. Methods: phase correlation (player masked),
phase correlation with NPCs also masked (RAM sprite table, standing in for perception/npc.py), and
matching with the pressed direction as a prior.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.truth import sprite_cells  # noqa: E402
from pixeltrainer.mapping import registration  # noqa: E402
from pixeltrainer.perception import grid  # noqa: E402

DIRECTIONS = ("up", "down", "left", "right")


OUTDOOR = {0, 1, 12}  # Pallet, Viridian, Route 1: joined by seamless borders


def pairs(run, same_map=True):
    d = Path(f"data/raw/{run:02d}")
    screen = json.loads((d / "screen.json").read_text())
    prev = None
    for line in open(d / "labels.jsonl"):
        t = json.loads(line)
        if not t["stable"]:
            continue
        both_overworld = prev is not None and screen[prev["i"]] == screen[t["i"]] == "overworld"
        if prev and t["action"] in DIRECTIONS and (prev["map_id"] == t["map_id"]) == same_map \
                and (both_overworld or not same_map):  # map changes include door fades
            yield prev, t, d, both_overworld
        prev = t


def frame(d, t):
    return (255 - cv2.imread(str(d / "frames" / f"{t['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-19")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    stats = collections.defaultdict(collections.Counter)
    wrong = []
    for run in range(int(a), int(b or a) + 1):
        for p, t, d, _ in pairs(run):
            truth = (t["x"] - p["x"], t["y"] - p["y"])
            f0, f1 = frame(d, p), frame(d, t)
            for m, got in (("phase", registration.step(f0, f1)),
                           ("phase+npc mask", registration.step(f0, f1, mask_cells=tuple(sprite_cells(p) | sprite_cells(t)))),
                           ("direction prior", registration.step(f0, f1, t["action"]))):
                kind = "jump" if max(map(abs, truth)) > 1 else "step" if any(truth) else "blocked"
                ok = got == truth
                stats[m][(kind, ok)] += 1
                if not ok and m == "direction prior" and len(wrong) < 30:
                    wrong.append((run, p["i"], t["i"], truth, got))
    for m, c in stats.items():
        n_ok = sum(v for (k, ok), v in c.items() if ok)
        print(f"{m}: {n_ok}/{sum(c.values())} correct; " +
              ", ".join(f"{k} {c[(k, True)]}/{c[(k, True)] + c[(k, False)]}" for k in ("step", "blocked", "jump")))
    print("first failures (run, prev, cur, truth, got):", wrong[:15])
    # Map changes. The agent registers only overworld -> overworld pairs; any other screen state
    # (a door fade is "transition") starts a new segment instead. A seamless outdoor border must
    # still read as one step; a door must never reach registration as a plain overworld pair.
    border, door = collections.Counter(), collections.Counter()
    for run in range(int(a), int(b or a) + 1):
        for p, t, d, both_overworld in pairs(run, same_map=False):
            if {p["map_id"], t["map_id"]} <= OUTDOOR:
                if not both_overworld:
                    border["gated"] += 1
                    continue
                ux, uy = registration.DELTA[t["action"]]
                border[registration.step(frame(d, p), frame(d, t), t["action"]) == (ux, uy)] += 1
            else:
                door["gated" if not both_overworld else registration.step(frame(d, p), frame(d, t), t["action"]) is None] += 1
    print(f"seamless borders, overworld pairs read as one step: {border[True]}/{border[True] + border[False]} "
          f"(+{border['gated']} with a non-overworld frame); doors caught: screen state {door['gated']}, "
          f"registration {door[True]}, missed {door[False]}")


if __name__ == "__main__":
    main()
