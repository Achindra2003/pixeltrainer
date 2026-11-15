"""Replay recorded runs through registration + world map, compare with RAM (tasks 3.2b and 3.3).

Usage: python evaluation/eval_worldmap.py [--runs 0-19] [--cnn] [--images 0]
Screen state per frame comes from the RAM labels (data/raw/<run>/screen.json), or from the screen-state
CNN with --cnn. Truth position: RAM wXCoord/wYCoord. Pallet, Route 1 and Viridian are one outdoor frame
for the agent but three maps in RAM; their offsets come from RAM at border crossings (the step direction
says where the player went). Each map segment is aligned to the truth by its first frame; the error is
the tile distance after that. A segment that later shows a different RAM map is a mix-up.
--images N saves runs/worldmap/<N>/mosaic_<seg>.png and render_<seg>.png for run N.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_registration import DIRECTIONS, OUTDOOR, frame  # noqa: E402
from pixeltrainer.mapping import registration, worldmap  # noqa: E402


def stable(run):
    d = Path(f"data/raw/{run:02d}")
    screen = json.loads((d / "screen.json").read_text())
    for line in open(d / "labels.jsonl"):
        t = json.loads(line)
        if t["stable"]:
            yield d, t, screen[t["i"]]


def outdoor_offsets(runs):
    """map_id -> (ox, oy) so that RAM (x, y) + offset is one outdoor grid; Pallet is (0, 0)."""
    edges = collections.defaultdict(dict)
    for run in runs:
        prev = None
        for _, t, _ in stable(run):
            if prev and t["action"] in DIRECTIONS and prev["map_id"] != t["map_id"] \
                    and {prev["map_id"], t["map_id"]} <= OUTDOOR:
                ux, uy = registration.DELTA[t["action"]]
                edges[prev["map_id"]][t["map_id"]] = (prev["x"] + ux - t["x"], prev["y"] + uy - t["y"])
            prev = t
    off, todo = {0: (0, 0)}, [0]
    while todo:
        a = todo.pop()
        for b, (dx, dy) in edges[a].items():
            if b not in off:
                off[b] = (off[a][0] + dx, off[a][1] + dy)
                todo.append(b)
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-19")
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--images", type=int)
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    runs = range(int(a), int(b or a) + 1)
    off = outdoor_offsets(range(20))
    print("outdoor offsets (map_id: dx, dy):", off)
    if args.cnn:
        from pixeltrainer.perception import screen_state
    total = collections.Counter()
    for run in runs:
        wm, align, err = worldmap.WorldMap(), {}, collections.Counter()
        for d, t, state in stable(run):
            f = frame(d, t)
            if args.cnn:
                state = screen_state.cnn(f)
            est = wm.update(f, state, t["action"] if t["action"] in DIRECTIONS else None, t=t["i"])
            if est is None:
                continue
            seg, x, y = est
            if t["map_id"] in OUTDOOR and t["map_id"] in off:
                group, tx, ty = "outdoor", t["x"] + off[t["map_id"]][0], t["y"] + off[t["map_id"]][1]
            else:
                group, tx, ty = t["map_id"], t["x"], t["y"]
            if seg not in align:
                align[seg] = (group, tx - x, ty - y)
            g, ax, ay = align[seg]
            if g != group:
                err["mixup"] += 1
                continue
            e = abs(x + ax - tx) + abs(y + ay - ty)
            err["frames"] += 1
            err["exact"] += e == 0
            err["max"] = max(err["max"], e)
        groups = collections.Counter(g for g, _, _ in align.values())
        print(f"run {run:02d}: {err['exact']}/{err['frames']} frames at 0 tiles error, max {err['max']}, "
              f"mix-ups {err['mixup']}; {wm.n_segments} segments for {len(groups)} RAM areas "
              f"{dict(groups)}, {len(wm.doors)} door links")
        for k in ("frames", "exact", "mixup"):
            total[k] += err[k]
        total["max"] = max(total["max"], err["max"])
        if run == args.images:
            out = Path(f"runs/worldmap/{run:02d}")
            out.mkdir(parents=True, exist_ok=True)
            for s in range(wm.n_segments):
                cv2.imwrite(str(out / f"mosaic_{s}.png"), wm.mosaic(s))
                cv2.imwrite(str(out / f"render_{s}.png"), wm.render(s))
            print("images in", out)
    print(f"all: {total['exact']}/{total['frames']} frames at 0 tiles error, max {total['max']}, "
          f"mix-ups {total['mixup']}")


if __name__ == "__main__":
    main()
