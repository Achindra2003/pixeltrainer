"""Battle perception against RAM (task 5.4).

Usage: python evaluation/eval_battle.py [--runs 15-19]
Frames: stable battle frames where perception/battle.py sees the main or fight menu (sprite and HUD in
place). Species top-1 among all 151 vs wEnemyMonSpecies; HP bar fraction vs HP / max HP (MAE, and how
often it equals the game's own pixel count floor(48 HP / max) / 48); own HP numbers; menu cursor vs
wCurrentMenuItem (main menu: its row only, the column is kept elsewhere; fight menu: 1-based).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_registration import frame  # noqa: E402
from pixeltrainer.perception import battle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="15-19")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    n, err = collections.Counter(), collections.defaultdict(list)
    confusions = collections.Counter()
    for run in range(int(a), int(b or a) + 1):
        d = Path(f"data/raw/{run:02d}")
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not (t["stable"] and t["in_battle"] and "enemy" in t):
                continue
            f = frame(d, t)
            m = battle.menu(f)
            n["menu " + m] += 1
            if m == "text":
                continue
            n["frames"] += 1
            sid, name, score = battle.species(f)
            n["species ok"] += sid == t["enemy"]["species"]
            if sid != t["enemy"]["species"]:
                confusions[(t["enemy"]["species"], name)] += 1
            err["ncc"].append(score)
            for side, key in (("enemy", "enemy"), ("own", "own")):
                hp, mx = t[key]["hp"], t[key]["max_hp"]
                got = battle.hp_fraction(f, side)
                err[side].append(abs(got - hp / mx))
                n[side + " pixel-exact"] += got == (48 * hp // mx) / 48
            n["own numbers ok"] += battle.own_hp(f) == (t["own"]["hp"], t["own"]["max_hp"])
            c = battle.cursor(f)  # main menu: wCurrentMenuItem holds the row only; fight menu: 1-based
            n["cursor ok"] += c is not None and (c // 2 if m == "main" else c + 1) == t["menu_item"]
    k = n["frames"]
    print(f"frames {k} (menus seen: main {n['menu main']}, fight {n['menu fight']}, text {n['menu text']})")
    print(f"species top-1 among 151: {n['species ok']}/{k} = {n['species ok'] / k:.2%}; "
          f"NCC min {min(err['ncc']):.3f}; confusions {confusions.most_common(5)}")
    for side in ("enemy", "own"):
        print(f"{side} HP bar: MAE {np.mean(err[side]):.2%}, max {np.max(err[side]):.2%}, "
              f"pixel-exact {n[side + ' pixel-exact']}/{k}")
    print(f"own HP numbers exact {n['own numbers ok']}/{k}; cursor = wCurrentMenuItem {n['cursor ok']}/{k}")


if __name__ == "__main__":
    main()
