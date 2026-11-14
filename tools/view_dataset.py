"""Step through recorded frames with their RAM labels (task 1.4).

Usage:
  python tools/view_dataset.py 3            view run 03
  python tools/view_dataset.py --summary    per-run counts (maps, battles, dialogue, stable)
Keys: d / right arrow = next, a / left arrow = previous, s = next stable frame, q / Esc = quit.
"""
import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path("data/raw")
MAP_NAMES = {0: "Pallet", 1: "Viridian", 12: "Route 1", 40: "Oak lab"}
RIGHT, LEFT = 2555904, 2424832  # cv2.waitKeyEx arrow codes on Windows


def load(run):
    return [json.loads(line) for line in open(RAW / f"{run:02d}" / "labels.jsonl")]


def summary():
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        L = load(int(d.name))
        S = [t for t in L if t["stable"]]
        maps = collections.Counter(MAP_NAMES.get(t["map_id"], f"map{t['map_id']}") for t in S)
        print(f"run {d.name}: {len(L):5} frames, {len(S):4} stable | battle {sum(1 for t in S if t['in_battle']):4} "
              f"| text {sum(1 for t in S if t['font_loaded']):3} | {dict(maps.most_common())}")


def panel(t):
    lines = [
        f"#{t['i']} {'STABLE' if t['stable'] else 'transition'}  action={t['action']}",
        f"map {t['map_id']} {MAP_NAMES.get(t['map_id'], '')}  xy=({t['x']},{t['y']}) {t['facing']}",
        f"battle={t['in_battle']} font={t['font_loaded']} menu={t['menu_item']}/{t['max_menu_item']}",
        f"party1 sp{t['party1']['species']} L{t['party1']['level']} {t['party1']['hp']}/{t['party1']['max_hp']}",
    ]
    if "enemy" in t:
        e = t["enemy"]
        lines.append(f"enemy sp{e['species']} L{e['level']} {e['hp']}/{e['max_hp']}")
    img = np.full((432, 420, 3), 255, np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(img, s, (8, 24 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def view(run):
    L = load(run)
    i = 0
    while True:
        t = L[i]
        frame = cv2.imread(str(RAW / f"{run:02d}" / "frames" / f"{t['i']:06d}.png"))
        big = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"run {run:02d}", np.hstack([big, panel(t)]))
        k = cv2.waitKeyEx(0)
        if k in (ord("q"), 27):
            break
        if k in (ord("d"), RIGHT):
            i = min(i + 1, len(L) - 1)
        elif k in (ord("a"), LEFT):
            i = max(i - 1, 0)
        elif k == ord("s"):
            i = next((j for j in range(i + 1, len(L)) if L[j]["stable"]), i)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary or a.run is None else view(a.run)
