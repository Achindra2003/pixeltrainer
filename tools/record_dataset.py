"""Record labelled frames by scripted play (task 1.3). A tool: it reads RAM to steer and to label.

Usage: python tools/record_dataset.py --runs 0-19 [--decisions 1200]
Split by run ID: 0-9 train, 10-14 validation, 15-19 test.
Even runs start at states/pallet_start.state and head north to Viridian; odd runs start at
states/viridian_north.state, wander there, then head south to Pallet. In each town they wander
for a while (~50 decisions), then head back, so Route 1 is crossed several times.

Output per run, data/raw/<run>/:
  frames/<i>.png   grey image, shade s stored as 255 - 85*s (s = (255 - g) // 85)
  labels.jsonl     one line per frame: truth.read() + stable flag, action, decision index;
                   stable frames also get the 18x20 tile collision grid and the wTileMap tile IDs
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import truth  # noqa: E402
from pixeltrainer.emu import DIRECTIONS, Emu  # noqa: E402

OUT = Path("data/raw")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
TRANSITION_EVERY = 20  # also save every 20th frame while waiting (scrolls, fades, text printing)


def walkable_cells(pyboy):
    c = np.asarray(pyboy.game_wrapper.game_area_collision())  # 18x20 tiles, 1 = walkable
    return c.reshape(9, 2, 10, 2).min(axis=(1, 3)).astype(bool)


def goal_direction(pyboy, goal):
    """First move of a BFS path (visible cells) to the reachable cell furthest in the goal direction."""
    walk = walkable_cells(pyboy)
    start = (4, 4)
    prev, q = {start: None}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DELTA.values():
            n = (r + dr, c + dc)
            if 0 <= n[0] < 9 and 0 <= n[1] < 10 and n not in prev and walk[n]:
                prev[n] = (r, c)
                q.append(n)
    sign_r, sign_c = DELTA[goal]
    target = max(prev, key=lambda rc: rc[0] * sign_r + rc[1] * sign_c)
    if target == start:
        return goal  # nothing better visible: push on (may be a ledge or the map edge)
    while prev[target] != start:
        target = prev[target]
    return next(d for d, v in DELTA.items() if v == (target[0] - 4, target[1] - 4))


class Recorder:
    def __init__(self, run, decisions):
        self.run, self.decisions = run, decisions
        self.dir = OUT / f"{run:02d}"
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.labels = open(self.dir / "labels.jsonl", "w")
        self.rng = np.random.default_rng(1000 + run)
        state = "states/pallet_start.state" if run % 2 == 0 else "states/viridian_north.state"
        self.goal_map, self.goal = 1, "up"  # odd runs start in Viridian: wander there first, then head south
        self.emu = Emu(state=state, seed=run)
        self.n, self.tick_count, self.action, self.k = 0, 0, "reset", 0
        self.battle_len = self.text_len = 0

    def save(self, stable):
        p = self.emu._pyboy
        t = truth.read(p)
        t.update(i=self.n, stable=stable, action=self.action, decision=self.k, frame=self.emu.frames)
        t["tilemap"] = truth.tilemap(p)
        if stable and not t["in_battle"]:
            t["collision"] = np.asarray(p.game_wrapper.game_area_collision()).tolist()
        cv2.imwrite(str(self.dir / "frames" / f"{self.n:06d}.png"), 255 - 85 * self.emu.frame())
        self.labels.write(json.dumps(t) + "\n")
        self.n += 1

    def on_frame(self):
        self.tick_count += 1
        if self.tick_count % TRANSITION_EVERY == 0:
            self.save(stable=False)

    def choose(self):
        p = self.emu._pyboy
        t = truth.read(p)
        r = self.rng.random()
        if t["in_battle"]:
            self.battle_len += 1
            if self.battle_len > 40 and r < 0.5:  # A-spam can loop (e.g. PKMN menu): back out, cursor to FIGHT
                return self.rng.choice(["b", "up", "left"])
            if r < 0.8:
                return "a"
            return self.rng.choice(["b", "up", "left"]) if r < 0.95 else self.rng.choice(["down", "right"])
        self.battle_len = 0
        if t["font_loaded"]:  # dialogue, but also start/party/status menus
            self.text_len += 1
            if self.text_len > 30 and r < 0.6:  # A can loop inside menus: back out
                return "b"
            return "a" if r < 0.9 else "b"
        self.text_len = 0
        if r < 0.02:
            return "start"
        if r < 0.07:  # talk to whatever is in front (NPCs, signs): the only source of dialogue frames
            return "a"
        if t["map_id"] == self.goal_map and self.rng.random() < 0.02:  # wander a while, then turn back
            self.goal_map, self.goal = (0, "down") if self.goal_map == 1 else (1, "up")
        if t["map_id"] == self.goal_map or r > 0.7:
            return self.rng.choice(DIRECTIONS)
        return goal_direction(p, self.goal)

    def act(self, action):
        self.action = action
        self.emu.step(action, on_frame=self.on_frame)
        self.save(stable=True)
        if action == "start":  # browse the start menu a little, then close it
            for a in list(self.rng.choice(["down", "up"], self.rng.integers(0, 4))) + ["b"]:
                self.action = a
                self.emu.press(a, on_frame=self.on_frame)
                self.emu.wait_until_stable(on_frame=self.on_frame)
                self.save(stable=True)

    def record(self):
        self.save(stable=True)
        for self.k in range(self.decisions):
            self.act(self.choose())
        self.labels.close()
        self.emu.close()
        return self.n


def parse_runs(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-19")
    ap.add_argument("--decisions", type=int, default=1200)
    args = ap.parse_args()
    for run in parse_runs(args.runs):
        n = Recorder(run, args.decisions).record()
        print(f"run {run:02d}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
