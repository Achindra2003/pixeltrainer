"""Measure the screen geometry facts PLAN.md §3.2 relies on (task 0.8).

Usage: python tools/probe_geometry.py
From states/pallet_start.state (headless) it:
  1. bumps into the fence below the lab door (blocked step),
  2. walks right 2 / left 2 (normal steps),
  3. enters and leaves Oak's lab (door fade),
  4. walks north to Route 1 (outdoor map border),
and records frames per step, measured screen shift, shades, and brightness during transitions.
This is a tool, so it reads RAM freely. Writes runs/probe/probe.json and a few PNGs.
"""
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from pyboy import PyBoy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import truth  # noqa: E402

OUT = Path("runs/probe")
PLAYER_CELL = (4, 4)  # (col, row) of the player's 16x16 cell, checked below


def grey(p):
    return p.screen.ndarray[:, :, 0].copy()


def pos(p):
    t = truth.read(p)
    return t["map_id"], t["x"], t["y"]


def best_shift(a, b):
    """Exact (dx, dy) in [-16, 16]^2 maximising matching pixels, player cell masked out."""
    mask = np.ones_like(a, bool)
    c, r = PLAYER_CELL
    mask[r * 16 - 4:(r + 1) * 16, c * 16:(c + 1) * 16] = False
    best = (0, 0, -1.0)
    for dy in range(-16, 17):
        for dx in range(-16, 17):
            ys, yd = (slice(dy, None), slice(None, -dy or None)) if dy >= 0 else (slice(None, dy), slice(-dy, None))
            xs, xd = (slice(dx, None), slice(None, -dx or None)) if dx >= 0 else (slice(None, dx), slice(-dx, None))
            m = mask[ys, xs] & mask[yd, xd]
            score = ((a[ys, xs] == b[yd, xd]) & m).sum() / m.sum()
            if score > best[2]:
                best = (dx, dy, score)
    return best  # content moved by (-dx, -dy) on screen; see summary


def phase_shift(a, b):
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return round(dx, 2), round(dy, 2), round(resp, 3)


def settle(p, cap=400):
    """Tick until no step is in progress and the screen is still (<1% pixels change) for 8 frames.
    Animated tiles (flowers, water) change a few pixels, hence the 1% tolerance.
    Returns frames ticked and per-frame means."""
    means, quiet, prev = [], 0, grey(p)
    for n in range(1, cap + 1):
        p.tick()
        g = grey(p)
        means.append(float(g.mean()))
        still = ((g != prev).mean() < 0.01 and g.std() > 10  # a blank (faded) screen is not settled
                 and p.memory[p.symbol_lookup("wWalkCounter")[1]] == 0)
        quiet = quiet + 1 if still else 0
        prev = g
        if quiet >= 8:
            return n - 8, means  # frames until the screen went still
    return cap, means


def step(p, d):
    before, pos0 = grey(p), pos(p)
    p.button_press(d)
    means, held = [], 0
    for held in range(1, 41):
        p.tick()
        means.append(float(grey(p).mean()))
        if pos(p) != pos0:
            break
    p.button_release(d)
    n, m2 = settle(p)
    after, pos1 = grey(p), pos(p)
    dx, dy, score = best_shift(before, after)
    return {
        "dir": d, "from": pos0, "to": pos1, "moved": pos1 != pos0,
        "frames_held": held, "frames_total": held + n,
        "exact_shift": [dx, dy], "match": round(score, 4),
        "phase_shift": phase_shift(before, after),
        "min_mean": round(min(means + m2), 1), "max_mean": round(max(means + m2), 1),
        "frame_mean": round(float(after.mean()), 1),
    }


def cells_walkable(p):
    c = np.asarray(p.game_wrapper.game_area_collision())  # 18x20 tiles, 1 = walkable
    return c.reshape(9, 2, 10, 2).min(axis=(1, 3)).astype(bool)


def next_dir_north(p, target_x):
    """BFS on the visible walkable cells to the most northern reachable cell (tie: nearest the exit column)."""
    walk = cells_walkable(p)
    _, x, _ = pos(p)
    tcol = PLAYER_CELL[0] + (target_x - x)
    start = (PLAYER_CELL[1], PLAYER_CELL[0])
    prev = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < 9 and 0 <= n[1] < 10 and n not in prev and walk[n]:
                prev[n] = (r, c)
                q.append(n)
    goal = min(prev, key=lambda rc: (rc[0], abs(rc[1] - tcol)))
    if goal == start:
        return "up"
    while prev[goal] != start:
        goal = prev[goal]
    return {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}[(goal[0] - start[0], goal[1] - start[1])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = PyBoy("roms/pokered.gb", window="null", symbols="roms/pokered.sym")
    p.set_emulation_speed(0)
    with open("states/pallet_start.state", "rb") as f:
        p.load_state(f)
    settle(p)
    cv2.imwrite(str(OUT / "start.png"), grey(p))

    t = truth.read(p)
    player = t["sprites"][0]
    res = {
        "shades": sorted(int(v) for v in np.unique(grey(p))),
        "scroll_registers_standing": {"SCX": p.memory[0xFF43], "SCY": p.memory[0xFF42]},
        "player_sprite_screen_xy": [player["screen_x"], player["screen_y"]],
        "steps": [],
    }

    # grid alignment: fewest unique 16x16 cells at the true offset
    g = grey(p)
    res["unique_cells_by_offset"] = {
        f"{ox},{oy}": len({g[y:y + 16, x:x + 16].tobytes() for y in range(oy, 144 - 15, 16) for x in range(ox, 160 - 15, 16)})
        for ox in (0, 4, 8, 12) for oy in (0, 4, 8, 12)
    }

    plan = [("blocked", "down"), ("walk", "right"), ("walk", "right"), ("walk", "left"), ("walk", "left"),
            ("door_in", "up")]
    for label, d in plan:
        s = step(p, d)
        s["label"] = label
        res["steps"].append(s)
    cv2.imwrite(str(OUT / "inside_lab.png"), grey(p))
    s = step(p, "down")
    s["label"] = "door_out"
    res["steps"].append(s)
    if s["to"][0] != 0:  # one step may only reach the mat; try once more
        s = step(p, "down")
        s["label"] = "door_out2"
        res["steps"].append(s)

    # walk north to Route 1 (map 12); Pallet's exit is at x = 10..11
    for _ in range(60):
        if pos(p)[0] == 12:
            break
        s = step(p, next_dir_north(p, 10))
        s["label"] = "north"
        res["steps"].append(s)
    cv2.imwrite(str(OUT / "route1.png"), grey(p))
    p.stop(save=False)

    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    print("shades", res["shades"], "| scroll", res["scroll_registers_standing"], "| player sprite xy", res["player_sprite_screen_xy"])
    print("unique cells by offset (x,y):", res["unique_cells_by_offset"])
    for s in res["steps"]:
        print(f"{s['label']:9} {s['dir']:5} {tuple(s['from'])}->{tuple(s['to'])} held={s['frames_held']:2} total={s['frames_total']:3} "
              f"exact={s['exact_shift']} match={s['match']} phase={s['phase_shift']} mean[min,max]=[{s['min_mean']},{s['max_mean']}]")


if __name__ == "__main__":
    main()
