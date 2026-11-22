"""Tile classifiers, lookup vs CNN (PLAN.md §4.3, task 2.8).

Usage: python evaluation/eval_tiles.py            (needs models/tiles.pt, tools/train_tiles.py)
       python evaluation/eval_tiles.py --collect  first record the unseen-area frames (data/unseen_tiles.npz)
Seen area: test runs 15-19 (Pallet, Route 1, Viridian), stable overworld frames, every cell not under a sprite.
Unseen area: from states/viridian_north.state a RAM-steered walk north (tools/record_dataset.goal_direction)
through Route 2 and its gate into Viridian Forest; overworld frames on maps the training runs never visited.
Walkable is scored against RAM collision (all 4 of the cell's 8x8 tiles walkable); a class against the
checked labels (assets/cell_labels.json), so only on cells whose exact content is labelled.
Writes runs/tiles_eval.txt.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from evaluation import truth  # noqa: E402
from pixeltrainer.perception import grid, tiles  # noqa: E402

UNSEEN = Path("data/unseen_tiles.npz")
TRAIN_MAPS = {0, 1, 12, 37, 38, 40}  # Pallet, Viridian, Route 1, the houses and Oak's lab (runs 0-14)
MAP_NAMES = {13: "Route 2", 50: "Forest gate", 51: "Viridian Forest"}


def frame_cells(frame, collision, sprites):
    """(cells (K,16,16), walkable (K,)) for the cells of one frame not under a sprite."""
    walk = np.asarray(collision).reshape(9, 2, 10, 2).min(axis=(1, 3)).astype(bool)
    cs = grid.cells(frame)
    keep = [(r, c) for r in range(grid.ROWS) for c in range(grid.COLS) if (r, c) not in sprites]
    return np.stack([cs[r, c] for r, c in keep]), np.array([walk[r, c] for r, c in keep])


def load_runs(runs, every=1):
    """Cells of stable overworld frames in data/raw/<run> (screen.json label "overworld"), every n-th frame."""
    xs, ws = [], []
    for run in runs:
        d = Path(f"data/raw/{run:02d}")
        screen = json.loads((d / "screen.json").read_text())
        n = 0
        for line in open(d / "labels.jsonl"):
            t = json.loads(line)
            if not t["stable"] or screen[t["i"]] != "overworld" or "collision" not in t:
                continue
            n += 1
            if n % every:
                continue
            f = (255 - cv2.imread(str(d / "frames" / f"{t['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85
            x, w = frame_cells(f, t["collision"], truth.sprite_cells(t))
            xs.append(x)
            ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def collect(seeds=range(3), decisions=500):
    """Walk north from Viridian; keep overworld frames on maps outside TRAIN_MAPS."""
    from record_dataset import goal_direction
    from pixeltrainer.emu import DIRECTIONS, Emu
    xs, ws, ms = [], [], []
    for seed in seeds:
        emu = Emu(state="states/viridian_north.state", seed=seed)
        emu.reset()
        rng = np.random.default_rng(seed)
        frame = emu.frame()
        for _ in range(decisions):
            t = truth.read(emu._pyboy)
            menu = t["in_battle"] or t["font_loaded"]
            if not menu and t["map_id"] not in TRAIN_MAPS and frame.std() > 0.3:  # std: not a fade
                x, w = frame_cells(frame, emu._pyboy.game_wrapper.game_area_collision(), truth.sprite_cells(t))
                xs.append(x)
                ws.append(w)
                ms.append(np.full(len(w), t["map_id"]))
            if menu:  # A through battles and text
                action = "a" if rng.random() < 0.8 else "b"
            else:
                action = goal_direction(emu._pyboy, "up") if rng.random() < 0.75 else str(rng.choice(DIRECTIONS))
            frame = emu.step(action)
        emu.close()
        print(f"seed {seed}: {sum(map(len, ws))} cells so far", flush=True)
    np.savez_compressed(UNSEEN, cells=np.concatenate(xs), walk=np.concatenate(ws), maps=np.concatenate(ms))


def score(name, x, w, out):
    keys = [grid.cell_key(c) for c in x]
    table = tiles.labels()
    found = np.array([k in table for k in keys])
    look_walk = np.array([table.get(k) in tiles.PASSABLE for k in keys])
    pred = predict(x)
    cnn_walk = np.isin(pred, list(tiles.PASSABLE))
    lab = np.array([table.get(k) for k in keys], dtype=object)
    uniq = len(set(keys))
    new = len({k for k, f in zip(keys, found) if not f})
    out += [f"## {name}: {len(x)} cells, {uniq} distinct, {new} distinct not in the labels "
            f"({(~found).mean():.1%} of cells)",
            f"lookup: found {found.mean():.2%}; walkable vs RAM on found {(look_walk == w)[found].mean():.2%}; "
            f"overall (unknown = wrong) {((look_walk == w) & found).mean():.2%}",
            f"CNN:    walkable vs RAM {(cnn_walk == w).mean():.2%}"
            + (f"; on cells not in the labels {(cnn_walk == w)[~found].mean():.2%}" if (~found).any() else "")
            + (f"; class vs labels {(pred == lab)[found].mean():.2%}" if found.any() else ""),
            f"CNN walkable errors: {int(((cnn_walk) & ~w).sum())} blocked read as walkable, "
            f"{int((~cnn_walk & w).sum())} walkable read as blocked",
            f"CNN classes: {dict(Counter(pred).most_common())}", ""]
    return cnn_walk == w


def predict(x):
    import torch
    net = tiles.make_net()
    net.load_state_dict(torch.load(tiles.MODEL, map_location="cpu"))
    net.eval()
    with torch.inference_mode():
        i = torch.cat([net(tiles.to_input(x[j:j + 4096])).argmax(1) for j in range(0, len(x), 4096)])
    return np.array(tiles.CLASSES, dtype=object)[i.numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect()
        return
    out = ["# Tile classifiers: exact lookup vs CNN (task 2.8)", ""]
    x, w = load_runs(range(15, 20))
    score("Seen area: test runs 15-19", x, w, out)
    u = np.load(UNSEEN)
    ok = score("Unseen area: north of Viridian (" + ", ".join(MAP_NAMES.get(m, f"map {m}") for m in sorted(set(u["maps"]))) + ")",
               u["cells"], u["walk"], out)
    out += ["CNN walkable vs RAM per unseen map: " + ", ".join(
        f"{MAP_NAMES.get(m, f'map {m}')} {ok[u['maps'] == m].mean():.2%} ({(u['maps'] == m).sum()} cells)" for m in sorted(set(u["maps"])))]
    text = "\n".join(out)
    print(text)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/tiles_eval.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
