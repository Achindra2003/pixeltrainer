"""Train the tile CNN (task 2.8) on the labelled cells; pick the epoch by walkable accuracy on validation runs 10-14.

Usage: python tools/train_tiles.py [--epochs 30]
Needs data/cells.npz (tools/cell_inventory.py, runs 0-9) and assets/cell_labels.json (task 2.6).
Each labelled cell is cut with a 1 px border from its example frame, so the shift augmentation (±1 px) shows
real neighbouring pixels. Noise augmentation: a random share (0-8%) of pixels set to a random shade, then
Gaussian noise (sigma 0-0.15) on the [0, 1] input. Every cell is drawn equally often (64 times per epoch),
however rare. Saves models/tiles.pt. Test with: python evaluation/eval_tiles.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_tiles import load_runs  # noqa: E402
from pixeltrainer.perception.tiles import CLASSES, MODEL, PASSABLE, make_net, to_input  # noqa: E402


def padded_cells():
    """(N, 18, 18) labelled cells with their 1 px neighbourhood (edge pixels repeated at the screen border)."""
    z = np.load("data/cells.npz")
    labels = json.loads(Path("assets/cell_labels.json").read_text())
    out = []
    for run, i, r, c in z["example"]:
        f = (255 - cv2.imread(f"data/raw/{run:02d}/frames/{i:06d}.png", cv2.IMREAD_GRAYSCALE)) // 85
        f = np.pad(f, 1, mode="edge")
        out.append(f[16 * r:16 * r + 18, 16 * c:16 * c + 18])
    y = np.array([CLASSES.index(labels[str(k)]) for k in z["keys"]])
    assert all((p[1:17, 1:17] == cell).all() for p, cell in zip(out, z["cells"]))
    return np.stack(out), y


def augment(p, rng):
    """(B, 18, 18) padded cells -> (B, 1, 16, 16) float input, shifted ±1 px, with shade and Gaussian noise."""
    dy, dx = rng.integers(0, 3, (2, len(p)))
    x = np.stack([q[a:a + 16, b:b + 16] for q, a, b in zip(p, dy, dx)])
    flip = rng.random(x.shape) < rng.uniform(0, 0.08, (len(x), 1, 1))
    x = np.where(flip, rng.integers(0, 4, x.shape), x)
    t = to_input(x)
    return t + torch.randn_like(t) * torch.as_tensor(rng.uniform(0, 0.15, (len(x), 1, 1, 1)), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    p, y = padded_cells()
    xva, wva = load_runs(range(10, 15), every=3)
    print(f"train {len(p)} cells {np.bincount(y, minlength=len(CLASSES))}, val {len(xva)} cells", flush=True)
    passable = torch.tensor([c in PASSABLE for c in CLASSES])
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = -1
    MODEL.parent.mkdir(exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(np.repeat(np.arange(len(p)), 64))
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            loss = loss_fn(net(augment(p[b], rng)), torch.as_tensor(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.inference_mode():
            pred = torch.cat([net(to_input(xva[j:j + 4096])).argmax(1) for j in range(0, len(xva), 4096)])
        acc = (passable[pred].numpy() == wva).mean()
        print(f"epoch {ep}: loss {loss.item():.4f} val walkable acc {acc:.4f}", flush=True)
        if acc >= best:  # ties go to the later, more trained epoch
            best = acc
            torch.save(net.state_dict(), MODEL)
    print(f"best val walkable acc {best:.4f}, saved {MODEL}")


if __name__ == "__main__":
    main()
