"""Train the screen-state CNN (task 2.3) on runs 0-9, pick the epoch by validation runs 10-14.

Usage: python tools/train_screen_state.py [--epochs 8]
Needs data/raw/<run>/screen.json (python evaluation/screen_labels.py). Saves models/screen_state.pt.
Test with: python evaluation/eval_screen_state.py --method cnn
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
from pixeltrainer.perception.screen_state import CLASSES, MODEL, make_net, to_input  # noqa: E402


def load(runs):
    xs, ys = [], []
    for run in runs:
        d = Path(f"data/raw/{run:02d}")
        for i, y in enumerate(json.loads((d / "screen.json").read_text())):
            xs.append((255 - cv2.imread(str(d / "frames" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85)
            ys.append(CLASSES.index(y))
    return np.stack(xs).astype(np.uint8), np.array(ys)


def accuracy(net, x, y):
    net.eval()
    with torch.inference_mode():
        p = torch.cat([net(to_input(x[i:i + 512])).argmax(1) for i in range(0, len(x), 512)])
    return (p.numpy() == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    xtr, ytr = load(range(10))
    xva, yva = load(range(10, 15))
    print("train", np.bincount(ytr, minlength=len(CLASSES)), "val", np.bincount(yva, minlength=len(CLASSES)), flush=True)
    net = make_net()
    freq = np.bincount(ytr, minlength=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor((freq.max() / freq) ** 0.5, dtype=torch.float32))  # rare dialogue
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = 0
    MODEL.parent.mkdir(exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        order = np.random.default_rng(ep).permutation(len(xtr))
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            loss = loss_fn(net(to_input(xtr[b])), torch.as_tensor(ytr[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = accuracy(net, xva, yva)
        print(f"epoch {ep}: loss {loss.item():.4f} val acc {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save(net.state_dict(), MODEL)
    print(f"best val acc {best:.4f}, saved {MODEL}")


if __name__ == "__main__":
    main()
