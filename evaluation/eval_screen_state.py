"""Score a screen-state classifier against evaluation/screen_labels.py on the test runs (15-19).

Usage: python evaluation/eval_screen_state.py [--method heuristic] [--runs 15-19]
Needs data/raw/<run>/screen.json (python evaluation/screen_labels.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixeltrainer.perception import screen_state  # noqa: E402

CLASSES = screen_state.CLASSES


def load(run):
    d = Path(f"data/raw/{run:02d}")
    labels = json.loads((d / "screen.json").read_text())
    for i, y in enumerate(labels):
        yield (255 - cv2.imread(str(d / "frames" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85, y


def evaluate(predict, runs):
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)  # rows truth, cols prediction
    secs = 0.0
    for run in runs:
        for frame, y in load(run):
            t0 = time.perf_counter()
            p = predict(frame)
            secs += time.perf_counter() - t0
            cm[CLASSES.index(y), CLASSES.index(p)] += 1
    return cm, secs / cm.sum()


def report(cm, per_frame):
    lines = [f"accuracy {np.trace(cm) / cm.sum():.4f} on {cm.sum()} frames, {per_frame * 1000:.2f} ms/frame",
             "truth \ pred " + " ".join(f"{c[:6]:>7}" for c in CLASSES) + "   recall"]
    for i, c in enumerate(CLASSES):
        lines.append(f"{c:>12} " + " ".join(f"{n:7d}" for n in cm[i]) + f"   {cm[i, i] / max(cm[i].sum(), 1):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="heuristic")
    ap.add_argument("--runs", default="15-19")
    args = ap.parse_args()
    a, _, b = args.runs.partition("-")
    cm, per_frame = evaluate(getattr(screen_state, args.method), range(int(a), int(b or a) + 1))
    text = report(cm, per_frame)
    print(text)
    out = Path(f"runs/screen_state_{args.method}.txt")
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
