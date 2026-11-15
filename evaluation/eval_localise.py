"""Screen corner error of capture/localise.py on degraded test frames (task 8.2).

Usage: python evaluation/eval_localise.py [--n 500] [--levels 0-4]
Frames: every k-th stable frame of test runs 15-19; each degraded with its own seed. Error = max over the
4 corners of the distance to the true corner (degrade.py meta), in captured-image pixels.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.degrade import LEVELS, degrade  # noqa: E402
from evaluation.eval_registration import frame  # noqa: E402
from pixeltrainer.capture.localise import localise  # noqa: E402


def test_frames(n):
    items = []
    for run in range(15, 20):
        d = Path(f"data/raw/{run:02d}")
        items += [(d, json.loads(line)) for line in open(d / "labels.jsonl")]
    items = [x for x in items if x[1]["stable"]]
    return [frame(d, t) for d, t in items[::max(1, len(items) // n)][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--levels", default="0-4")
    args = ap.parse_args()
    frames = test_frames(args.n)
    a, _, b = args.levels.partition("-")
    for level in range(int(a), int(b or a) + 1):
        errs = []
        for k, f in enumerate(frames):
            img, meta = degrade(f, LEVELS[level], seed=k)
            got = localise(img)
            errs.append(np.inf if got is None else float(np.linalg.norm(got - meta["corners"], axis=1).max()))
        e = np.array(errs)
        print(f"level {level}: corner error median {np.median(e):.2f} px, 95th pct {np.percentile(e, 95):.2f}, "
              f"<= 1 px {np.mean(e <= 1):.1%}, failed (> 5 px) {np.mean(e > 5):.1%}", flush=True)


if __name__ == "__main__":
    main()
