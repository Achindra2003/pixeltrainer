"""Shade accuracy of the capture chain localise -> rectify -> quantise (tasks 8.3, 8.5).

Usage: python evaluation/eval_capture.py [--n 200] [--levels 0-4]
Same test frames as eval_localise.py. Accuracy = share of the 160x144 pixels whose shade equals the clean
frame's. "true corners" rectifies with degrade.py's corners, so it isolates rectify + quantise from
localisation; "found corners" is the chain without restoration; "restored" adds restore.restore() with the
found corners (the agent's chain). A frame with no corners scores 0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.degrade import LEVELS, degrade  # noqa: E402
from evaluation.eval_localise import test_frames  # noqa: E402
from pixeltrainer.capture.localise import localise  # noqa: E402
from pixeltrainer.capture.quantise import quantise  # noqa: E402
from pixeltrainer.capture.rectify import rectify  # noqa: E402
from pixeltrainer.capture.restore import restore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--levels", default="0-4")
    args = ap.parse_args()
    frames = test_frames(args.n)
    a, _, b = args.levels.partition("-")
    for level in range(int(a), int(b or a) + 1):
        true_acc, found_acc, rest_acc = [], [], []
        for k, f in enumerate(frames):
            img, meta = degrade(f, LEVELS[level], seed=k)
            true_acc.append(np.mean(quantise(rectify(img, meta["corners"])) == f))
            got = localise(img)
            found_acc.append(0.0 if got is None else np.mean(quantise(rectify(img, got)) == f))
            rest_acc.append(0.0 if got is None else np.mean(quantise(rectify(restore(img, got)[0], got)) == f))
        t, g, r = np.array(true_acc), np.array(found_acc), np.array(rest_acc)
        print(f"level {level}: true corners mean {t.mean():.2%} (frames >= 99%: {np.mean(t >= 0.99):.1%}); "
              f"found corners mean {g.mean():.2%} (frames >= 99%: {np.mean(g >= 0.99):.1%}); "
              f"restored mean {r.mean():.2%} (frames >= 99%: {np.mean(r >= 0.99):.1%})", flush=True)


if __name__ == "__main__":
    main()
