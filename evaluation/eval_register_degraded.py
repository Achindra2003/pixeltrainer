"""Registration under degradation: phase correlation vs ORB + RANSAC (task 8.6).

Usage: python evaluation/eval_register_degraded.py [--n 300] [--levels 0-4]
Step pairs are eval_registration.py's (runs 15-19, same map, both overworld), evenly sampled. Each frame of
a pair is degraded independently (seeds 2k, 2k+1) and run through the agent's capture chain
(localise -> restore -> rectify -> quantise). Truth = RAM coordinate change. A pair whose screen is not
found counts as wrong for every method.
Methods: phase correlation (registration.step, no direction); ORB keypoints on the 2x shade image, Hamming
brute-force matching with cross-check, RANSAC translation-plus-rotation (estimateAffinePartial2D), the
translation snapped to whole cells; the agent's direction-prior matching for reference.
On clean pairs (134, every 20th) ORB is right on 96: every miss reads a real step as (0, 0). Grass and path
repeat every 16 px, so after a one-cell scroll those keypoints match their own position and RANSAC's
largest consensus is zero shift. Lowe's ratio test (0.7 / 0.8: 91 / 94) and masking the player sprite out
of detection (no change) do not help.
"""
import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.degrade import LEVELS, degrade  # noqa: E402
from evaluation.eval_registration import frame, pairs  # noqa: E402
from pixeltrainer.capture.localise import localise  # noqa: E402
from pixeltrainer.capture.quantise import quantise  # noqa: E402
from pixeltrainer.capture.rectify import rectify  # noqa: E402
from pixeltrainer.capture.restore import restore  # noqa: E402
from pixeltrainer.mapping import registration  # noqa: E402

ORB = cv2.ORB_create(nfeatures=500, edgeThreshold=15, patchSize=15, fastThreshold=10)
MIN_INLIERS = 8  # fewer RANSAC inliers than this: no confident shift (a transition, or nothing matched)
SNAP_TOL = 4.0  # px from a whole cell


def capture(img):
    got = localise(img)
    return None if got is None else quantise(rectify(restore(img, got)[0], got))


def orb_step(prev, cur):
    """Player move in cells from ORB matches, or None. Shade images are upscaled 2x (nearest) so the
    8 px tiles are wide enough for the FAST/BRIEF patch."""
    a, b = (cv2.resize((85 * f).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for f in (prev, cur))
    ka, da = ORB.detectAndCompute(a, None)
    kb, db = ORB.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < MIN_INLIERS:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl.sum() < MIN_INLIERS:
        return None
    dx, dy = M[0, 2] / 2, M[1, 2] / 2  # content shift in screen px; the player moved the other way
    cx, cy = round(-dx / 16), round(-dy / 16)
    if abs(-dx - 16 * cx) > SNAP_TOL or abs(-dy - 16 * cy) > SNAP_TOL:
        return None
    return cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--levels", default="0-4")
    args = ap.parse_args()
    items = [(p, t, d) for run in range(15, 20) for p, t, d, _ in pairs(run)]
    items = items[::max(1, len(items) // args.n)][:args.n]
    a, _, b = args.levels.partition("-")
    for level in range(int(a), int(b or a) + 1):
        stats = collections.defaultdict(collections.Counter)
        for k, (p, t, d) in enumerate(items):
            truth = (t["x"] - p["x"], t["y"] - p["y"])
            kind = "jump" if max(map(abs, truth)) > 1 else "step" if any(truth) else "blocked"
            f0, f1 = (capture(degrade(frame(d, x), LEVELS[level], seed=2 * k + j)[0]) for j, x in enumerate((p, t)))
            ok = f0 is not None and f1 is not None
            for m, fn in (("phase", lambda: registration.step(f0, f1)),
                          ("orb+ransac", lambda: orb_step(f0, f1)),
                          ("direction prior", lambda: registration.step(f0, f1, t["action"]))):
                stats[m][(kind, ok and fn() == truth)] += 1
        for m, c in stats.items():
            n_ok = sum(v for (_, good), v in c.items() if good)
            print(f"level {level} {m}: {n_ok}/{sum(c.values())} ({n_ok / sum(c.values()):.1%}); " +
                  ", ".join(f"{kd} {c[(kd, True)]}/{c[(kd, True)] + c[(kd, False)]}" for kd in ("step", "blocked", "jump")),
                  flush=True)


if __name__ == "__main__":
    main()
