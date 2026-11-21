"""PSNR / SSIM / shade accuracy of every restoration method x degradation type (task 8.4).

Usage: python evaluation/eval_restore.py [--n 30] [--strength 2]
Each type alone at the given strength (no geometric, so corners are degrade.py's and the background is flat).
The method runs on the captured grey image (flatten_light on the BGR one); the result is rectified with the
true corners and compared with the clean screen (palette grey) for PSNR and SSIM, and quantised for shade
accuracy. "frame average" averages 4 captures with independent noise (seeds k, k+1000, ...): it applies
to the noise types only. "auto gamma" maps the mean to mid-grey: y = log(0.5) / log(mean / 255).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.degrade import PALETTE, TYPES, degrade  # noqa: E402
from evaluation.eval_localise import test_frames  # noqa: E402
from pixeltrainer.capture import restore as R  # noqa: E402
from pixeltrainer.capture.quantise import quantise  # noqa: E402
from pixeltrainer.capture.rectify import rectify  # noqa: E402

NOISE = ("gaussian", "saltpepper", "poisson")


def _grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _auto_gamma(g):
    m = np.clip(g.mean() / 255, 0.02, 0.98)
    return R.gamma(g, np.log(0.5) / np.log(m))


def _blind_wiener(img, corners):
    g = _grey(img)
    return min(((R._shade_spread(rectify(w, corners)), w) for s in R.WIENER_SIGMAS for w in [R.wiener(g, s)]),
               key=lambda x: x[0])[1]


METHODS = {
    "none": lambda img, c: _grey(img),
    "mean 3x3": lambda img, c: R.mean(_grey(img)),
    "gaussian s=1": lambda img, c: R.gaussian(_grey(img)),
    "median 3x3": lambda img, c: R.median(_grey(img)),
    "adaptive median": lambda img, c: R.adaptive_median(_grey(img)),
    "unsharp": lambda img, c: R.unsharp(_grey(img)),
    "laplacian sharpen": lambda img, c: R.laplacian_sharpen(_grey(img)),
    "gaussian low-pass": lambda img, c: R.gaussian_lowpass(_grey(img)),
    "butterworth low-pass": lambda img, c: R.butterworth_lowpass(_grey(img)),
    "fft notch": lambda img, c: R.notch_moire(_grey(img)),
    "wiener (blind sigma)": _blind_wiener,
    "auto gamma": lambda img, c: _auto_gamma(_grey(img)),
    "hist. equalisation": lambda img, c: R.equalise(_grey(img)),
    "clahe": lambda img, c: R.clahe(_grey(img)),
    "flatten light": lambda img, c: R.flatten_light(img),
    "chain (restore)": lambda img, c: R.restore(img, c)[0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--strength", type=int, default=2)
    args = ap.parse_args()
    frames = test_frames(args.n)
    types = [t for t in TYPES if t != "geometric"]
    names = list(METHODS) + ["frame average x4"]
    res = {}  # (method, type) -> (psnr, ssim, acc) means
    for t in types:
        rows = {m: [] for m in names}
        for k, f in enumerate(frames):
            ref = PALETTE[f].astype(np.float32)
            imgs = [degrade(f, {t: args.strength}, seed=k + 1000 * i) for i in range(4 if t in NOISE else 1)]
            img, meta = imgs[0]
            c = meta["corners"]
            outs = {m: fn(img, c) for m, fn in METHODS.items()}
            if t in NOISE:
                outs["frame average x4"] = R.frame_average([_grey(i) for i, _ in imgs])
            for m, out in outs.items():
                r = np.clip(rectify(out, c), 0, 255)
                rows[m].append((np.mean((ref - r) ** 2),  # MSE; PSNR is taken of the mean (an exact frame is inf dB)
                                structural_similarity(ref, r, data_range=255), np.mean(quantise(r) == f)))
        for m, v in rows.items():
            if v:
                res[m, t] = np.mean(v, 0)
                res[m, t][0] = 10 * np.log10(255 ** 2 / max(res[m, t][0], 1e-10))
        print(f"done {t}", file=sys.stderr, flush=True)

    for title, i, fmt in (("PSNR (dB)", 0, "{:6.1f}"), ("SSIM", 1, "{:6.3f}"), ("shade accuracy (%)", 2, "{:6.1f}")):
        print(f"\n{title}, strength {args.strength}, {len(frames)} frames, rectified with true corners")
        print(f"{'method':22s}" + "".join(f"{t[:9]:>10s}" for t in types))
        for m in names:
            cells = [(fmt.format(res[m, t][i] * (100 if i == 2 else 1)) if (m, t) in res else "     -").rjust(10)
                     for t in types]
            print(f"{m:22s}" + "".join(cells))


if __name__ == "__main__":
    main()
