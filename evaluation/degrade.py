"""Simulated capture degradation (PLAN.md §6A.1, task 8.1). Evaluation side: it gets the clean frame,
the agent gets only the returned image.

Model g = h * f + n: the screen is warped onto a cluttered background (geometric), blurred (h),
lit unevenly and colour-shifted (photometric), overlaid with a periodic pattern (moiré), then noise n.
Every type has strengths 1-4; LEVELS combines them into levels 0-4. Seeded, so runs repeat.

Usage: python evaluation/degrade.py [frame.png]   saves runs/degrade/contact.png (type x strength)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PALETTE = np.array([255, 153, 85, 0], np.uint8)  # shade index -> grey, as PyBoy draws it
CANVAS = (360, 400)  # (h, w) of the captured image
TYPES = ("geometric", "defocus", "motion", "gaussian", "saltpepper", "poisson",
         "gamma", "illumination", "cast", "moire")
LEVELS = {
    0: {},
    1: {"geometric": 1, "defocus": 1, "gaussian": 1, "cast": 1},
    2: {"geometric": 2, "defocus": 2, "gaussian": 2, "cast": 2, "gamma": 2, "illumination": 1},
    3: {"geometric": 3, "motion": 2, "gaussian": 3, "saltpepper": 1, "cast": 3, "gamma": 3,
        "illumination": 2, "moire": 2},
    4: {"geometric": 4, "motion": 3, "gaussian": 3, "poisson": 3, "saltpepper": 2, "cast": 4, "gamma": 4,
        "illumination": 3, "moire": 3},
}


def _corners(s, rng):
    """Screen corners (tl, tr, br, bl) on the canvas. Strength 0: centred at 2x, axis-aligned."""
    h, w = CANVAS
    scale = 2.0 * (1 - 0.06 * s * rng.random())
    c = np.array([[-80, -72], [80, -72], [80, 72], [-80, 72]], np.float64) * scale
    a = np.deg2rad(3 * s * rng.uniform(-1, 1))
    c = c @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    c += rng.uniform(-6 * s, 6 * s, c.shape)  # perspective: move each corner independently
    c += [w / 2 + rng.uniform(-8, 8) * s, h / 2 + rng.uniform(-8, 8) * s]
    return c.astype(np.float32)


def _background(s, rng):
    h, w = CANVAS
    bg = np.full((h, w, 3), 110, np.uint8)
    for _ in range(12 * s):  # clutter: random boxes and lines, edges the localiser must reject
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(-80, 80)), y0 + int(rng.integers(-80, 80))
        if rng.random() < 0.5:
            cv2.rectangle(bg, (x0, y0), (x1, y1), col, -1)
        else:
            cv2.line(bg, (x0, y0), (x1, y1), col, int(rng.integers(1, 4)))
    return bg


def _motion_psf(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1), (length, length))
    return k / k.sum()


def degrade(frame, types, seed=0):
    """frame: (144, 160) shade indices 0-3. types: {type: strength 1-4}, e.g. LEVELS[2].
    Returns (image (360, 400, 3) uint8 BGR, meta) where meta holds the true screen corners and the blur PSF."""
    rng = np.random.default_rng(seed)
    s = {t: types.get(t, 0) for t in TYPES}
    unknown = set(types) - set(TYPES)
    if unknown:
        raise ValueError(f"unknown degradation types: {unknown}")
    screen = cv2.cvtColor(PALETTE[frame], cv2.COLOR_GRAY2BGR)

    corners = _corners(s["geometric"], rng)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [160, 0], [160, 144], [0, 144]]), corners)
    warped = cv2.warpPerspective(screen, H, CANVAS[::-1], flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((144, 160), np.float32), H, CANVAS[::-1])[..., None]
    img = warped * mask + _background(s["geometric"], rng) * (1 - mask)

    psf = np.ones((1, 1), np.float32)
    if s["defocus"]:
        sigma = 0.7 * s["defocus"]
        g = cv2.getGaussianKernel(2 * int(3 * sigma) + 1, sigma)
        psf = (g @ g.T).astype(np.float32)
    if s["motion"]:  # combined PSF = motion kernel convolved with the defocus kernel
        r = psf.shape[0] // 2
        psf = cv2.filter2D(np.pad(_motion_psf(4 * s["motion"] + 1, rng.uniform(0, 180)), r), -1, psf,
                           borderType=cv2.BORDER_CONSTANT)
    if psf.size > 1:
        img = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REFLECT)

    img = np.clip(img / 255.0, 0, 1)  # filter2D leaves tiny negatives; a fractional power of those is NaN
    if s["gamma"]:
        img = img ** (1 + 0.25 * s["gamma"] * rng.choice([-1, 1]) * rng.uniform(0.6, 1))
    if s["illumination"]:  # linear light falloff across the image in a random direction
        h, w = CANVAS
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = ((xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a)) / (w / 2)
        img = img * (1 + 0.15 * s["illumination"] * ramp)[..., None]
    if s["cast"]:
        img = img * (1 + 0.08 * s["cast"] * rng.uniform(-1, 1, 3))
    if s["moire"]:
        h, w = CANVAS
        f, a = rng.uniform(0.15, 0.3), rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        img = img + (0.025 * s["moire"] * np.sin(2 * np.pi * f * (xx * np.cos(a) + yy * np.sin(a))))[..., None]

    img = np.clip(img, 0, 1)
    if s["poisson"]:
        peak = {1: 120, 2: 50, 3: 20, 4: 8}[s["poisson"]]
        img = rng.poisson(img * peak) / peak
    if s["gaussian"]:
        img = img + rng.normal(0, 0.02 * s["gaussian"], img.shape)
    img = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    if s["saltpepper"]:
        r = rng.random(CANVAS)
        d = 0.01 * s["saltpepper"]
        img[r < d / 2], img[r > 1 - d / 2] = 0, 255
    return img, {"corners": corners, "psf": psf, "types": dict(types)}


def contact_sheet(frame, seed=0):
    rows = [[degrade(frame, {t: k}, seed)[0] for k in range(1, 5)] for t in TYPES]
    rows.append([degrade(frame, LEVELS[k], seed)[0] for k in range(1, 5)])
    sheet = np.vstack([np.hstack(r) for r in rows])
    for i, name in enumerate(list(TYPES) + ["levels"]):
        cv2.putText(sheet, name, (6, i * CANVAS[0] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return sheet


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/00/frames/000000.png"
    frame = ((255 - cv2.imread(src, cv2.IMREAD_GRAYSCALE)) // 85).astype(np.uint8)  # dataset stores 255 - 85*s
    out = Path("runs/degrade/contact.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(contact_sheet(frame), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    print(f"saved {out}")
