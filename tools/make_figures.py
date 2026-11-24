"""Input/output figures for the user manual (docs/figures/, USER_MANUAL.md).

Plays one clean episode (seed 0, type-aware policy) and keeps example frames for each stage of the
pipeline, then runs the simulated-capture chain on one overworld frame at levels 1 and 2. Every figure
shows the input the stage gets next to the output it produces. RAM is not read (only the raw RGB screen
for figure 1, which the agent never sees in that form).

Usage: python tools/make_figures.py [--seed 0] [--out docs/figures]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.degrade import LEVELS, PALETTE, degrade  # noqa: E402
from pixeltrainer import viz  # noqa: E402
from pixeltrainer.agent import Agent, run  # noqa: E402
from pixeltrainer.capture.localise import localise  # noqa: E402
from pixeltrainer.capture.quantise import quantise  # noqa: E402
from pixeltrainer.capture.rectify import rectify  # noqa: E402
from pixeltrainer.capture.restore import restore  # noqa: E402
from pixeltrainer.emu import DIRECTIONS, Emu  # noqa: E402
from pixeltrainer.mapping import registration  # noqa: E402
from pixeltrainer.perception import battle, grid, landmark, npc, ocr, screen_state, tiles  # noqa: E402

COLOURS = {"walkable": "#4caf50", "grass": "#1b5e20", "wall": "#e53935", "water": "#1e88e5",
           "door": "#ffb300", "ledge_s": "#8e24aa", "ledge_e": "#8e24aa", "ledge_w": "#8e24aa",
           "sign": "#00acc1", None: "#9e9e9e"}
STATES = ("overworld", "battle", "dialogue", "menu", "transition")


def show(ax, frame, title=None):
    """Shade frame (0 white .. 3 black) or any grey/BGR image, pixel-exact."""
    if frame.ndim == 2 and frame.max() <= 3:
        ax.imshow(PALETTE[frame], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    elif frame.ndim == 2:
        ax.imshow(frame, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def save(fig, out, name, caption):
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


class Collector:
    """on_decision callback: keeps the first useful example of each case."""

    def __init__(self, agent):
        self.agent, self.prev = agent, None
        self.states, self.step, self.main, self.fight, self.plan = {}, None, None, None, None
        self.first = None

    def __call__(self, n, frame, action):
        a, f = self.agent, frame.copy()
        self.states.setdefault(a.state, f)
        if a.state == "overworld":
            self.first = self.first if self.first is not None else f
            if (self.step is None and n > 20 and self.prev is not None and self.prev[1] in DIRECTIONS
                    and self.prev[2] == "overworld"
                    and registration.step(self.prev[0], f, self.prev[1]) not in (None, (0, 0))):
                self.step = (self.prev[0], f, self.prev[1])
            if self.plan is None and n >= 40 and a.plan and a.plan[0] and len(a.plan[0]) >= 4:
                self.plan = viz.compose(f, a, n, action)
        elif a.state == "battle":
            m = battle.menu(f)
            if m == "main" and self.main is None:
                self.main = f
            if m == "fight" and self.fight is None and a.reason:
                self.fight = (f, a.reason, battle.moves(f))
        self.prev = (f, action, a.state)


def fig_input(emu, frame, out):
    raw = emu._pyboy.screen.ndarray[:, :, :3]  # figure only; the agent gets Emu.frame()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(raw, interpolation="nearest")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_title("Emulator screen, RGBA 160x144", fontsize=9)
    show(ax[1], frame, "Agent input: shade indices 0-3, 9x10 grid of 16x16 cells")
    for k in range(1, 10):
        ax[1].axvline(16 * k - 0.5, color="red", lw=0.4)
    for k in range(1, 9):
        ax[1].axhline(16 * k - 0.5, color="red", lw=0.4)
    r, c = grid.PLAYER
    ax[1].add_patch(patches.Rectangle((16 * c - 0.5, 16 * r - 0.5), 16, 16, fill=False, ec="blue", lw=1.5))
    vals, counts = np.unique(frame, return_counts=True)
    ax[2].bar(vals, counts, color=["#eeeeee", "#999999", "#555555", "#000000"][:len(vals)], ec="k")
    ax[2].set_xticks(range(4), ["0 white", "1 light", "2 dark", "3 black"])
    ax[2].set_title("Histogram: exactly 4 shades (no noise)", fontsize=9)
    save(fig, out, "fig1_input.png",
         "Fig 1. The only input: one 160x144 frame. Quantised to 4 shade indices (sampling and quantisation). "
         "The world is a 9x10 grid of 16x16 cells; the player is always the blue cell (row 4, col 4).")


def test_examples(run=18):
    """{state: frame}: the middle frame of the longest labelled stretch of each state in a test run."""
    d = Path(f"data/raw/{run:02d}")
    labels = json.loads((d / "screen.json").read_text())
    best = {}
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i > best.get(labels[i], (0, 0))[0]:
            best[labels[i]] = (j - i, (i + j) // 2)
        i = j
    return {s: (255 - cv2.imread(str(d / "frames" / f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85
            for s, (_, k) in best.items()}


def fig_states(out):
    states = test_examples()
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    for a, s in zip(ax, STATES):
        f = states[s]
        show(a, f, f"label: {s}\nCNN: {screen_state.cnn(f)} | heuristic: {screen_state.heuristic(f)}")
    save(fig, out, "fig2_screen_state.png",
         "Fig 2. Screen-state classification. Input: whole frame. Output: one of 5 states, which picks the "
         "rest of the pipeline. Frames from unseen test run 18. Test accuracy 99.49% (CNN) vs 95.36% (glyph heuristic) on 16,432 frames.")


def overlay(ax, labels, sprites=()):
    for r in range(grid.ROWS):
        for c in range(grid.COLS):
            if (r, c) in sprites:  # the map masks sprite cells (worldmap.sprite_mask); drawn as such
                ax.add_patch(patches.Rectangle((16 * c - 0.5, 16 * r - 0.5), 16, 16, fill=False, ec="blue", lw=1.5,
                                               hatch="//"))
                continue
            lab = labels[r, c]
            ax.add_patch(patches.Rectangle((16 * c - 0.5, 16 * r - 0.5), 16, 16, color=COLOURS[lab], alpha=0.45,
                                           hatch=None if lab else "xx", lw=0))


def fig_tiles(frame, out):
    lk, _ = tiles.lookup(frame)
    cn, conf = tiles.cnn(frame)
    sprites = {grid.PLAYER} | {d["cell"] for d in npc.detect(frame)}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    show(ax[0], frame, "Input: overworld frame")
    show(ax[1], frame, f"Lookup: {int((lk != None).sum())}/90 cells known")  # noqa: E711
    overlay(ax[1], lk, sprites)
    show(ax[2], frame, f"CNN: all 90 cells, mean confidence {conf.mean():.3f}")
    overlay(ax[2], cn, sprites)
    ax[2].legend(handles=[patches.Patch(color=COLOURS[k], label=k) for k in
                          ("walkable", "grass", "wall", "water", "door", "ledge_s", "sign", None)]
                 + [patches.Patch(fill=False, ec="blue", hatch="//")],
                 labels=["walkable", "grass", "wall", "water", "door", "ledge", "sign", "unknown (lookup miss)",
                         "sprite cell (masked)"],
                 loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, out, "fig3_tiles.png",
         "Fig 3. Tile classification per 16x16 cell. Output: the walkable/blocked map the planner needs. "
         "Sprite cells (player, NPCs) are masked, the map keeps their last background label. "
         "Lookup (exact hash) and CNN both agree with the game's collision data on 100% of test cells.")


def busiest_npc_frame(runs=(15, 16, 17, 18, 19), every=5):
    """The overworld test frame with the most NPC detections (first one on ties)."""
    best, frame = -1, None
    for run in runs:
        d = Path(f"data/raw/{run:02d}")
        labels = json.loads((d / "screen.json").read_text())
        for i in range(0, len(labels), every):
            if labels[i] != "overworld":
                continue
            f = (255 - cv2.imread(str(d / "frames" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)) // 85
            k = len(npc.detect(f))
            if k > best:
                best, frame = k, f
    return frame


def fig_npc(out):
    frame = busiest_npc_frame()
    found = npc.detect(frame)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    show(ax[0], frame, "Input: overworld frame")
    show(ax[1], frame, f"Output: {len(found)} NPC(s) by masked NCC")
    for d in found:
        y, x, h, w = d["box"]
        ax[1].add_patch(patches.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, ec="magenta", lw=2))
        ax[1].text(x, y - 2, f"{d['facing']} {d['score']:.2f}", color="magenta", fontsize=8)
    save(fig, out, "fig4_npc.png",
         "Fig 4. NPC detection: dark connected components -> candidate boxes -> masked NCC against templates. "
         "Output: box, facing and score; the planner treats the cell as blocked. Test precision 100%, recall 97.5%. "
         "Frame shown: the unseen test frame (runs 15-19) with the most detections.")


def corr_surface(a, b):
    fa, fb = (np.fft.fft2(registration._prep(x, (grid.PLAYER,))) for x in (a, b))
    r = fa * np.conj(fb)
    return np.fft.fftshift(np.real(np.fft.ifft2(r / (np.abs(r) + 1e-9))))


def fig_registration(prev, cur, direction, out):
    dx, dy, resp = registration.shift(prev, cur)
    move = registration.step(prev, cur, direction)
    surf = corr_surface(prev, cur)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    show(ax[0], prev, "Input: frame t")
    show(ax[1], cur, f"Input: frame t+1 (pressed {direction})")
    ax[2].imshow(surf, cmap="magma", extent=(-80, 80, 72, -72))
    ax[2].set_title(f"Phase correlation: peak dx={dx:+.1f}, dy={dy:+.1f}, response {resp:.2f}", fontsize=9)
    diff = np.abs(cur.astype(int) - np.roll(prev, (int(round(dy)), int(round(dx))), (0, 1)).astype(int))
    show(ax[3], (diff * 85).astype(np.uint8), f"Output: move {move} cells; |t+1 - shifted t|")
    save(fig, out, "fig5_registration.png",
         "Fig 5. Frame registration by phase correlation (Fourier transform). One step scrolls the screen by "
         "exactly 16 px; the peak gives the shift, and the aligned difference is near zero except sprites.")


def fig_map(agent, plan_img, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    show(ax[0], plan_img, "Decision mid-route: game (left), agent's map with A* path (yellow), target (magenta)")
    show(ax[1], agent.map.render(), f"Final map of the last segment ({agent.map.n_segments} segment(s))")
    save(fig, out, "fig6_map_plan.png",
         "Fig 6. World map built only from registered frames (mosaic by translation), walls red, doors green, "
         "player blue. Region growing finds the frontier; A* plans the path. Map position error vs RAM: 0 tiles.")


def fig_battle(main, fight, out):
    f, reason, moves = fight
    sid, name, score = battle.species(main)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    show(ax[0], main, f"Main menu: species {name} (NCC {score:.2f})\nenemy HP {battle.hp_fraction(main):.0%}, "
                      f"own HP {battle.hp_fraction(main, 'own'):.0%}")
    ax[0].add_patch(patches.Rectangle((95.5, -0.5), 56, 56, fill=False, ec="magenta", lw=2))
    for side, (y, x0) in battle.BARS.items():
        ax[0].add_patch(patches.Rectangle((x0 - 0.5, y - 0.5), 48, 2, fill=False, ec="red", lw=1.5))
    cur = battle.cursor(main)
    ax[0].set_xlabel(f"menu={battle.menu(main)}, cursor={cur} -> press "
                     f"{'A (FIGHT)' if cur == 0 else 'direction to FIGHT'}", fontsize=9, color="blue")
    show(ax[1], f, "Fight menu: OCR of the move list")
    ax[1].add_patch(patches.Rectangle((47.5, 103.5), 104, 32, fill=False, ec="red", lw=1.5))
    ax[1].set_xlabel(f"OCR {moves} -> policy: {reason}", fontsize=9, color="blue")
    save(fig, out, "fig7_battle.png",
         "Fig 7. Battle reading. Species by NCC against 151 sprites (100%), HP bars by thresholded fill "
         "(MAE 0.63%), menu and moves by 8x8 glyph OCR (99.99%). The policy picks the best expected damage.")


def fig_landmark(frame, out):
    score, (x, y) = landmark.find(frame)
    h, w = landmark.load("poke_sign").shape
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 3, 3]})
    show(ax[0], landmark.load("poke_sign").astype(np.uint8), "Template")
    show(ax[1], frame, "Input: last frame")
    show(ax[2], frame, f"Output: NCC {score:.3f} >= {landmark.THRESHOLD} -> goal reached")
    ax[2].add_patch(patches.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, ec="lime", lw=2))
    save(fig, out, "fig8_landmark.png",
         "Fig 8. Goal test by template matching (normalised cross-correlation). The Viridian City Poke Center "
         "sign is found; the agent stops. 60/60 evaluation episodes ended here.")


def fig_capture(frame, out, seed=7):
    rows = []
    for level in (1, 2):
        img, meta = degrade(frame, LEVELS[level], seed=seed)
        corners = localise(img)
        restored, sigma = restore(img, corners)
        plain = quantise(rectify(img, corners))
        q = quantise(rectify(restored, corners))
        rows.append((level, img, meta["corners"], corners, restored, sigma, plain, q))
    fig, ax = plt.subplots(2, 5, figsize=(19, 7.5))
    for row, (level, img, true, found, restored, sigma, plain, q) in zip(ax, rows):
        err = np.abs(np.float32(found) - np.float32(true)).max()
        show(row[0], img, f"Level {level} input:\nsimulated camera photo")
        marked = img.copy()
        cv2.polylines(marked, [np.int32(found)], True, (0, 255, 0), 2)
        show(row[1], marked, f"Screen found (green)\nmax corner error {err:.2f} px")
        show(row[2], restored, f"Light flattened +\nWiener deconvolution (sigma {sigma})")
        show(row[3], q, f"Rectified + quantised: {(q == frame).mean():.2%} right\n"
                        f"(no restoration {(plain == frame).mean():.2%})")
        show(row[4], ((q != frame) * 255).astype(np.uint8), "Wrong shades (white)\nvs the clean frame")
    save(fig, out, "fig9_capture.png",
         "Fig 9. Simulated capture (sim mode). Level 1: localise (Hough lines + edges), restore, rectify "
         "(perspective warp), quantise (histogram valleys) -> nearly the clean frame. Level 2: blur erases "
         "1-px detail and restoration cannot bring it back; shades fail, which is why the agent fails from level 2 up.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emu = Emu(state="states/pallet_start.state", seed=args.seed)
    first = emu.frame()
    fig_input(emu, first, out)
    agent = Agent(seed=args.seed, policy="type")
    col = Collector(agent)
    last = {}
    try:
        n = run(emu, agent, 1500, lambda n, f, a: (col(n, f, a), last.update(frame=f.copy())))
    finally:
        emu.close()
    print(f"episode: {'landmark seen' if agent.done else 'budget used up'} after {n} decisions")
    fig_states(out)
    fig_tiles(col.first, out)
    fig_npc(out)
    if col.step:
        fig_registration(*col.step, out)
    if col.plan is not None:
        fig_map(agent, col.plan, out)
    if col.main is not None and col.fight:
        fig_battle(col.main, col.fight, out)
    if agent.done:
        fig_landmark(last["frame"], out)
    fig_capture(col.first, out)


if __name__ == "__main__":
    main()
