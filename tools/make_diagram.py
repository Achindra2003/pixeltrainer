"""System overview diagram for the user manual (docs/figures/fig0_pipeline.png).

Static drawing (no emulator): the frame goes in, each stage is a box with its method and syllabus unit,
one button comes out. Usage: python tools/make_diagram.py [--out docs/figures]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INPUT, STATE, WALK, FIGHT, OUT, CAM = "#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec", "#ede7f6", "#f5f5f5"


def box(ax, x, y, w, h, title, method, unit, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=colour, ec="#555", lw=1))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=8.6, weight="bold")
    ax.text(x + w / 2, y + h * 0.42, method, ha="center", va="center", fontsize=7.8)
    if unit:
        ax.text(x + w / 2, y + h * 0.16, unit, ha="center", va="center", fontsize=7, color="#666", style="italic")


def arrow(ax, a, b, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=1, color="#444", ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def lane_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=9, weight="bold", color="#333", va="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    out = Path(ap.parse_args().out)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    W, H = 1.9, 0.95

    # optional camera input (dashed), feeding the frame
    ax.add_patch(FancyBboxPatch((0.2, 5.85), 6.1, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec="#888", lw=1, ls="--"))
    ax.text(0.35, 6.9, "Camera mode (optional input): a simulated photo of the screen", fontsize=8.5,
            weight="bold", color="#555", va="center")
    cam = [("Find screen", "Sobel + Hough", "U3-T1"), ("Restore", "Wiener deblur", "U2-T1/T2"),
           ("Straighten", "perspective warp", "U1-T2"), ("4 shades", "histogram valleys", "U2-T1")]
    for i, (t, m, u) in enumerate(cam):
        box(ax, 0.35 + i * 1.5, 5.95, 1.35, 0.78, t, m, u, CAM)
        if i:
            arrow(ax, (0.35 + i * 1.5 - 0.15, 6.34), (0.35 + i * 1.5, 6.34))

    # input frame and screen type
    box(ax, 0.2, 3.55, W, 1.6, "Screen frame", "160×144, 4 grey shades\nthe ONLY input", "U1-T2 quantisation", INPUT)
    arrow(ax, (5.37, 5.95), (1.9, 5.17), ls="--", rad=-0.1)  # from the last camera step
    box(ax, 2.55, 3.85, W, H + 0.1, "Screen type", "CNN (4 conv layers)", "U1-T1 convolution", STATE)
    arrow(ax, (2.12, 4.35), (2.55, 4.35))

    # overworld lane
    lane_label(ax, 4.75, 5.55, "Overworld: walk to the goal")
    walk = [("Where to walk?", "cell hash + CNN", "U3-T1 features"),
            ("People nearby?", "components + NCC", "U1-T2, U1-T1"),
            ("Where am I?", "MAD + phase corr.", "U2-T2 Fourier"),
            ("Arrived?", "template match", "U3-T1 threshold"),
            ("Where next?", "region growing + A*", "U3-T1, U1-T2")]
    xs = [4.75 + i * 1.62 for i in range(5)]
    for x, (t, m, u) in zip(xs, walk):
        box(ax, x, 4.45, 1.5, H, t, m, u, WALK)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 1.5, 4.92), (b, 4.92))

    # battle lane
    lane_label(ax, 4.75, 3.62, "Battle: pick the best attack")
    fight = [("Which Pokémon?", "NCC vs 151 sprites", "U1-T1 correlation"),
             ("Which moves?", "OCR (glyph match)", "U3-T1 matching"),
             ("How much HP?", "count filled pixels", "U1-T2"),
             ("Best move", "expected damage", "")]
    for i, (t, m, u) in enumerate(fight):
        box(ax, xs[i], 2.5, 1.5, H, t, m, u, FIGHT)
        if i:
            arrow(ax, (xs[i - 1] + 1.5, 2.97), (xs[i], 2.97))

    # dialogue / menu lane
    lane_label(ax, 4.75, 1.72, "Dialogue / menu / transition")
    box(ax, xs[0], 0.75, 1.5, 0.75, "Text or menu", "press A / B, or wait", "", STATE)

    # branches from screen type
    arrow(ax, (4.45, 4.55), (xs[0], 4.92), rad=-0.15)
    arrow(ax, (4.45, 4.15), (xs[0], 2.97), rad=0.15)
    arrow(ax, (3.5, 3.85), (xs[0], 1.12), rad=0.25)

    # button out
    box(ax, 11.25, 0.55, 1.55, 1.1, "One button", "up / down / left /\nright / A / B", "", OUT)
    arrow(ax, (xs[4] + 0.75, 4.45), (12.0, 1.67), rad=-0.1)
    arrow(ax, (xs[3] + 1.5, 2.97), (11.6, 1.67), rad=-0.2)
    arrow(ax, (xs[0] + 1.5, 1.12), (11.25, 1.1))

    # loop back
    ax.plot([12.0, 12.0, 1.15, 1.15], [0.55, 0.3, 0.3, 3.0], color="#444", lw=1, ls=":")
    arrow(ax, (1.15, 3.0), (1.15, 3.53), ls=":")
    ax.text(6.3, 0.08, "the game moves on, the screen settles, and the loop repeats with the next frame",
            fontsize=8, color="#555", ha="center", style="italic")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig0_pipeline.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
