"""Robustness curve from the 8.7 sweep (task 8.7).

Usage: python evaluation/plot_robustness.py
Reads runs/jobs/agent_sim<level>[_norestore]_s<seed>.txt (tools/sweep_queue.sh) and writes
runs/robustness.txt (success, decisions, segments, screen-not-found per level x restoration) and
runs/robustness.png (success rate vs degradation level, restoration on / off).
"""
import collections
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JOB = re.compile(r"agent_sim(\d)(_norestore)?_s(\d)\.txt$")
LINE = re.compile(r"(OK|FAIL) +decisions (\d+).*segments (\d+).*screen not found (\d+)/(\d+)")
COLORS = {"on": "#2a78d6", "off": "#eb6834"}  # dataviz palette slots 1-2, validated (light surface)


def load(folder=Path("runs/jobs")):
    """{(level, "on"|"off"): [(ok, decisions, segments, missed fraction)]}"""
    out = collections.defaultdict(list)
    for p in sorted(folder.glob("agent_sim*_s*.txt")):
        m, r = JOB.search(p.name), LINE.search(p.read_text())
        if m and r:
            out[int(m[1]), "off" if m[2] else "on"].append(
                (r[1] == "OK", int(r[2]), int(r[3]), int(r[4]) / int(r[5])))
    return out


def main():
    runs = load()
    rows = ["level  restore  success  decisions(ok)  segments  screen not found"]
    for (lvl, rest), v in sorted(runs.items()):
        ok = [d for o, d, _, _ in v if o]
        rows.append(f"{lvl:5}  {rest:7}  {len(ok)}/{len(v)}      {sum(ok) / len(ok) if ok else float('nan'):9.0f}"
                    f"  {sum(s for *_, s, _ in v) / len(v):8.1f}  {100 * sum(f for *_, f in v) / len(v):10.1f}%")
    Path("runs/robustness.txt").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for rest, marker, label in (("on", "o", "restoration on"), ("off", "s", "restoration off")):
        pts = sorted((lvl, 100 * sum(o for o, *_ in v) / len(v), len(v)) for (lvl, r), v in runs.items() if r == rest)
        xs, ys, ns = zip(*pts)
        ax.plot(xs, ys, color=COLORS[rest], lw=2, marker=marker, ms=7, label=label,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        y1 = dict(zip(xs, ys))[1]  # label at level 1, where the lines part; from level 2 both sit at 0%
        ax.annotate(f"{label}: {y1:.0f}%", (1, y1), xytext=(10, 4), textcoords="offset points",
                    color="#3d3d3a", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xlabel("degradation level (evaluation/degrade.py)", color="#3d3d3a")
    ax.set_ylabel("episodes reaching Viridian (%)", color="#3d3d3a")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ad")
    ax.tick_params(colors="#6b6a64")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Agent success under simulated capture (3 seeds at levels 0-2, 1 at 3-4)",
                 fontsize=9, color="#1f1f1d", loc="left")
    fig.tight_layout()
    fig.savefig("runs/robustness.png")


if __name__ == "__main__":
    main()
