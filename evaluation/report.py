"""Evaluation report: PLAN.md §5 tables from one run_eval folder plus every component evaluation (task 7.4).

Usage: python evaluation/report.py runs/eval/<run>
Writes <run>/report/report.md and copies the plots next to it.
From the run_eval traces (<run>/*.jsonl), per policy: task success, decisions (successful episodes),
decisions per second, battle win rate, odometry drift and battle-state agreement.
  drift: consecutive overworld decisions in the same map segment and the same RAM map whose
  (RAM x - map x, RAM y - map y) offset changes, per 100 such steps. 0 means the map never slips.
  battle agreement: share of decisions where the screen-state reading "battle" matches RAM wIsInBattle != 0,
  transition frames left out (the fades into and out of a battle are transitions while RAM is already in it).
The component evaluations (screen state, tiles/map, registration, NPC, battle, OCR, capture chain,
robustness sweep) are the files their own scripts wrote under runs/; they are included verbatim.
"""
import collections
import json
import shutil
import sys
from pathlib import Path

COMPONENTS = [  # (heading, PLAN.md §5 target, file written by its evaluation script)
    ("Screen state, CNN (eval_screen_state.py)", ">= 99%", "runs/screen_state_cnn.txt"),
    ("Screen state, heuristic baseline", "", "runs/screen_state_heuristic.txt"),
    ("Position and map, lookup tiles (eval_worldmap.py)", "0 drift on clean runs", "runs/worldmap_eval.txt"),
    ("Position and map, CNN tiles", "", "runs/worldmap_eval_cnn.txt"),
    ("Registration (eval_registration.py)", "", "runs/registration_eval.txt"),
    ("NPC detection (eval_npc.py)", "P/R >= 95%", "runs/npc_eval.txt"),
    ("Battle perception: species, HP (eval_battle.py)", "species >= 95%, HP MAE <= 2%", "runs/battle_eval.txt"),
    ("OCR (eval_ocr.py)", ">= 99.5%", "runs/ocr_eval.txt"),
    ("Screen localisation (eval_localise.py)", "", "runs/localise_eval.txt"),
    ("Capture chain shade accuracy (eval_capture.py)", "", "runs/capture_eval.txt"),
    ("Restoration methods (eval_restore.py)", "", "runs/restore_eval.txt"),
    ("Registration under degradation (eval_register_degraded.py)", "", "runs/register_degraded_eval.txt"),
    ("Robustness sweep (plot_robustness.py)", "", "runs/robustness.txt"),
]
PLOTS = ["runs/robustness.png", "runs/npc_templates.png", "runs/degrade/contact.png"]


def load(run):
    eps = []
    for p in sorted(Path(run).glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        eps.append((rows[0], rows[1:]))
    return eps


def drift(trace):
    """(offset changes, comparable overworld steps)."""
    changes = steps = 0
    prev = None
    for r in trace:
        if r["state"] != "overworld" or r["pos"] is None:
            prev = None
            continue
        key = (r["seg"], r["map"])
        off = (r["x"] - r["pos"][0], r["y"] - r["pos"][1])
        if prev and prev[0] == key:
            steps += 1
            changes += off != prev[1]
        prev = (key, off)
    return changes, steps


def agent_table(eps):
    by = collections.defaultdict(list)
    for s, t in eps:
        by[s["policy"]].append((s, t))
    lines = ["| policy | success | decisions (successes, mean) | decisions/s | battles | win rate | "
             "drift / 100 steps | battle agreement |", "|---|---|---|---|---|---|---|---|"]
    for policy, v in sorted(by.items()):
        ok = [s["decisions"] for s, _ in v if s["ok"]]
        battles = [o for s, _ in v for o, _ in s["battles"]]
        ch, st = map(sum, zip(*(drift(t) for _, t in v)))
        agree = [(r["state"] == "battle") == bool(r["in_battle"]) for _, t in v for r in t if r["state"] != "transition"]
        lines.append(f"| {policy} | {len(ok)}/{len(v)} | {sum(ok) / max(len(ok), 1):.0f} | "
                     f"{sum(s['decisions'] for s, _ in v) / sum(s['seconds'] for s, _ in v):.1f} | {len(battles)} | "
                     f"{battles.count('won') / max(len(battles), 1):.1%} | {100 * ch / max(st, 1):.2f} ({ch} in {st}) | "
                     f"{sum(agree) / len(agree):.2%} |")
    return lines


def main():
    run = Path(sys.argv[1])
    eps = load(run)
    out = run / "report"
    out.mkdir(exist_ok=True)
    s0 = eps[0][0]
    md = ["# PixelTrainer evaluation report", "",
          f"Agent episodes: `{run}`, {len(eps)} episodes, input {s0['input']}, Pallet Town -> Viridian City "
          "(success = landmark declared while RAM says Viridian).", "", *agent_table(eps), "",
          "Drift counts slips of the map position against RAM within one map segment and one RAM map "
          "(a segment can span several RAM maps joined without a warp).", ""]
    for heading, target, f in COMPONENTS:
        p = Path(f)
        md += [f"## {heading}", "", f"Target: {target}" if target else "", "",
               "```", p.read_text(encoding="utf-8", errors="replace").rstrip() if p.exists() else f"(missing: {f})",
               "```", ""]
    for f in PLOTS:
        if Path(f).exists():
            shutil.copy(f, out / Path(f).name)
            md += [f"![{Path(f).stem}]({Path(f).name})", ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(agent_table(eps)))


if __name__ == "__main__":
    main()
