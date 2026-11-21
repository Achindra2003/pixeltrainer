"""Seeded evaluation episodes with per-decision logs (task 7.3).

Usage: python evaluation/run_eval.py [--seeds 0-19] [--policy all|random|greedy|type] [--max 1500]
                                     [--input clean|sim:<level>] [--no-restore] [--out runs/eval/<timestamp>]
One episode per (policy, seed): evaluation/run_agent.episode with a trace of every decision (agent state,
map segment and position next to RAM map, x, y, in_battle). Each episode writes <out>/<policy>_s<seed>.jsonl
only when it finishes: line 1 is the summary, then one line per decision. Finished episodes are skipped, so
pass the same --out to resume. Only the "north" goal is evaluated: its success test (Viridian City, RAM
map 1) is the one the landmark marks; the Viridian Forest goal was cut (task 6.4).
evaluation/report.py turns the folder into tables and plots.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.run_agent import episode  # noqa: E402
from pixeltrainer.policy.battle_policy import POLICIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--policy", default="all", choices=POLICIES + ("all",))
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--input", default="clean", help="clean or sim:<level 0-4>")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--out", default=f"runs/eval/{time.strftime('%Y%m%d-%H%M')}")
    args = ap.parse_args()
    level = None if args.input == "clean" else int(args.input.removeprefix("sim:"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a, _, b = args.seeds.partition("-")
    for policy in POLICIES if args.policy == "all" else (args.policy,):
        for seed in range(int(a), int(b or a) + 1):
            path = out / f"{policy}_s{seed}.jsonl"
            if path.exists():
                continue
            trace, battles = [], []
            start = time.perf_counter()
            ok, line = episode(seed, args.max, policy, battles, level, not args.no_restore,
                               f"_eval_{policy}", trace)
            summary = {"policy": policy, "seed": seed, "input": args.input, "restore": not args.no_restore,
                       "ok": ok, "decisions": len(trace), "seconds": round(time.perf_counter() - start, 1),
                       "battles": battles, "line": line}
            tmp = path.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(r) for r in [summary] + trace) + "\n")
            tmp.replace(path)
            print(policy, line, flush=True)


if __name__ == "__main__":
    main()
