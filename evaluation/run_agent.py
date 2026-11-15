"""Seeded end-to-end runs of the agent, Pallet Town -> Viridian City (task 4.5).

Usage: python evaluation/run_agent.py [--seeds 0-9] [--max 1500]
The agent decides from pixels only; this evaluator reads RAM (wCurMap) to check the result.
Success = the agent declared the landmark seen while RAM says Viridian City (map 1).
Writes runs/agent_eval.txt and, per seed, runs/agent/<seed>/render_<seg>.png.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import truth  # noqa: E402
from pixeltrainer.agent import Agent, run  # noqa: E402
from pixeltrainer.emu import Emu  # noqa: E402

VIRIDIAN = 1


def episode(seed, max_decisions):
    emu = Emu(state="states/pallet_start.state", seed=seed)
    emu.reset()
    agent = Agent(seed=seed)
    seen = Counter()

    def log(n, frame, action):
        t = truth.read(emu._pyboy)
        seen["battle decisions"] += bool(t["in_battle"])
        seen["max map"] = max(seen["max map"], t["map_id"] == VIRIDIAN)

    n = run(emu, agent, max_decisions, log)
    t = truth.read(emu._pyboy)
    out = Path(f"runs/agent/{seed}")
    out.mkdir(parents=True, exist_ok=True)
    for s in range(agent.map.n_segments):
        cv2.imwrite(str(out / f"render_{s}.png"), agent.map.render(s))
    emu.close()
    ok = agent.done and t["map_id"] == VIRIDIAN
    return ok, (f"seed {seed}: {'OK  ' if ok else 'FAIL'} decisions {n}, frames {emu.frames}, "
                f"declared {agent.done}, RAM map {t['map_id']} at ({t['x']}, {t['y']}), "
                f"reached Viridian {bool(seen['max map'])}, battle decisions {seen['battle decisions']}, "
                f"segments {agent.map.n_segments}, watchdog {dict(agent.escapes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--max", type=int, default=1500)
    args = ap.parse_args()
    a, _, b = args.seeds.partition("-")
    lines, wins = [], 0
    for seed in range(int(a), int(b or a) + 1):
        ok, line = episode(seed, args.max)
        wins += ok
        lines.append(line)
        print(line, flush=True)
    lines.append(f"success {wins}/{len(lines)}")
    print(lines[-1])
    Path("runs/agent_eval.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
