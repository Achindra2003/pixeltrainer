"""Seeded end-to-end runs of the agent, Pallet Town -> Viridian City (tasks 4.5 and 5.8).

Usage: python evaluation/run_agent.py [--seeds 0-9] [--max 1500] [--policy random|greedy|type|all]
The agent decides from pixels only; this evaluator reads RAM (wCurMap) to check the result.
Success = the agent declared the landmark seen while RAM says Viridian City (map 1).
Battles: each wild battle is scored from RAM when it ends: won (enemy HP 0), lost (own HP 0) or other
(fled), with the decisions it took.
Writes runs/agent_eval_<policy>.txt and, per seed, runs/agent/<seed>/render_<seg>.png.
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
from pixeltrainer.policy.battle_policy import POLICIES  # noqa: E402

VIRIDIAN = 1


def episode(seed, max_decisions, policy="type", battles=None):
    emu = Emu(state="states/pallet_start.state", seed=seed)
    emu.reset()
    agent = Agent(seed=seed, policy=policy)
    seen, last = Counter(), {}

    def log(n, frame, action):
        t = truth.read(emu._pyboy)
        seen["battle decisions"] += bool(t["in_battle"])
        seen["max map"] = max(seen["max map"], t["map_id"] == VIRIDIAN)
        if t["in_battle"] and "enemy" in t:
            last.setdefault("start", n)
            last["t"] = t
        elif "t" in last:
            e, o = last["t"]["enemy"], last["t"]["own"]
            if battles is not None:
                battles.append(("won" if e["hp"] == 0 else "lost" if o["hp"] == 0 else "other", n - last["start"]))
            last.clear()

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
    ap.add_argument("--policy", default="type", choices=POLICIES + ("all",))
    args = ap.parse_args()
    a, _, b = args.seeds.partition("-")
    for policy in POLICIES if args.policy == "all" else (args.policy,):
        lines, wins, battles = [], 0, []
        for seed in range(int(a), int(b or a) + 1):
            ok, line = episode(seed, args.max, policy, battles)
            wins += ok
            lines.append(line)
            print(policy, line, flush=True)
        outcomes = Counter(o for o, _ in battles)
        won = [k for o, k in battles if o == "won"]
        lines.append(f"policy {policy}: success {wins}/{len(lines)}; battles {len(battles)}: won {outcomes['won']}, "
                     f"lost {outcomes['lost']}, other {outcomes['other']}; win rate "
                     f"{outcomes['won'] / max(len(battles), 1):.1%}; decisions per won battle "
                     f"{sum(won) / max(len(won), 1):.1f}")
        print(lines[-1], flush=True)
        Path(f"runs/agent_eval_{policy}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
