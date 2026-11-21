"""Seeded end-to-end runs of the agent, Pallet Town -> Viridian City (tasks 4.5 and 5.8).

Usage: python evaluation/run_agent.py [--seeds 0-9] [--max 1500] [--policy random|greedy|type|all]
                                     [--input clean|sim:<level>] [--no-restore]
The agent decides from pixels only; this evaluator reads RAM (wCurMap) to check the result.
Success = the agent declared the landmark seen while RAM says Viridian City (map 1).
Battles: each wild battle is scored from RAM when it ends: won (enemy HP 0), lost (own HP 0) or other
(fled), with the decisions it took.
sim:<level> (task 8.7): every frame the agent sees is degraded (evaluation/degrade.py, a new seed per
frame) and recovered by the capture chain localise -> restore -> rectify -> quantise; --no-restore skips
restore. If the screen is not found, the last corners found are reused (a fixed camera barely moves).
The emulator's own stability wait still reads clean frames: it stands in for "wait until nothing moves".
Writes runs/agent_eval_<policy>[_sim<level>[_norestore]].txt and, per seed,
runs/agent/<seed>[_sim...]/render_<seg>.png.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import truth  # noqa: E402
from evaluation.degrade import LEVELS, degrade  # noqa: E402
from pixeltrainer.capture.localise import localise  # noqa: E402
from pixeltrainer.capture.quantise import quantise  # noqa: E402
from pixeltrainer.capture.rectify import rectify  # noqa: E402
from pixeltrainer.capture.restore import restore  # noqa: E402
from pixeltrainer.agent import Agent, run  # noqa: E402
from pixeltrainer.emu import Emu  # noqa: E402
from pixeltrainer.policy.battle_policy import POLICIES  # noqa: E402

VIRIDIAN = 1


def sim_view(level, seed, use_restore=True):
    """frame -> degraded, captured and quantised frame; stats counts frames and screens not found."""
    state = {"k": 0, "corners": None, "missed": 0}

    def view(frame):
        img, _ = degrade(frame, LEVELS[level], seed=1_000_000 * (seed + 1) + state["k"])
        state["k"] += 1
        got = localise(img)
        if got is None:
            state["missed"] += 1
            got = state["corners"]
        if got is None:
            return frame * 0  # no screen ever found: a blank view
        state["corners"] = got
        return quantise(rectify(restore(img, got)[0] if use_restore else img, got))
    return view, state


def episode(seed, max_decisions, policy="type", battles=None, level=None, use_restore=True, tag=""):
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

    view, sim = sim_view(level, seed, use_restore) if level is not None else (None, None)
    n = run(emu, agent, max_decisions, log, view)
    t = truth.read(emu._pyboy)
    out = Path(f"runs/agent/{seed}{tag}")
    out.mkdir(parents=True, exist_ok=True)
    for s in range(agent.map.n_segments):
        cv2.imwrite(str(out / f"render_{s}.png"), agent.map.render(s))
    emu.close()
    ok = agent.done and t["map_id"] == VIRIDIAN
    return ok, (f"seed {seed}: {'OK  ' if ok else 'FAIL'} decisions {n}, frames {emu.frames}, "
                f"declared {agent.done}, RAM map {t['map_id']} at ({t['x']}, {t['y']}), "
                f"reached Viridian {bool(seen['max map'])}, battle decisions {seen['battle decisions']}, "
                f"segments {agent.map.n_segments}, watchdog {dict(agent.escapes)}" +
                (f", screen not found {sim['missed']}/{sim['k']}" if sim else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--policy", default="type", choices=POLICIES + ("all",))
    ap.add_argument("--input", default="clean", help="clean or sim:<level 0-4>")
    ap.add_argument("--no-restore", action="store_true")
    args = ap.parse_args()
    level = None if args.input == "clean" else int(args.input.removeprefix("sim:"))
    tag = "" if level is None else f"_sim{level}" + ("_norestore" if args.no_restore else "")
    a, _, b = args.seeds.partition("-")
    for policy in POLICIES if args.policy == "all" else (args.policy,):
        lines, wins, battles = [], 0, []
        for seed in range(int(a), int(b or a) + 1):
            ok, line = episode(seed, args.max, policy, battles, level, not args.no_restore, tag)
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
        Path(f"runs/agent_eval_{policy}{tag}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
