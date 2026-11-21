"""Run the agent once (task 7.2).

Usage: python -m pixeltrainer.run [--state states/pallet_start.state] [--goal north] [--policy type] [--seed 0]
                                  [--max 1500] [--headless] [--record runs/demo.mp4]
Without --headless the PyBoy window shows the game at real speed; --record writes the demo video
(pixeltrainer/viz.py). The agent stops when it sees the landmark (the Viridian City Poke Mart sign).
"""
import argparse

from pixeltrainer.agent import Agent, run
from pixeltrainer.emu import Emu
from pixeltrainer.planning.explore import COMPASS
from pixeltrainer.policy.battle_policy import POLICIES
from pixeltrainer.viz import Recorder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/pallet_start.state")
    ap.add_argument("--goal", default="north", choices=tuple(COMPASS))
    ap.add_argument("--policy", default="type", choices=POLICIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=1500, help="decision budget")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", help="mp4 path for the demo video")
    args = ap.parse_args(argv)

    emu = Emu(state=args.state, headless=args.headless, seed=args.seed)
    agent = Agent(goal=args.goal, seed=args.seed, policy=args.policy)
    rec = Recorder(args.record, agent) if args.record else None
    try:
        n = run(emu, agent, args.max, rec)
    finally:
        if rec:
            rec.close()
        emu.close()
    print(f"{'landmark seen' if agent.done else 'budget used up'} after {n} decisions, "
          f"{agent.map.n_segments} map segments, watchdog {dict(agent.escapes)}")
    return agent.done


if __name__ == "__main__":
    main()
