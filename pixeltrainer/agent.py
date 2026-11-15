"""The agent: frames in, buttons out (PLAN.md §4.9, task 4.4). Overworld only; battles use a placeholder.

State machine on the screen state (perception/screen_state.cnn):
- overworld: tile lookup -> world map update -> stop if the landmark is seen -> explore toward the goal
  direction; the first step of the plan is the action. A door entered by accident is left at once (and
  is labelled "door" in the map, so the planner avoids it). With no frontier left (inside a building),
  walk back to the arrival cell and out.
- battle: press A (FIGHT, then the first move, then through the text). Placeholder until 5.7.
- dialogue: A. menu: B (the party menu opened by A-spam in battle, the start menu).
- transition: wait.
Watchdog: if the overworld position or the screen does not change for STUCK decisions, take a random
action and count it.

Usage: python -m pixeltrainer.agent [--state states/pallet_start.state] [--seed 0] [--window]
"""
import argparse
from collections import Counter

import numpy as np

from pixeltrainer.emu import DIRECTIONS, Emu
from pixeltrainer.mapping.worldmap import WorldMap
from pixeltrainer.perception import landmark, screen_state, tiles
from pixeltrainer.planning.astar import astar
from pixeltrainer.planning.explore import explore

STUCK = 8
BACK = {"up": "down", "down": "up", "left": "right", "right": "left"}


class Agent:
    def __init__(self, goal="north", target="poke_sign", seed=0):
        self.goal, self.target = goal, target
        self.rng = np.random.default_rng(seed)
        self.map = WorldMap()
        self.last_action = None
        self.same, self.prev_key = 0, None
        self.escapes = Counter()
        self.plan = None  # (path, target cell) of the last overworld decision, for the live view
        self.done = False
        self.n_doors, self.leaving = 0, False

    def act(self, frame, t=0):
        """One decision: the button to press for this stable frame (None = wait)."""
        state = screen_state.cnn(frame)
        key = (state, frame.tobytes()) if state != "overworld" else None
        direction = self.last_action if self.last_action in DIRECTIONS else None
        if state == "overworld":
            labels, conf = tiles.lookup(frame)
            _, x, y = self.map.update(frame, state, direction, labels, conf, t)
            key = (state, self.map.seg, x, y)
            if landmark.seen(frame, self.target):
                self.done = True
                return None
            action = self._overworld((x, y))
        else:
            self.map.update(frame, state, direction)  # the map remembers the step that led into a door
            action = {"battle": "a", "dialogue": "a", "menu": "b", "transition": None}[state]
        self.same = self.same + 1 if key == self.prev_key else 0
        self.prev_key = key
        if self.same >= STUCK:
            action = str(self.rng.choice(DIRECTIONS + ("a", "b")))
            self.escapes[state] += 1
            self.same = 0
        self.last_action = action
        return action

    def _overworld(self, pos):
        if len(self.map.doors) > self.n_doors:  # just went through a door (or a warp)
            self.n_doors = len(self.map.doors)
            if self.leaving:
                self.leaving = False
            elif self.map.last_dir:  # walked in by accident: walk straight back out
                self.leaving = True
                return BACK[self.map.last_dir]
        path, target = explore(self.map.label, pos, self.goal)
        if not path:
            # Nothing left to explore here (a building): back to where we arrived, then out.
            # shortcut: Gen 1 indoor exits are mats left by walking south; generalise for other exits.
            arrivals = [b[1:] for _, b in self.map.doors if b[0] == self.map.seg]
            if arrivals and pos == arrivals[-1]:
                self.leaving = True
                return "down"
            path, target = (astar(self.map.label, pos, {arrivals[-1]}), arrivals[-1]) if arrivals else (None, None)
        self.plan = (path, target)
        return path[0] if path else str(self.rng.choice(DIRECTIONS))


def run(emu, agent, max_decisions=1500, on_decision=None):
    """Play until the agent declares success or the decision budget runs out. Returns decisions used."""
    frame = emu.frame()
    for n in range(max_decisions):
        action = agent.act(frame, n)
        if on_decision:
            on_decision(n, frame, action)
        if agent.done:
            return n
        if action is None:
            emu.wait_until_stable()
            frame = emu.frame()
        else:
            frame = emu.step(action)
    return max_decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/pallet_start.state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", action="store_true")
    args = ap.parse_args()
    emu = Emu(state=args.state, headless=not args.window, seed=args.seed)
    emu.reset()
    agent = Agent(seed=args.seed)
    n = run(emu, agent)
    print(f"{'landmark seen' if agent.done else 'gave up'} after {n} decisions; "
          f"{agent.map.n_segments} map segments; watchdog escapes {dict(agent.escapes)}")
    emu.close()


if __name__ == "__main__":
    main()
