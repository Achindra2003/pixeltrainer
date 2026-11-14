"""Play Pokémon Red by hand, with terminal commands for states, RAM truth and frames.

Usage: python tools/play.py [--state states/pallet_start.state]

Game window (PyBoy defaults): arrows = D-pad, A = a, B = s, Start = Enter, Select = Backspace,
Space = toggle fast-forward.

Type in this terminal while the game runs:
  save <name>    save state to states/<name>.state
  load <name>    load states/<name>.state
  truth          print RAM truth (evaluation/truth.py)
  frame [name]   save the current frame to runs/frames/<name>.png
  quit
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
from pyboy import PyBoy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import truth  # noqa: E402

STATES, FRAMES = Path("states"), Path("runs/frames")


def read_commands(q):
    for line in sys.stdin:
        q.put(line.strip())


def run(cmd, pyboy):
    name, _, arg = cmd.partition(" ")
    arg = arg.strip()
    if name == "save" and arg:
        STATES.mkdir(exist_ok=True)
        with open(STATES / f"{arg}.state", "wb") as f:
            pyboy.save_state(f)
        print(f"saved {STATES / f'{arg}.state'}")
    elif name == "load" and arg:
        path = STATES / f"{arg}.state"
        if not path.exists():
            print(f"no such state: {path}")
            return
        with open(path, "rb") as f:
            pyboy.load_state(f)
        print(f"loaded {path}")
    elif name == "truth":
        print(json.dumps(truth.read(pyboy), indent=1))
    elif name == "frame":
        FRAMES.mkdir(parents=True, exist_ok=True)
        out = FRAMES / f"{arg or time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(out), cv2.cvtColor(pyboy.screen.ndarray, cv2.COLOR_RGBA2BGR))
        print(f"saved {out}")
    elif cmd:
        print("commands: save <name> | load <name> | truth | frame [name] | quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokered.gb")
    ap.add_argument("--state")
    args = ap.parse_args()

    pyboy = PyBoy(args.rom, window="SDL2", scale=3, sound_emulated=False,
                  symbols=str(Path(args.rom).with_suffix(".sym")))
    if args.state:
        with open(args.state, "rb") as f:
            pyboy.load_state(f)

    q = queue.Queue()
    threading.Thread(target=read_commands, args=(q,), daemon=True).start()
    print(__doc__)
    running = True
    while running and pyboy.tick():
        while not q.empty():
            cmd = q.get()
            if cmd == "quit":
                running = False
                break
            run(cmd, pyboy)
    pyboy.stop(save=False)
    # The stdin reader thread deadlocks normal interpreter shutdown on Windows; exit hard.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
