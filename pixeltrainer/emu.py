"""PyBoy wrapper. The agent sees only frames (4 shade indices) and presses buttons.

Shade index: 0 = white, 1 = light grey, 2 = dark grey, 3 = black.
The evaluator may reach the emulator through `Emu._pyboy`; agent code must not (tests/test_ram_boundary.py).
"""
from pathlib import Path

import numpy as np
from pyboy import PyBoy

# PyBoy's default DMG palette has 4 grey levels (measured in Phase 0): 255, 153, 85, 0.
_SHADE = np.zeros(256, np.uint8)
_SHADE[[255, 153, 85, 0]] = [0, 1, 2, 3]

BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
DIRECTIONS = ("up", "down", "left", "right")

# The player's 16x16 cell (column 4, row 4), drawn 4 px above the cell. Masked when detecting scroll.
_PLAYER = (slice(60, 80), slice(64, 80))


def _changed(a, b):
    """Fraction of pixels that differ, ignoring the player sprite."""
    d = a != b
    d[_PLAYER] = False
    return d.mean()


class Emu:
    def __init__(self, rom="roms/pokered.gb", state=None, headless=True, seed=0):
        sym = Path(rom).with_suffix(".sym")
        self._pyboy = PyBoy(str(rom), window="null" if headless else "SDL2", scale=3,
                            symbols=str(sym) if sym.exists() else None)
        self._pyboy.set_emulation_speed(0 if headless else 1)
        self.state, self.seed = state, seed
        self.frames = 0  # frames ticked since reset
        if state:
            self.reset()

    def reset(self, seed=None):
        """Load the start state, then idle 0-300 frames. Gen 1 RNG runs every frame, so the idle
        length (from the seed) varies wild encounters and NPC moves between episodes."""
        if seed is not None:
            self.seed = seed
        with open(self.state, "rb") as f:
            self._pyboy.load_state(f)
        self.frames = 0
        self.tick(int(np.random.default_rng(self.seed).integers(0, 301)))
        self.wait_until_stable()
        return self.frame()

    def frame(self):
        """(144, 160) uint8 shade indices 0-3."""
        return _SHADE[self._pyboy.screen.ndarray[:, :, 0]]

    def tick(self, n=1, on_frame=None):
        for _ in range(n):
            self._pyboy.tick()
            self.frames += 1
            if on_frame:
                on_frame()

    def wait_until_stable(self, max_frames=600, on_frame=None):
        """Tick until no *new* screen (player sprite masked) appears for 8 frames and the screen is
        not blank. Animations (flowers, water, party-menu icons, the blinking text arrow) cycle
        through screens already seen, so they settle; printing text makes a new screen every few
        frames, so a half-printed line never counts as stable; door fades sit on a blank screen,
        hence the blank check. Returns frames ticked."""
        seen, quiet = set(), 0
        for n in range(1, max_frames + 1):
            self.tick(on_frame=on_frame)
            cur = self.frame()
            cur[_PLAYER] = 0
            key = cur.tobytes()
            quiet = quiet + 1 if key in seen and cur.std() > 0.3 else 0
            seen.add(key)
            if quiet >= 8:
                return n
        return max_frames

    def press(self, button, frames=4, on_frame=None):
        """Press and release a button (A/B/Start/Select, or a menu cursor move)."""
        self._pyboy.button_press(button)
        self.tick(frames, on_frame)
        self._pyboy.button_release(button)

    def walk(self, direction, max_hold=24, on_frame=None):
        """Hold a direction until the screen starts scrolling (one step begins), then release.
        Gen 1 always finishes a step once it starts, so this gives exactly one step.
        Returns True if the screen scrolled; False means blocked (or a non-overworld screen)."""
        start = self.frame()
        self._pyboy.button_press(direction)
        moved = False
        for _ in range(max_hold):
            self.tick(on_frame=on_frame)
            if _changed(start, self.frame()) > 0.03:
                moved = True
                break
        self._pyboy.button_release(direction)
        return moved

    def step(self, button, on_frame=None):
        """One agent decision: act, then wait for a stable frame. Returns the new frame."""
        if button in DIRECTIONS:
            self.walk(button, on_frame=on_frame)
        else:
            self.press(button, on_frame=on_frame)
        self.wait_until_stable(on_frame=on_frame)
        return self.frame()

    def close(self):
        self._pyboy.stop(save=False)
