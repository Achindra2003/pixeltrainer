"""Emulator wrapper checks against RAM truth. Needs the ROM and start states (skipped otherwise)."""
from pathlib import Path

import numpy as np
import pytest

from evaluation import truth

ROM, STATE = Path("roms/pokered.gb"), Path("states/pallet_start.state")
pytestmark = pytest.mark.skipif(not (ROM.exists() and STATE.exists()), reason="needs ROM and pallet_start state")


@pytest.fixture
def emu():
    from pixeltrainer.emu import Emu
    e = Emu(str(ROM), str(STATE), seed=0)
    yield e
    e.close()


def xy(emu):
    t = truth.read(emu._pyboy)
    return t["map_id"], t["x"], t["y"]


def test_frame_is_shade_indices(emu):
    for d in ["right", "right", "left", "up"]:  # last one enters the lab: covers a fade
        assert set(np.unique(emu.step(d))) <= {0, 1, 2, 3}


def test_walk_is_exactly_one_step(emu):
    assert xy(emu) == (0, 12, 12)
    assert emu.walk("down") is False  # fence below the lab door
    emu.wait_until_stable()
    assert xy(emu) == (0, 12, 12)
    for x in (13, 14, 15):  # repeated steps, first one also turns
        emu.step("right")
        assert xy(emu) == (0, x, 12)
    emu.step("left")  # turn + step
    assert xy(emu) == (0, 14, 12)


def test_seed_determinism():
    from pixeltrainer.emu import Emu

    def run(seed):
        e = Emu(str(ROM), str(STATE), seed=seed)
        frames = [e.step(d) for d in ["right", "right", "left"]]
        rng = e._pyboy.memory[e._pyboy.symbol_lookup("hRandomAdd")[1]]
        n = e.frames
        e.close()
        return frames, rng, n

    a, b, c = run(1), run(1), run(2)
    assert a[2] == b[2] and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    assert (a[1], a[2]) != (c[1], c[2])  # different idle length and RNG state
