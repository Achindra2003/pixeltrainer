import numpy as np

from pixeltrainer.mapping.worldmap import WorldMap
from pixeltrainer.perception import grid, npc

T = dict(zip(npc.KEYS, npc._TEMPLATES.astype(np.uint8)))


def test_template_pasted_on_grass_is_found_with_facing():
    frame = np.kron(np.random.default_rng(0).integers(0, 2, (72, 80)), np.ones((2, 2), np.uint8)).astype(np.uint8)
    frame[28:44, 96:112] = T["47_up"]  # cell (2, 6): drawn 4 px above it
    found = npc.detect(frame)
    assert [(d["cell"], d["facing"]) for d in found] == [((2, 6), "up")]


def test_empty_frame_has_no_npcs():
    assert npc.detect(np.zeros((144, 160), np.uint8)) == []


def test_bump_into_npc_is_not_a_wall():
    frame = np.kron(np.random.default_rng(1).integers(0, 4, (9, 10)), np.ones((16, 16), np.uint8)).astype(np.uint8)
    m = WorldMap()
    m.update(frame, "overworld")
    above = (grid.PLAYER[0] - 1, grid.PLAYER[1])
    m.update(frame, "overworld", "up", npc_cells={above})
    assert m.label((m.pos[0], m.pos[1] - 1)) != "wall"
    m.update(frame, "overworld", "up")
    assert m.label((m.pos[0], m.pos[1] - 1)) == "wall"
