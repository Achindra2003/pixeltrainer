import numpy as np

from pixeltrainer.mapping import worldmap

RNG = np.random.default_rng(0)
# Two worlds (outdoor, indoor) built from one set of 50 random 16x16 cell textures, like a tileset.
# The view is a 144x160 window, the player at cell (4, 4).
TILES = RNG.integers(0, 4, (50, 16, 16), dtype=np.uint8)


def world():
    return TILES[RNG.integers(0, 50, (40, 40))].transpose(0, 2, 1, 3).reshape(640, 640)


OUT, IN = world(), world()


def view(world, col, row):
    return world[16 * row:16 * row + 144, 16 * col:16 * col + 160].copy()


def test_walk_block_door_and_return():
    wm = worldmap.WorldMap()
    assert wm.update(view(OUT, 10, 10), "overworld") == (0, 0, 0)
    assert wm.update(view(OUT, 11, 10), "overworld", "right") == (0, 1, 0)
    assert wm.update(view(OUT, 11, 10), "overworld", "up") == (0, 1, 0)  # blocked
    assert wm.label((1, -1)) == "wall"
    assert wm.update(view(OUT, 11, 10), "battle") is None
    assert wm.update(view(OUT, 11, 10), "overworld") == (0, 1, 0)  # back where it was after a battle
    wm.update(view(OUT, 11, 10), "transition", "down")
    assert wm.update(view(IN, 5, 5), "overworld") == (1, 0, 0)  # a door: new segment
    assert wm.doors == [((0, 1, 1), (1, 0, 0))]
    wm.update(view(IN, 5, 5), "transition", "down")
    assert wm.update(view(OUT, 11, 11), "overworld") == (0, 1, 1)  # relocalised outside, one cell south
    assert wm.n_segments == 2


def test_merge_rule():
    wm = worldmap.WorldMap()
    wm.observe((0, 0, 0), "grass", 0.5)
    wm.observe((0, 0, 0), "walkable", 0.4)  # old label not confident: replaced
    assert wm.label((0, 0), seg=0) == "walkable"
    wm.observe((0, 0, 0), "grass", 0.9)
    wm.observe((0, 0, 0), "wall", 0.85)  # old label confident and more so: kept
    assert wm.label((0, 0), seg=0) == "grass"


def test_mosaic_is_the_world():
    wm = worldmap.WorldMap()
    for col in range(10, 14):
        wm.update(view(OUT, col, 10), "overworld", "right")
    img = wm.mosaic(0)
    # Seen area: cols 10..13+9, rows 10..18. The player's sprite box is never pasted, so check row 10 only.
    assert np.array_equal(img[:16], 255 - 85 * OUT[160:176, 160:160 + 16 * 13])
