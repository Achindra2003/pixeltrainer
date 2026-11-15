from pixeltrainer.planning.explore import explore, frontier, grow

# Partial map, y grows south; "?" unknown, "." walkable, "#" wall, "v" ledge (jump south), "P" player.
MAP = """
???????????
#?###v###?#
#.#.....#.#
#.#.###.#.#
#.......#.#
####P######
"""
LABEL = {".": "walkable", "#": "wall", "v": "ledge_s", "P": None}


def parse(text):
    grid = {}
    for y, row in enumerate(text.strip().splitlines()):
        for x, ch in enumerate(row):
            if ch != "?":
                grid[(x, y)] = LABEL[ch]
    return grid, next((x, y) for (x, y), v in grid.items() if v is None)


def test_grow_respects_walls_and_ledges():
    grid, start = parse(MAP)
    dist = grow(grid.get, start)
    assert (9, 2) not in dist  # walled-off column
    assert (5, 0) not in dist  # behind a south-facing ledge: cannot climb it
    assert dist[(1, 2)] == 6


def test_explore_goes_north_through_the_reachable_gap():
    grid, start = parse(MAP)
    front = frontier(grid.get, grow(grid.get, start))
    assert (5, 1) not in front and (1, 1) in front  # the ledge is known, the gap at (1, 1) is not
    assert (9, 1) not in front  # next to the walled-off column only
    path, target = explore(grid.get, start, "north")
    assert target == (1, 1)
    assert path == ["up", "left", "left", "left", "up", "up", "up"]


def test_no_frontier():
    grid = {(0, 0): None, (1, 0): "wall", (-1, 0): "wall", (0, 1): "wall", (0, -1): "wall"}
    assert explore(grid.get, (0, 0)) == (None, None)
