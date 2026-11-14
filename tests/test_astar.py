from pixeltrainer.planning.astar import STEP, astar

LEGEND = {".": "walkable", "#": "wall", "g": "grass", "v": "ledge_s", "D": "door", "?": None}


def grid(rows):
    """Map from ASCII rows; cells outside the drawing are walls so the search stays bounded."""
    cells = {(x, y): LEGEND[ch] for y, row in enumerate(rows) for x, ch in enumerate(row)}
    return lambda c: cells.get(c, "wall")


def walk(start, path):
    x, y = start
    for d in path:
        x, y = x + STEP[d][0], y + STEP[d][1]
    return x, y


def test_wall_detour():
    m = grid(["....",
              ".##.",
              "...."])
    path = astar(m, (0, 1), [(3, 1)])
    assert len(path) == 5 and walk((0, 1), path) == (3, 1)


def test_grass_avoided_when_detour_is_cheaper():
    m = grid([".gg.",
              "...."])
    assert astar(m, (0, 0), [(3, 0)]) == ["down", "right", "right", "right", "up"]  # 5 < 3+3+1
    m = grid([".ggg.",
              "#####"])
    assert astar(m, (0, 0), [(4, 0)]) == ["right"] * 4  # grass is costly, not blocked



def test_ledge_is_one_way():
    m = grid(["...",
              "#v#",
              "..."])
    down = astar(m, (1, 0), [(1, 2)])
    assert down == ["down"]  # one jump lands two cells south
    assert astar(m, (1, 2), [(1, 0)]) is None  # cannot climb back up


def test_door_only_as_goal():
    m = grid(["D..",
              "..."])
    assert astar(m, (2, 0), [(0, 0)]) == ["left", "left"]
    m = grid([".D.",
              "..."])
    assert astar(m, (0, 0), [(2, 0)]) == ["down", "right", "right", "up"]  # never walks through a door


def test_unknown_is_passable_and_unreachable_is_none():
    assert astar(grid([".?."]), (0, 0), [(2, 0)]) == ["right", "right"]
    assert astar(grid([".#."]), (0, 0), [(2, 0)]) is None
    assert astar(grid(["."]), (0, 0), [(0, 0)]) == []
