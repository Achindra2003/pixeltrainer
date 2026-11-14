"""4-connected grid A* with terrain costs and one-way ledges (PLAN.md §4.6).

Cells are (x, y) with y growing south. `label(cell)` returns a tile class from perception/tiles.py
(walkable, wall, grass, water, door, ledge_s, ledge_e, ledge_w, sign, npc) or None if unknown.
"""
import heapq

STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
LEDGE = {"ledge_s": "down", "ledge_e": "right", "ledge_w": "left"}  # ledge class -> jump direction
COST = {"walkable": 1, "grass": 3, None: 2}  # missing classes are impassable; doors only as a goal


def astar(label, start, goals, cost=COST, penalty=None, max_nodes=20000):
    """Cheapest path from start to any goal cell. Returns a list of directions, [] if start is a goal,
    or None if no goal is reachable. `penalty(cell)` adds extra cost, e.g. NPC sight lines.
    A ledge is an edge from the cell before it to the cell after it, only in its jump direction."""
    goals = set(goals)
    if not goals:
        return None

    def h(c):
        return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

    def enter(c):
        if c in goals and label(c) == "door":
            return 1
        k = cost.get(label(c))
        return None if k is None else k + (penalty(c) if penalty else 0)

    best, prev = {start: 0}, {start: None}
    heap = [(h(start), 0, start)]
    while heap and len(prev) <= max_nodes:
        _, g, c = heapq.heappop(heap)
        if g > best[c]:
            continue
        if c in goals:
            path = []
            while prev[c]:
                c, d = prev[c]
                path.append(d)
            return path[::-1]
        for d, (dx, dy) in STEP.items():
            n = (c[0] + dx, c[1] + dy)
            jump = LEDGE.get(label(n))
            if jump == d:
                n, k = (n[0] + dx, n[1] + dy), enter((n[0] + dx, n[1] + dy))
                k = None if k is None else k + 1  # the jump covers two cells
            elif jump:
                continue  # a ledge is a wall from every other side
            else:
                k = enter(n)
            if k is not None and g + k < best.get(n, float("inf")):
                best[n], prev[n] = g + k, (c, d)
                heapq.heappush(heap, (g + k + h(n), g + k, n))
    return None
