"""Frontier exploration toward a compass direction (PLAN.md §4.6, task 4.2).

Region growing from the player cell over known passable cells (ledges only in their jump direction)
gives the reachable set and the step distance to each cell. Frontier = an unknown cell next to the
reachable set: walking there reveals more map. Frontier cells next to only unreachable cells are never
considered. The target is the frontier cell with the best score = progress in the goal direction -
LAMBDA * distance, and the path to it comes from astar (which enters unknown cells optimistically).
"""
from collections import deque

from pixeltrainer.planning.astar import LEDGE, STEP, astar

PASSABLE = {"walkable", "grass"}
COMPASS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
LAMBDA = 0.5  # one step of detour is worth half a cell of progress


def grow(label, start):
    """Region growing from start: {cell: steps} for every reachable passable cell (start included,
    whatever its label: the player's own cell is hidden under the sprite)."""
    dist, todo = {start: 0}, deque([start])
    while todo:
        c = todo.popleft()
        for d, (dx, dy) in STEP.items():
            n = (c[0] + dx, c[1] + dy)
            jump = LEDGE.get(label(n))
            if jump == d:
                n = (n[0] + dx, n[1] + dy)
            elif jump:
                continue
            if label(n) in PASSABLE and n not in dist:
                dist[n] = dist[c] + 1
                todo.append(n)
    return dist


def frontier(label, dist):
    """{unknown cell: steps} for unknown cells one step from the reachable set."""
    out = {}
    for (x, y), k in dist.items():
        for dx, dy in STEP.values():
            n = (x + dx, y + dy)
            if label(n) is None and k + 1 < out.get(n, float("inf")):
                out[n] = k + 1
    return out


def explore(label, start, goal="north", lam=LAMBDA):
    """(path, target): directions toward the best frontier cell, or (None, None) if there is none."""
    front = frontier(label, grow(label, start))
    gx, gy = COMPASS[goal]
    best = max(front, default=None,
               key=lambda c: (c[0] - start[0]) * gx + (c[1] - start[1]) * gy - lam * front[c])
    if best is None:
        return None, None
    return astar(label, start, {best}), best
