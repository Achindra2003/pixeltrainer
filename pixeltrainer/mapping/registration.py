"""Frame-to-frame shift by phase correlation (PLAN.md §4.5, task 3.1).

Measured in Phase 0 (PLAN.md §3.2): one step scrolls exactly 16 px on one axis; cv2.phaseCorrelate
gives +-16.0 +- 0.1 px with response 0.56-0.81; a blocked step gives shift 0 with response ~1.0;
door fades drop the response to <= 0.07. Content moves opposite to the walk: walking right gives dx -16.
A ledge jump moves two cells (32 px). Only the player is masked: zeroed NPC boxes sit at the same
place in both frames and pull the peak to zero shift (eval: 98.2% vs 99.9% of steps, task 3.1).
Grass and path textures repeat every 16 px, so phase correlation alone sometimes reads a real step
as 0. The agent knows the direction it pressed, so step() scores only the shifts that direction allows
(0, 1 or 2 cells) by mean absolute difference over the overlap (correlation-style matching).
"""
import cv2
import numpy as np

from pixeltrainer.perception import grid

TRANSITION_RESPONSE = 0.15  # below this the two frames do not show the same place
SNAP_TOL = 3.0  # px
MAX_MAD = 0.25  # mean |difference| in shade units above which no candidate fits: a transition
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _prep(frame, mask_cells):
    f = frame.astype(np.float32)
    f -= f.mean()
    for r, c in mask_cells:
        f[max(0, 16 * r - 4):16 * r + 12, 16 * c:16 * c + 16] = 0  # sprites sit 4 px above their cell
    return f


def shift(prev, cur, mask_cells=(grid.PLAYER,)):
    """(dx, dy, response) of cur relative to prev, in pixels. Cells in mask_cells (the player, NPCs)
    are blanked in both frames, since sprites move independently of the background."""
    (dx, dy), response = cv2.phaseCorrelate(_prep(prev, mask_cells), _prep(cur, mask_cells))
    return dx, dy, response


def mad(prev, cur, dcol, drow, mask_cells=(grid.PLAYER,)):
    """Mean |cur - prev| over the overlap, assuming the player moved (dcol, drow) cells (content moves
    the other way), ignoring the masked cells (in cur's position)."""
    dx, dy = 16 * dcol, 16 * drow
    a = prev[max(dy, 0):144 + min(dy, 0), max(dx, 0):160 + min(dx, 0)].astype(np.float32)
    b = cur[max(-dy, 0):144 + min(-dy, 0), max(-dx, 0):160 + min(-dx, 0)].astype(np.float32)
    keep = np.ones(cur.shape, bool)
    for r, c in mask_cells:
        keep[max(0, 16 * r - 4):16 * r + 12, 16 * c:16 * c + 16] = False
    keep = keep[max(-dy, 0):144 + min(-dy, 0), max(-dx, 0):160 + min(-dx, 0)]
    return float(np.abs(a - b)[keep].mean())


def step(prev, cur, direction=None, mask_cells=(grid.PLAYER,)):
    """Player move in cells as (dcol, drow): (0, 0) blocked, (1, 0) one step right, (0, 2) a ledge jump south.
    With the pressed direction, picks the best of 0/1/2 cells that way; without it, uses phase correlation.
    Returns None for a transition (fade, door, battle) or when nothing fits."""
    if direction:
        ux, uy = DELTA[direction]
        cost, k = min((mad(prev, cur, k * ux, k * uy, mask_cells), k) for k in (0, 1, 2))
        return (k * ux, k * uy) if cost <= MAX_MAD else None
    dx, dy, response = shift(prev, cur, mask_cells)
    if response < TRANSITION_RESPONSE:
        return None
    snapped = []
    for v in (dx, dy):
        k = round(v / 16)
        if abs(v - 16 * k) > SNAP_TOL or abs(k) > 2:
            return None
        snapped.append(-k)
    if snapped[0] and snapped[1]:
        return None  # a step moves on one axis only
    return tuple(snapped)


def is_transition(prev, cur, mask_cells=(grid.PLAYER,)):
    return shift(prev, cur, mask_cells)[2] < TRANSITION_RESPONSE
