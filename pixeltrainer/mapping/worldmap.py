"""World map from registered frames (PLAN.md §4.5, tasks 3.2 and 3.2b).

The map is a set of segments, each with its own cell grid (x, y), y growing south. A segment starts at
(0, 0) where it was first seen. Outdoor towns and routes join by seamless scrolling, so they stay one
segment; a door fade starts or re-enters another segment, and the door cells on both sides are linked.

update() takes every frame with its screen state:
- overworld after overworld: registration.step with the pressed direction moves the position. A
  blocked move marks the target cell as a wall.
- overworld after a gap (battle, dialogue, menu, fade): if the frame still registers against the last
  overworld frame, the player is where they were. Otherwise it is a new place: relocalise against
  every segment by exact cell keys, or start a new segment.
Observations: cells (x, y) -> (label, confidence, frame index). A newer label replaces an older one
only when it is strictly more confident or the older one is not confident (< CONFIDENT).
Mosaic: per cell, a count of each of the 4 shades at each pixel, so the median over all frames that
saw it is exact and cheap (task 3.2b). Moving NPCs drop out of the median.
"""
import numpy as np

from pixeltrainer.mapping import registration
from pixeltrainer.perception import grid

CONFIDENT = 0.8
MATCH = 0.8  # share of known overlapping cells whose key must match to relocalise
MIN_OVERLAP = 30  # known overlapping cells needed to relocalise
ANCHORS = 5


def sprite_mask(mask_cells):
    """(144, 160) bool, True where a sprite may be drawn (16x16 box, 4 px above its cell)."""
    m = np.zeros((144, 160), bool)
    for r, c in mask_cells:
        m[max(0, 16 * r - 4):16 * r + 12, 16 * c:16 * c + 16] = True
    return m


class WorldMap:
    def __init__(self):
        self.n_segments = 0
        self.seg, self.pos = None, None
        self.labels = {}  # (seg, x, y) -> (label, confidence, t)
        self.counts = {}  # (seg, x, y) -> (16, 16, 4) uint16 shade counts
        self.keys = {}  # (seg, x, y) -> cell key last seen clean
        self.where = {}  # cell key -> set of (seg, x, y) where it was seen
        self.doors = []  # ((seg, x, y), (seg, x, y)): door cell left -> cell where the player arrived
        self.last = None  # last overworld frame
        self.gap = False  # a non-overworld frame came after self.last
        self.last_dir = None

    # --- position ---------------------------------------------------------------------------
    def update(self, frame, state, direction=None, labels=None, conf=None, t=0, mask_cells=(grid.PLAYER,)):
        """Feed one stable frame. labels/conf: optional (9, 10) tile classes and confidences.
        Returns (seg, x, y), or None while not in the overworld."""
        if direction:
            self.last_dir = direction
        if state != "overworld":
            self.gap = self.gap or self.last is not None
            return None
        if self.last is None:
            self._place(frame)
        elif not self.gap:
            move = registration.step(self.last, frame, direction, mask_cells)
            if move is None:
                self._jump(frame)
            else:
                self._move(move, direction)
        else:
            move = registration.step(self.last, frame, None, mask_cells)
            if move is None:
                self._jump(frame)
            else:
                self._move(move, None)
        self.last, self.gap = frame, False
        self._observe(frame, labels, conf, t, mask_cells)
        return (self.seg, *self.pos)

    def _move(self, move, direction):
        if move == (0, 0) and direction:
            ux, uy = registration.DELTA[direction]
            self.labels[(self.seg, self.pos[0] + ux, self.pos[1] + uy)] = ("wall", 1.0, -1)
            # shortcut: a cell an NPC blocked stays a wall; un-mark it once perception/npc.py exists.
        self.pos = (self.pos[0] + move[0], self.pos[1] + move[1])

    def _jump(self, frame):
        """New place (door or warp): link the door cell in front of the old position to the new one."""
        ux, uy = registration.DELTA.get(self.last_dir, (0, 0))
        door = (self.seg, self.pos[0] + ux, self.pos[1] + uy)
        self._place(frame)
        self.doors.append((door, (self.seg, *self.pos)))

    def _place(self, frame):
        found = self.relocalise(frame)
        if found:
            self.seg, *pos = found
            self.pos = tuple(pos)
        else:
            self.seg, self.pos = self.n_segments, (0, 0)
            self.n_segments += 1

    def relocalise(self, frame, mask_cells=(grid.PLAYER,)):
        """(seg, x, y) where this frame's cell keys match the stored ones best, or None."""
        ks, clean = self._clean_keys(frame, mask_cells)
        cells = [(r, c) for r in range(grid.ROWS) for c in range(grid.COLS) if clean[r, c]]
        if not cells:
            return None
        # Anchor on the ANCHORS frame cells whose keys are rarest in the map (but seen); each place where
        # one was seen gives a candidate position. Several anchors, since animated or occluded cells change.
        seen = sorted((rc for rc in cells if ks[rc] in self.where), key=lambda rc: len(self.where[ks[rc]]))
        candidates = {(seg, x - (c0 - grid.PLAYER[1]), y - (r0 - grid.PLAYER[0]))
                      for r0, c0 in seen[:ANCHORS] for seg, x, y in self.where[ks[r0, c0]]}
        best, best_score = None, 0.0
        for seg, px, py in candidates:
            hit = known = 0
            for r, c in cells:
                k = self.keys.get((seg, px + c - grid.PLAYER[1], py + r - grid.PLAYER[0]))
                if k is not None:
                    known += 1
                    hit += k == ks[r, c]
            if known >= MIN_OVERLAP and hit / known >= MATCH and hit / known > best_score:
                best, best_score = (seg, px, py), hit / known
        return best

    # --- observations -----------------------------------------------------------------------
    @staticmethod
    def _clean_keys(frame, mask_cells):
        masked = sprite_mask(mask_cells).reshape(grid.ROWS, 16, grid.COLS, 16).any(axis=(1, 3))
        return grid.keys(frame), ~masked

    def _observe(self, frame, labels, conf, t, mask_cells):
        ks, clean = self._clean_keys(frame, mask_cells)
        onehot = np.eye(4, dtype=np.uint16)[frame]  # (144, 160, 4)
        onehot[sprite_mask(mask_cells)] = 0
        for r in range(grid.ROWS):
            for c in range(grid.COLS):
                cell = (self.seg, self.pos[0] + c - grid.PLAYER[1], self.pos[1] + r - grid.PLAYER[0])
                block = onehot[16 * r:16 * r + 16, 16 * c:16 * c + 16]
                if cell in self.counts:
                    self.counts[cell] += block
                else:
                    self.counts[cell] = block.copy()
                if not clean[r, c]:
                    continue
                self.keys[cell] = ks[r, c]
                self.where.setdefault(ks[r, c], set()).add(cell)
                if labels is not None:
                    self.observe(cell, labels[r][c], 1.0 if conf is None else float(conf[r][c]), t)

    def observe(self, cell, label, confidence, t=0):
        """Merge rule: replace when strictly more confident, or when the stored label is not confident."""
        old = self.labels.get(cell)
        if old is None or confidence > old[1] or old[1] < CONFIDENT:
            self.labels[cell] = (label, confidence, t)

    def label(self, xy, seg=None):
        """Tile class at (x, y) in a segment (default: the current one), or None; for planning/astar."""
        v = self.labels.get((self.seg if seg is None else seg, *xy))
        return v[0] if v else None

    # --- rendering --------------------------------------------------------------------------
    def bounds(self, seg):
        xs = [x for s, x, _ in self.counts if s == seg]
        ys = [y for s, _, y in self.counts if s == seg]
        return min(xs), min(ys), max(xs), max(ys)

    def mosaic(self, seg):
        """Grey image of the segment (shade s drawn as 255 - 85 s), median per pixel; unseen = 128."""
        x0, y0, x1, y1 = self.bounds(seg)
        img = np.full((16 * (y1 - y0 + 1), 16 * (x1 - x0 + 1)), 128, np.uint8)
        for (s, x, y), n in self.counts.items():
            total = n.sum(-1)
            if s != seg or not total.any():
                continue
            median = (2 * n.cumsum(-1) >= total[..., None]).argmax(-1)
            block = img[16 * (y - y0):16 * (y - y0 + 1), 16 * (x - x0):16 * (x - x0 + 1)]
            block[total > 0] = (255 - 85 * median)[total > 0].astype(np.uint8)
        return img

    def render(self, seg=None):
        """Colour debug image: mosaic, walls tinted red, door cells green, the player boxed in blue."""
        seg = self.seg if seg is None else seg
        x0, y0, _, _ = self.bounds(seg)
        img = np.repeat(self.mosaic(seg)[..., None], 3, -1).astype(np.int16)

        def box(x, y):
            return img[16 * (y - y0):16 * (y - y0 + 1), 16 * (x - x0):16 * (x - x0 + 1)]

        for (s, x, y), (lab, _, _) in self.labels.items():
            if s == seg and lab == "wall" and (s, x, y) in self.counts:
                box(x, y)[..., 2] += 80  # BGR: red
        for a, b in self.doors:
            for s, x, y in (a, b):
                if s == seg and (s, x, y) in self.counts:
                    box(x, y)[..., 1] += 80
        if seg == self.seg:
            b = box(*self.pos)
            b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = (255, 0, 0)
        return img.clip(0, 255).astype(np.uint8)
