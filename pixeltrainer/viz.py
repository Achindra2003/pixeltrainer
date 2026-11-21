"""Demo video frames (PLAN.md §4.10, task 7.1; the live window was cut 2026-11-20, so this only writes mp4).

compose(): left, the game at 3x (480x432); right, the agent's map of the current segment with its planned
path (yellow) and target (magenta), and a status line (decision, screen state, last battle reasoning).
Recorder: compose() each decision into an mp4 via cv2.VideoWriter.
"""
from pathlib import Path

import cv2
import numpy as np

from pixeltrainer.mapping.registration import DELTA

W, H = 480, 432  # each half
STATUS = 48  # px of text under the map


def _map_panel(agent):
    panel = np.full((H - STATUS, W, 3), 40, np.uint8)
    m = agent.map
    if m.seg is None or not any(s == m.seg for s, _, _ in m.counts):
        return panel
    img = m.render()
    x0, y0, _, _ = m.bounds(m.seg)
    path, target = agent.plan or (None, None)
    if path and m.pos:
        x, y = m.pos
        for d in path:
            x, y = x + DELTA[d][0], y + DELTA[d][1]
            cv2.circle(img, (16 * (x - x0) + 8, 16 * (y - y0) + 8), 3, (0, 220, 255), -1)
    if target:
        tx, ty = 16 * (target[0] - x0), 16 * (target[1] - y0)
        cv2.rectangle(img, (tx, ty), (tx + 15, ty + 15), (255, 0, 255), 2)
    k = min(W / img.shape[1], (H - STATUS) / img.shape[0])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST if k >= 1 else cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (H - STATUS - h) // 2, (W - w) // 2
    panel[top:top + h, left:left + w] = img
    return panel


def compose(frame, agent, n=0, action=None):
    """frame: (144, 160) shades 0-3 as the agent saw it -> (432, 960, 3) BGR."""
    game = cv2.cvtColor(np.kron(255 - 85 * frame.astype(np.int16), np.ones((3, 3), np.int16)).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)
    status = np.full((STATUS, W, 3), 25, np.uint8)
    lines = (f"decision {n}  state {agent.state}  action {action}  segment {agent.map.seg}",
             agent.reason if agent.state == "battle" else f"position {agent.map.pos}")
    for i, text in enumerate(lines):
        cv2.putText(status, str(text), (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                    cv2.LINE_AA)
    return np.hstack([game, np.vstack([_map_panel(agent), status])])


class Recorder:
    """on_decision callback for agent.run that writes compose() frames to an mp4."""

    def __init__(self, path, agent, fps=4):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent = agent
        self.out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))

    def __call__(self, n, frame, action):
        self.out.write(compose(frame, self.agent, n, action))

    def close(self):
        self.out.release()
