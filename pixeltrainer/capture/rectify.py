"""Undo the perspective view (PLAN.md §6A.1, task 8.3): the screen quad back to 160x144.

Corners follow evaluation/degrade.py and cv2.getPerspectiveTransform: the images of screen points
(0, 0), (160, 0), (160, 144), (0, 144); pixel centres are integers. Bilinear interpolation.
"""
import cv2
import numpy as np

CORNERS = np.float32([[0, 0], [160, 0], [160, 144], [0, 144]])


def rectify(image, corners):
    """Captured image (BGR or grey) -> (144, 160) float32 grey screen."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    H = cv2.getPerspectiveTransform(np.float32(corners), CORNERS)
    return cv2.warpPerspective(grey.astype(np.float32), H, (160, 144), flags=cv2.INTER_LINEAR)
