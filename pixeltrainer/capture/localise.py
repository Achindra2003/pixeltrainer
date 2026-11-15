"""Find the Game Boy screen in a captured image (PLAN.md §6A.1, task 8.2).

1. Coarse: background subtraction. Gaussian blur; fit a plane per colour channel to a ring at the image
   border (the background: one grey under a linear light gradient and colour cast), refitted without
   outliers (clutter); screen = pixels further than BG_DIST from that model AND of low saturation (the
   screen is grey under a mild cast; clutter is coloured); opening (drops thin lines), closing (fills
   dark tiles and fence rows of the same grey as the background), largest contour, convex hull, polygon fit to
   4 corners (approxPolyDP); if the fit does not give 4, the minimum-area rectangle.
2. Fine: along each side, sample points and search the gradient magnitude (Sobel) along the side's
   normal for the edge (the first strong peak coming from outside: screen content has stronger edges
   than the screen border), to sub-pixel by a parabola through the peak; fit a line to them (Huber, so edge
   points pulled off by screen content that matches the background do not drag it) and intersect
   neighbouring lines for the corners.
3. Step model (last pass): along each normal, background level b = median of the samples 2-4 px outside,
   first screen pixel c = the sample 1 px inside; the edge is where the profile crosses (b + c) / 2
   (exact for any symmetric blur), to sub-pixel by linear interpolation. Samples with |c - b| < MIN_STEP
   are skipped (content the same grey as the background).
The edges found are where the outermost screen pixels end, i.e. screen coordinates -0.5 and 159.5 / 143.5
(pixel centres are integers). Corners are returned in the convention of evaluation/degrade.py and
cv2.getPerspectiveTransform: the images of screen points (0, 0), (160, 0), (160, 144), (0, 144).
"""
import cv2
import numpy as np

from pixeltrainer.capture.rectify import CORNERS

PASSES = (15, 4)  # search half-widths (px) of the two refinement passes: coarse corners can be cut off by
#                   dark tiles in a screen corner, the middle of each side is close
SAMPLES = 60  # per side
PEAK = 0.3  # an edge peak counts when it reaches this share of the strongest one on the profile
EDGE_QUAD = np.float32([[-0.5, -0.5], [159.5, -0.5], [159.5, 143.5], [-0.5, 143.5]])
MIN_STEP = 15  # grey levels
STEP_PASSES = 3


MAX_SAT = 60  # HSV saturation (0-255) above which a pixel is clutter, not screen


RING = 12  # px of image border used to model the background
BG_DIST = 20  # grey levels (max over channels) from the background model that count as screen


def background(image):
    """Per-channel plane a + b x + c y fitted to the border ring, twice (second fit without outliers)."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ring = np.zeros((h, w), bool)
    ring[:RING], ring[-RING:], ring[:, :RING], ring[:, -RING:] = True, True, True, True
    A = np.stack([np.ones(ring.sum()), xx[ring], yy[ring]], 1)
    img = image.reshape(h, w, -1).astype(np.float32)
    model = np.empty_like(img)
    for ch in range(img.shape[2]):
        v = img[..., ch][ring]
        keep = np.ones(len(v), bool)
        for _ in range(2):
            coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
            r = np.abs(A @ coef - v)
            keep = r < max(3 * np.median(r), 5)
        model[..., ch] = coef[0] + coef[1] * xx + coef[2] * yy
    return model


def coarse(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    diff = np.abs(image.reshape(*image.shape[:2], -1).astype(np.float32) - background(image)).max(-1)
    mask = np.where(diff > BG_DIST, 255, 0).astype(np.uint8)
    if image.ndim == 3:
        mask &= np.where(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1] < MAX_SAT, 255, 0).astype(np.uint8)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float32)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return order(quad)


def order(pts):
    """(tl, tr, br, bl) by angle around the centre, starting top-left."""
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # clockwise from left in image coords
    k = int(np.argmin(pts.sum(1)))
    return np.roll(pts, -k, axis=0)


def _step_points(grey, a, b):
    """Edge points by the step model, searching +-2 px around the line a-b (outside = negative offsets)."""
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    offs = np.arange(-4, 2.01, 0.25)
    pts = []
    for s in np.linspace(0.05, 0.95, 3 * SAMPLES):
        p = a + s * (b - a)
        xy = (p + offs[:, None] * n).astype(np.float32)
        if xy.min() < 0 or xy[:, 0].max() >= grey.shape[1] - 1 or xy[:, 1].max() >= grey.shape[0] - 1:
            continue
        v = cv2.remap(grey, xy[None, :, 0], xy[None, :, 1], cv2.INTER_LINEAR)[0]
        bg, c = np.median(v[offs <= -2]), v[offs == 1.0][0]
        if abs(c - bg) < MIN_STEP:
            continue
        half = (bg + c) / 2
        side = (v - half) * np.sign(c - bg) > 0  # True once past the edge
        k = np.nonzero(side & (offs > -2) & (offs <= 1))[0]
        if not len(k) or k[0] == 0:
            continue
        k = k[0]
        if v[k] == v[k - 1]:
            continue
        t = (half - v[k - 1]) / (v[k] - v[k - 1])
        pts.append(p + (offs[k - 1] + t * 0.25) * n)
    return np.array(pts, np.float32)


def _edge_points(mag, a, b, search):
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    offs = np.arange(-search, search + 1)
    pts = []
    for s in np.linspace(0.1, 0.9, SAMPLES):
        p = a + s * (b - a)
        xy = p + offs[:, None] * n
        x, y = xy[:, 0], xy[:, 1]
        if x.min() < 0 or y.min() < 0 or x.max() >= mag.shape[1] - 1 or y.max() >= mag.shape[0] - 1:
            continue
        v = cv2.remap(mag, x.astype(np.float32)[None], y.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        peaks = [i for i in range(1, len(v) - 1) if v[i] >= v[i - 1] and v[i] >= v[i + 1] and v[i] >= PEAK * v.max()]
        if not peaks:
            continue
        k = peaks[0]
        if 0 < k < len(v) - 1:
            den = v[k - 1] - 2 * v[k] + v[k + 1]
            k = k + (0.5 * (v[k - 1] - v[k + 1]) / den if den else 0)
        pts.append(p + (k - search) * n)
    return np.array(pts, np.float32)


def _intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    if abs(np.linalg.det(A)) < 1e-6:  # parallel sides: no quad
        raise ValueError("parallel lines")
    t = np.linalg.solve(A, [x2 - x1, y2 - y1])[0]
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def localise(image):
    """image: captured BGR or grey. Returns (4, 2) float32 corners (tl, tr, br, bl), or None."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    quad = coarse(image)
    if quad is None:
        return None
    try:
        return _refine(grey, quad)
    except ValueError:
        return None


def _refine(grey, quad):
    g = cv2.GaussianBlur(grey, (3, 3), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edges = quad
    for search in PASSES:
        lines = []
        for i in range(4):
            pts = _edge_points(mag, edges[i], edges[(i + 1) % 4], search)
            if len(pts) < 10:
                return None
            lines.append(cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        edges = np.stack([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    g = grey.astype(np.float32)
    for _ in range(STEP_PASSES):
        lines = []
        for i in range(4):
            pts = _step_points(g, edges[i], edges[(i + 1) % 4])
            if len(pts) < 10:
                break
            lines.append(cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        else:
            edges = np.stack([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    H = cv2.getPerspectiveTransform(EDGE_QUAD, edges)
    return cv2.perspectiveTransform(CORNERS[None], H)[0]
