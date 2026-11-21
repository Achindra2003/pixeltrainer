"""Find the Game Boy screen in a captured image (PLAN.md §6A.1, task 8.2).

1. Coarse: background subtraction. Fit a plane per colour channel to a ring at the image border (the
   background: one grey under a linear light gradient and colour cast), refitted without outliers
   (clutter). Screen = pixels whose mean absolute (channel-mean) difference from that model (5x5 Gaussian) is > BG_DIST,
   and whose colour, divided by the model, is grey (the cast cancels; clutter is coloured). Min then max filter (drops
   thin lines), max then min filter (fills tiles the same grey as the background), largest contour. Its sides: on each
   side of the centroid, the strongest Hough line of the outline, refitted to the outline points near it
   (a clutter box joined to the mask is shorter than a screen side). Fallback: hull, approxPolyDP to 4
   corners, else the minimum-area rectangle.
2. Gradient passes: along each side, sample points and search the gradient magnitude (Sobel) along the
   side's normal for the edge (the first strong peak coming from outside), to sub-pixel by a parabola;
   fit a line to them (Huber) and intersect neighbouring lines for the corners.
3. Edge-model passes (only while the fitted blur is <= MAX_MODEL_SIGMA): each normal profile is fitted
   with background, then screen pixels of the known pitch with free shades, all Gaussian-blurred
   (_model_points). Unbiased for symmetric blur; with more blur the pixels next to the edge are not
   resolved and the gradient passes do better.
The edges found are where the outermost screen pixels end, i.e. screen coordinates -0.5 and 159.5 / 143.5
(pixel centres are integers). Corners are returned in the convention of evaluation/degrade.py and
cv2.getPerspectiveTransform: the images of screen points (0, 0), (160, 0), (160, 144), (0, 144).
"""
import cv2
import numpy as np

from pixeltrainer.capture.rectify import CORNERS

PASSES = (6, 3)  # search half-widths (px) of the two refinement passes: wider ones catch clutter edges next to
#                  the screen; the Hough coarse quad is within a few px
SAMPLES = 60  # per side
PEAK = 0.3  # an edge peak counts when it reaches this share of the strongest one on the profile
EDGE_QUAD = np.float32([[-0.5, -0.5], [159.5, -0.5], [159.5, 143.5], [-0.5, 143.5]])
MIN_STEP = 15  # grey levels
STEP_PASSES = 4
PROFILE = np.arange(-6, 7.01, 0.5)  # px along the normal, sampled for the edge model
E_GRID = np.arange(-1.5, 1.51, 0.1)  # px: edge offsets tried around the current line; inside +-pitch, where
#                                     shifting the model one pixel out (c0 = background) fits almost as well
SIGMAS = (0.5, 0.8, 1.1, 1.5, 2.0, 2.6)  # px: blur widths tried
MAX_MODEL_SIGMA = 1.3  # px: above this the gradient-pass edges are kept (measured: better there)
N_PIX = 4  # screen pixels with their own shade in the edge model (the last one runs on)


MAX_CHROMA = 0.3  # channel spread of image / background above which a pixel is clutter, not screen


RING = 12  # px of image border used to model the background
BG_DIST = 20  # grey levels (channel mean) from the background model that count as screen
HOUGH_VOTES = 60  # outline pixels on a line for it to count (a screen side has ~300)
LINE_TOL = 3  # px: outline points this close to a Hough line refit it


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
    # Mean absolute difference from the background model: textured dark tile rows average to the
    # background grey but their absolute difference stays high; noise averages to well below BG_DIST.
    img = image.reshape(*image.shape[:2], -1).astype(np.float32)
    bg = background(image)
    diff = np.abs((img - bg).mean(-1))  # channel mean first: noise falls by sqrt(3), the grey screen stays
    mask = np.where(cv2.GaussianBlur(diff, (5, 5), 0) > BG_DIST, 255, 0).astype(np.uint8)
    if img.shape[2] == 3:
        # Divided by the background model, the colour cast cancels: screen pixels come out grey at any
        # brightness (HSV saturation does not: dark pixels under a cast read as saturated). Clutter is coloured.
        ratio = cv2.GaussianBlur(img, (5, 5), 0) / np.maximum(bg, 1)
        mask[ratio.max(-1) - ratio.min(-1) > MAX_CHROMA] = 0
    # Order-statistic (min/max) filters on the binary mask (U2-T1), i.e. opening then closing.
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    quad = _hough_quad(contour, mask.shape)
    if quad is not None:
        return quad
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float32)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return order(quad)


def _hough_quad(contour, shape):
    """Screen sides = the strongest Hough line of the mask outline on each side of its centroid (above,
    below, left, right). A screen side is ~300 px long, a clutter box joined to the mask <= 80 px, so the
    side outvotes it. Each line is refitted to the outline points within LINE_TOL px. None if a side is
    missing."""
    outline = np.zeros(shape, np.uint8)
    cv2.drawContours(outline, [contour], -1, 255, 1)
    lines = cv2.HoughLines(outline, 1, np.pi / 360, HOUGH_VOTES)
    if lines is None:
        return None
    m = cv2.moments(contour)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    pts = contour.reshape(-1, 2).astype(np.float32)
    sides = {}
    for rho, theta in lines[:, 0]:  # sorted by votes, strongest first
        n = np.array([np.cos(theta), np.sin(theta)])
        horizontal = abs(n[1]) > abs(n[0])
        d = rho - n @ (cx, cy)  # signed distance of the line from the centroid along n
        side = ("bottom" if d * n[1] > 0 else "top") if horizontal else ("right" if d * n[0] > 0 else "left")
        if side not in sides:
            near = pts[np.abs(pts @ n - rho) < LINE_TOL]
            if len(near) >= 10:
                sides[side] = cv2.fitLine(near, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if len(sides) == 4:
            break
    if len(sides) < 4:
        return None
    ring = [sides[k] for k in ("left", "top", "right", "bottom")]
    try:
        return order(np.stack([_intersect(ring[i], ring[(i + 1) % 4]) for i in range(4)]))
    except ValueError:
        return None


def order(pts):
    """(tl, tr, br, bl) by angle around the centre, starting top-left."""
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # clockwise from left in image coords
    k = int(np.argmin(pts.sum(1)))
    return np.roll(pts, -k, axis=0)


def _model_points(grey, a, b, w):
    """Edge points along the side a-b by fitting each normal profile (outside = negative offsets) with
    background b, then screen pixels of pitch w (px) with free shades c0..c3, all blurred by a Gaussian of
    width sigma: profile(x) = b (1 - F(x - e)) + sum_i c_i (F(x - e - i w) - F(x - e - (i + 1) w)), F the
    Gaussian CDF, c3 running on. For fixed (e, sigma) this is linear in (b, c0..c3), so a grid over e and
    sigma needs only least squares, for all profiles at once. sigma is shared along the side (minimum total
    residual, returned); each profile gives its best e, to sub-grid by a parabola. Profiles whose fitted step
    |c0 - b| < MIN_STEP (screen edge the same grey as the background) or whose e is at the grid end are dropped."""
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    ss = np.linspace(0.05, 0.95, 3 * SAMPLES)
    p = a + ss[:, None] * (b - a)  # (P, 2)
    xy = (p[:, None] + PROFILE[None, :, None] * n).astype(np.float32)  # (P, K, 2)
    ok = (xy.min((1, 2)) >= 0) & (xy[..., 0].max(1) < grey.shape[1] - 1) & (xy[..., 1].max(1) < grey.shape[0] - 1)
    p, xy = p[ok], xy[ok]
    if len(p) < 10:
        return np.zeros((0, 2), np.float32), 0.0
    V = cv2.remap(grey, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR).T.astype(np.float64)  # (K, P)
    vv = (V * V).sum(0)
    z = (PROFILE[None, None, :, None] - E_GRID[None, :, None, None] - w * np.arange(N_PIX)) / np.array(SIGMAS)[:, None, None, None]
    F = _cdf(z)  # (S, E, K, N_PIX)
    A = np.concatenate([1 - F[..., :1], F[..., :-1] - F[..., 1:], F[..., -1:]], -1)  # (S, E, K, 5)
    Q, R = np.linalg.qr(A)
    QV = np.einsum("sekc,kp->secp", Q, V)
    res = vv - (QV * QV).sum(2)  # (S, E, P)
    coef = np.linalg.solve(R, QV)[:, :, :2]  # b, c0
    i = int(np.argmin(res.min(1).sum(1)))
    r = res[i]  # (E, P)
    j = np.argmin(r, 0)
    cols = np.arange(r.shape[1])
    keep = (j > 0) & (j < len(E_GRID) - 1)
    keep &= np.abs(coef[i, j, 1, cols] - coef[i, j, 0, cols]) >= MIN_STEP
    jj = np.clip(j, 1, len(E_GRID) - 2)
    r0, r1, r2 = r[jj - 1, cols], r[jj, cols], r[jj + 1, cols]
    den = r0 - 2 * r1 + r2
    frac = np.where(den > 0, 0.5 * (r0 - r2) / np.where(den > 0, den, 1), 0)
    e = E_GRID[jj] + np.clip(frac, -0.5, 0.5) * (E_GRID[1] - E_GRID[0])
    return (p + e[:, None] * n)[keep].astype(np.float32), SIGMAS[i]


def _cdf(z):
    return 0.5 * (1 + _erf(z / np.sqrt(2)))


def _erf(x):  # Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    t = 1 / (1 + 0.3275911 * np.abs(x))
    y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * np.exp(-x * x)
    return np.sign(x) * y


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
        # pixel pitch across each side: the neighbouring sides' length / pixel count
        lens = [np.linalg.norm(edges[(i + 1) % 4] - edges[i]) for i in range(4)]
        pitch = [(lens[1] + lens[3]) / 288, (lens[0] + lens[2]) / 320] * 2
        lines = []
        for i in range(4):
            pts, sigma = _model_points(g, edges[i], edges[(i + 1) % 4], pitch[i])
            if len(pts) < 10 or sigma > MAX_MODEL_SIGMA:  # too blurred: shades near the edge are unresolved
                break
            lines.append(cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        else:
            edges = np.stack([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    H = cv2.getPerspectiveTransform(EDGE_QUAD, edges)
    return cv2.perspectiveTransform(CORNERS[None], H)[0]
