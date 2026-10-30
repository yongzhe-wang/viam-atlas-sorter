"""
The drop target when it is an orange pumpkin bucket: found by colour, its rim fitted as a circle, located at the rim's height.

WHY: the user replaced the blue square with a Halloween bucket and moves it around. It stands ~180 mm tall, so its rim is much
nearer the camera than the table: at z 400 the rim is inside the depth blind range and the bucket read as an intruder ("blob 119
mm tall", "blind-range hole"), which pauses the arm. Its xy must be computed at the rim's depth (perspective), and it may run off
the frame, so the rim is fitted from the arc in view.
{VIAM CENTERAL 2026-10-30 "用户把蓝方框换成了橙色南瓜桶"; "桶沿比桌面高约 180 MM；桶口直径约 197 MM"; Z580 "ORANGE MASK BBOX U444-920, V45-511"}
[CONFIDENCE: CONFIRMED 80% — two real frames of one bucket]

find_mask(rgb) needs only colour (segment_full cuts the bucket out before it fits the table); find_bucket() adds where it is.
Map entry (tmap["bucket"]): {xy, radius_mm, rim_height_mm, partial, confidence, seen_ts, misses}.
"""
import math
import os

import numpy as np
from scipy import ndimage

TARGET = os.environ.get("SORTER_TARGET", "bucket")  # "bucket": use it when one is in view; "square": ignore buckets
RIM_HEIGHT_MM = 180.0  # measured on this bucket; used while the rim is too close for depth
RADIUS_MM = 98.5  # 197 mm mouth
RADIUS_RANGE_MM = (70.0, 130.0)  # at the rim's depth; anything else orange is not this bucket
MIN_AREA_PX = 15000  # at the 580 mm view it is ~150000 px; a corner of it in view may be much less
MIN_AREA_NO_CIRCLE_PX = 40000  # cut out without a circle only when this big: an orange block is < 15000 px
MIN_INLIERS = 0.45  # share of the contour on the fitted circle (handle lugs, the frame and objects take the rest)
MASK_GROW_PX = 14  # segmentation stays this far from the bucket: its shadow, and parallax between colour and depth
ARC_FULL = 0.8  # share of the circle seen for it not to count as partial
EDGE_PX = 6
MOVED_MM = 25.0  # a sighting this far from the map's entry: the bucket was moved, jump to it
ALPHA = 0.5


def orange_mask(rgb: np.ndarray) -> np.ndarray:
    """The bucket's colour. {VIAM CENTERAL "(R>130)&(R-G>45)&(G-B>15)&(R-B>70)"}: the red knife fails on g-b, yellow on r-g."""
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    return (r > 130) & (r - g > 45) & (g - b > 15) & (r - b > 70)


def _fit_circle(pts: np.ndarray, tol: float, iters: int = 150) -> tuple:
    """RANSAC circle on contour points (seeded: same frame, same answer): (u, v, r, inlier mask), or None."""
    if len(pts) < 20:
        return None
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        A = np.array([b - a, c - a]) * 2.0
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        o = np.linalg.solve(A, np.array([b @ b - a @ a, c @ c - a @ a]))
        r = math.dist(o, a)
        n = int((np.abs(np.hypot(*(pts - o).T) - r) < tol).sum())
        if n > best_n:
            best, best_n = (o, r), n
    if best is None:
        return None
    o, r = best
    for _ in range(2):  # least squares on the inliers
        inl = np.abs(np.hypot(*(pts - o).T) - r) < tol
        M = np.column_stack([2 * pts[inl], np.ones(int(inl.sum()))])
        sol, *_ = np.linalg.lstsq(M, (pts[inl] ** 2).sum(1), rcond=None)
        o = sol[:2]
        r = math.sqrt(max(sol[2] + o @ o, 1.0))
    inl = np.abs(np.hypot(*(pts - o).T) - r) < tol
    return float(o[0]), float(o[1]), float(r), inl


def find_mask(rgb: np.ndarray) -> dict:
    """
    {"mask" (bool, the bucket and a margin, circle included), "center_px", "radius_px", "arc", "inliers", "cut"} or None.

    Largest orange region, holes filled (the pumpkin's dark face, anything lying in it); the rim is the circle through its
    outline, ignoring the part of the outline that is the image border.
    """
    H, W = rgb.shape[:2]
    orange = ndimage.binary_closing(orange_mask(rgb), iterations=3)
    lab, n = ndimage.label(orange)
    if not n:
        return None
    sizes = ndimage.sum(orange, lab, range(1, n + 1))
    k = int(np.argmax(sizes))
    if sizes[k] < MIN_AREA_PX:
        return None
    region = ndimage.binary_fill_holes(lab == k + 1)
    vs, us = np.nonzero(region & ~ndimage.binary_erosion(region, border_value=1))
    keep = (us > EDGE_PX) & (us < W - 1 - EDGE_PX) & (vs > EDGE_PX) & (vs < H - 1 - EDGE_PX)
    pts = np.column_stack([us[keep], vs[keep]]).astype(float)
    if len(pts) < 50:
        return None
    pts = pts[:: max(1, len(pts) // 800)]
    fit = _fit_circle(pts, tol=max(4.0, 0.012 * math.sqrt(sizes[k] / math.pi)))
    share = float(fit[3].mean()) if fit is not None else 0.0
    if fit is None or share < MIN_INLIERS:
        # No circle: a tilted view sees the mouth as an ellipse. A region this big is still the bucket to cut out (else it
        # is a 170 mm "intruder" and the arm pauses), but there is no centre to put on the map.
        # {TILT_Y+ "INTR ['BLOB 174 MM TALL, SKIN 7%'…] BUCKET FALSE"} [CONFIDENCE: CONFIRMED 75% — one tilted frame]
        if sizes[k] < MIN_AREA_NO_CIRCLE_PX:
            return None
        mask = ndimage.binary_dilation(region, iterations=MASK_GROW_PX)
        return {"mask": mask, "center_px": None, "radius_px": None, "arc": 0.0, "inliers": share, "cut": bool((~keep).any())}
    cu, cv, r, inl = fit
    angles = np.arctan2(pts[inl, 1] - cv, pts[inl, 0] - cu)
    arc = len(np.unique(((angles + math.pi) / (2 * math.pi) * 72).astype(int))) / 72.0
    vv, uu = np.ogrid[:H, :W]
    disc = (uu - cu) ** 2 + (vv - cv) ** 2 <= (r + MASK_GROW_PX) ** 2
    mask = disc | ndimage.binary_dilation(region, iterations=MASK_GROW_PX)
    return {"mask": mask, "center_px": (cu, cv), "radius_px": r, "arc": arc, "inliers": share, "cut": bool((~keep).any())}


def locate(hit: dict, depth: np.ndarray, table_depth_mm: float, pose: dict, intr: dict) -> dict:
    """
    Where a find_mask() hit is on the table: {xy, radius_mm, rim_height_mm, partial, confidence}, or None if it is not this size.

    The rim's depth is read just inside the fitted circle when the camera can see it; at z 400 it is inside the blind range,
    then it is the table's depth minus the known rim height. xy is at that depth: at table depth the same pixels would be ~50%
    further from the camera axis.
    """
    from sorter import perception

    if hit.get("center_px") is None:
        return None  # no circle fitted (tilted view): cut out, but not located
    cu, cv, r = *hit["center_px"], hit["radius_px"]
    H, W = depth.shape
    vv, uu = np.ogrid[:H, :W]
    dist = np.hypot(uu - cu, vv - cv)
    ring = depth[(dist > r - 25) & (dist < r - 5) & (depth > 0)]
    rim_height, measured = RIM_HEIGHT_MM, False
    if ring.size >= 300:
        near = float(np.percentile(ring, 20))
        if 100.0 <= table_depth_mm - near <= 260.0:
            rim_height, measured = table_depth_mm - near, True
    rim_depth = table_depth_mm - rim_height
    radius = r * rim_depth / float(intr["fx"])
    if not RADIUS_RANGE_MM[0] <= radius <= RADIUS_RANGE_MM[1]:
        return None
    xy = perception.world_xy(pose, intr, cu, cv, rim_depth)
    return {"xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)], "radius_mm": round(float(radius), 1),
            "rim_height_mm": round(float(rim_height), 1), "partial": bool(hit["arc"] < ARC_FULL),
            "confidence": round(float(hit["arc"] * hit["inliers"]) * (1.0 if measured else 0.9), 2)}


def find_bucket(rgb: np.ndarray, depth: np.ndarray, pose: dict, intr: dict) -> dict:
    """{xy, radius_mm, rim_height_mm, partial, confidence} of the bucket in this frame, or None."""
    from sorter import perception

    hit = find_mask(rgb)
    if hit is None:
        return None
    d = depth.astype(float).copy()
    d[hit["mask"]] = 0
    a, b, c = perception.fit_table(d)
    H, W = d.shape
    return locate(hit, depth.astype(float), float(a * hit["center_px"][0] + b * hit["center_px"][1] + c), pose, intr)


def update(entry: dict, seen: dict, now: float) -> dict:
    """Fold one sighting into the map's entry: a full view beats a partial one, a far sighting is a moved bucket."""
    if seen is None:
        return entry
    if entry is None or math.dist(entry["xy"], seen["xy"]) > MOVED_MM or (entry.get("partial") and not seen["partial"]):
        return {**seen, "seen_ts": now, "misses": 0}
    if seen["partial"] and not entry.get("partial"):
        return {**entry, "seen_ts": now, "misses": 0}  # a full view's centre is better than an arc's
    xy = [round((1 - ALPHA) * entry["xy"][k] + ALPHA * seen["xy"][k], 1) for k in (0, 1)]
    return {**seen, "xy": xy, "seen_ts": now, "misses": 0}


def contains(entry: dict, xy, margin: float = 0.0) -> bool:
    """Is table point xy inside the bucket's mouth?"""
    return bool(entry) and math.dist(entry["xy"], xy) <= entry["radius_mm"] + margin
