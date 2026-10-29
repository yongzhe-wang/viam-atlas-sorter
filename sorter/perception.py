"""
Find every object on the table by height, not colour.

WHY: the demo must pick "whatever is on the table", so segmentation cannot assume a colour. The RealSense
depth map gives height above the table directly; flat blue tape has ~0 height and drops out by
construction.

Flow: depth -> robust table plane fit -> height map -> blobs 8-100 mm tall (depth grid) -> mapped onto the
colour grid -> Obj (geometry from the colour mask, crop from colour, heights from depth).
Intruders (hands, arms) are reported separately so the executor can pause; see find_intruders().

Contract: docs/EXEC_PLAN.md (Obj, segment). segment_full() also returns intruders and the alignment used.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

MIN_HEIGHT_MM = 8.0  # objects stand at least this far above the fitted table plane
MAX_OBJ_HEIGHT_MM = 100.0  # taller is not a graspable object (blocks are < 80 mm): hand or arm
MIN_AREA_PX = 800  # ~12x12 mm at the 378 mm observe depth (about 2.4 px/mm)
CROP_MARGIN = 0.15  # crop = bbox grown by this fraction on each side
# depth -> colour pixel model: colour = scale * (depth - centre) + centre + (tu, tv)
IDENTITY = (1.0, 0.0, 0.0)  # align_color_depth on
# Unaligned depth has a wider field of view than colour, so it needs a scale, not just a shift. Fitted by
# overlaying the depth object mask on colour (laptop + phone at the image edges line up at 1.35; 1.25 and
# 1.45 visibly miss) and consistent with the one point measured by AGENT1.
# {FRAME f01 2026-10-29 overlay test} {PICK_PLACE_AUTO.PY "SAME BLOCK SEEN AT COLOUR (767,284) AND DEPTH (707,306)"}
# [CONFIDENCE: SINGLE-SRC 75% — one frame at ~378 mm; parallax makes the shift depth dependent]
UNALIGNED = (1.35, 37.0, -3.0)
GRIPPER_BAND_V = 640  # fingers / held object occupy the image below this row (colour grid)
GRIPPER_U = (400, 1280)  # ... and these columns
DEPTH_LEFT_INVALID_U = 60  # RealSense depth is invalid in a band at its left edge (stereo baseline)
INTRUDER_MIN_AREA_PX = 6000  # blind-range hole this large could be a hand
SKIN_FRAC = 0.30  # fraction of skin-coloured pixels that turns a hole or blob into an intruder
# An intruder must reach this far into the image: a hand reaching for an object does, while static tall
# scenery cut by the border does not (a laptop hinge at the top edge read 102 mm and paused everything).
INTRUDER_INNER_MARGIN_PX = 150


def _reaches_inside(bbox: tuple, shape: tuple) -> bool:
    """True if the bbox overlaps the image shrunk by INTRUDER_INNER_MARGIN_PX on every side."""
    u0, v0, u1, v1 = bbox
    m, (h, w) = INTRUDER_INNER_MARGIN_PX, shape
    return u1 > m and u0 < w - m and v1 > m and v0 < h - m


@dataclass
class Obj:
    id: int
    bbox: tuple  # (u0, v0, u1, v1) colour pixels
    centroid_px: tuple  # (u, v)
    yaw_deg: float  # long axis in the image, ccw from +u, [0, 180)
    length_px: float
    width_px: float
    height_mm: float  # top above the table plane
    top_depth_mm: float  # camera -> top surface
    mask: np.ndarray = field(repr=False)  # bool (H, W), colour image grid
    crop: np.ndarray = field(repr=False)  # uint8 RGB crop around bbox
    touches_edge: bool = False  # cut by the image border: geometry and crop are partial
    colour: str = ""  # dominant colour word of the mask, see colour_name()


@dataclass
class Segmentation:
    table_depth_mm: float  # fitted plane at the image centre
    objects: list
    intruders: list  # [{"bbox", "area", "reason"}]: pause the arm while non-empty
    align: tuple  # depth -> colour model that was used (IDENTITY or UNALIGNED)
    plane: tuple  # (a, b, c): table depth = a*u + b*v + c on the depth grid


def fit_table(depth: np.ndarray) -> tuple:
    """
    Robust plane fit of the table: depth = a*u + b*v + c (depth grid, mm).

    WHY: on sfhack4 the table reads 373-379 mm across one frame (measured), as large as the 8 mm object
    threshold, so a single table depth would lose low objects. Start from the histogram peak, then
    least squares on near-peak pixels, twice, tightening the inlier band.
    """
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:4, 0:w:4]
    d = depth[::4, ::4]
    valid = d > 0
    hist, edges = np.histogram(d[valid], bins=np.arange(100, 2000, 4))
    peak = edges[np.argmax(hist)] + 2
    inl = valid & (np.abs(d - peak) < 8)
    a = b = 0.0
    c = float(peak)
    for band in (6.0, 3.0):
        A = np.column_stack([uu[inl], vv[inl], np.ones(int(inl.sum()))])
        (a, b, c), *_ = np.linalg.lstsq(A, d[inl], rcond=None)
        res = d - (a * uu + b * vv + c)
        inl = valid & (np.abs(res) < band)
    return float(a), float(b), float(c)


def to_colour(img: np.ndarray, model: tuple) -> np.ndarray:
    """Resample a depth-grid label/mask image onto the colour grid (nearest neighbour)."""
    s, tu, tv = model
    if model == IDENTITY:
        return img.copy()
    h, w = img.shape
    c = np.array([h / 2.0, w / 2.0])
    # affine_transform maps each output (colour) pixel back to its input (depth) pixel
    off = c - (c + np.array([tv, tu])) / s
    return ndimage.affine_transform(img, np.eye(2) / s, offset=off, order=0, output_shape=img.shape)


def non_table_colour(rgb: np.ndarray, table_px: np.ndarray) -> np.ndarray:
    """Pixels whose colour is far from the table's (Mahalanobis on RGB, model from table pixels)."""
    px = rgb.reshape(-1, 3).astype(float)
    sample = rgb[table_px].astype(float)
    if len(sample) > 20000:
        sample = sample[:: len(sample) // 20000]
    mu, cov = sample.mean(0), np.cov(sample.T) + np.eye(3) * 4.0
    diff = px - mu
    m2 = np.einsum("ij,jk,ij->i", diff, np.linalg.inv(cov), diff)
    return (m2 > 16.0).reshape(rgb.shape[:2])


def pick_alignment(rgb: np.ndarray, above: np.ndarray, table: np.ndarray) -> tuple:
    """
    IDENTITY or UNALIGNED, whichever puts the depth object mask on non-table-coloured pixels.

    WHY: AGENT1 turns align_color_depth on at some point; checking per frame means nobody has to flip a
    flag in this code at the same moment. Score = IoU with the colour "not table" mask.
    """
    if above.sum() < MIN_AREA_PX:
        return UNALIGNED
    cue = non_table_colour(rgb, to_colour(table, UNALIGNED) & to_colour(table, IDENTITY))
    best, best_iou = UNALIGNED, -1.0
    for model in (UNALIGNED, IDENTITY):
        m = to_colour(above, model)
        iou = (m & cue).sum() / max((m | cue).sum(), 1)
        if iou > best_iou:
            best, best_iou = model, iou
    return best


def _geometry(mask: np.ndarray) -> tuple:
    """(centroid, yaw_deg, length_px, width_px) from mask PCA; a w x l rectangle has variance l^2/12."""
    vs, us = np.nonzero(mask)
    cu, cv = us.mean(), vs.mean()
    w_, vec = np.linalg.eigh(np.cov(np.vstack([us - cu, vs - cv])))
    yaw = math.degrees(math.atan2(-vec[1, 1], vec[0, 1])) % 180.0
    return (float(cu), float(cv)), yaw, math.sqrt(12 * max(w_[1], 0)), math.sqrt(12 * max(w_[0], 0))


def skin_fraction(rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Fraction of masked pixels in the YCrCb skin range (Cr 140-175, Cb 85-125, Y > 60).

    Heuristic: light wood also falls near this range, so it is applied only to depth holes and to blobs
    the robot is about to treat as objects, never to the table itself. Not yet tested with a real hand.
    """
    px = rgb[mask].astype(float)
    if len(px) == 0:
        return 0.0
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    return float(((cr > 140) & (cr < 175) & (cb > 85) & (cb < 125) & (y > 60)).mean())


def find_intruders(rgb: np.ndarray, valid: np.ndarray, model: tuple) -> list:
    """
    Hands/arms: large invalid-depth regions that are skin-coloured.

    WHY: at the observe pose the camera is ~378 mm above the table and depth is blind closer than ~280 mm,
    so a hand more than ~100 mm above the table appears as a hole, not as a tall blob. Holes at the depth
    sensor's left edge and in the gripper zone (fingers, held object) are excluded; a glossy phone screen
    is also a hole but not skin-coloured.
    """
    hole = ndimage.binary_opening(~valid, iterations=6)
    hole[:, :DEPTH_LEFT_INVALID_U] = False
    hole = to_colour(hole, model)
    hole[GRIPPER_BAND_V:, GRIPPER_U[0]:GRIPPER_U[1]] = False
    lab, n = ndimage.label(hole)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < INTRUDER_MIN_AREA_PX:
            continue
        vs, us = np.nonzero(m)
        bbox = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
        sk = skin_fraction(rgb, m)
        if sk >= SKIN_FRAC and _reaches_inside(bbox, m.shape):
            out.append({"bbox": bbox, "area": area, "reason": f"blind-range hole, skin {sk:.0%}"})
    return out


def segment_full(rgb: np.ndarray, depth: np.ndarray, align=None) -> Segmentation:
    """
    Objects, intruders and table depth from one colour + depth frame (both 1280x720).

    `align`: depth -> colour model; None picks IDENTITY or UNALIGNED from the frame itself.
    """
    d = depth.astype(float)
    valid = (d > 0) & (d < 2000)
    a, b, c = fit_table(d)
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    height = np.where(valid, a * uu + b * vv + c - d, 0.0)

    above = ndimage.binary_opening(valid & (height > MIN_HEIGHT_MM) & (height < 300), iterations=2)
    table = valid & (np.abs(height) < 3)
    model = align if align is not None else pick_alignment(rgb, above, table)

    lab_d, n = ndimage.label(above)
    lab_c = to_colour(lab_d, model)
    lab_c[GRIPPER_BAND_V:, GRIPPER_U[0]:GRIPPER_U[1]] = 0
    H, W = d.shape
    objs, intruders = [], []
    for i in range(1, n + 1):
        m_d = lab_d == i
        if m_d.sum() < MIN_AREA_PX:
            continue
        m = lab_c == i
        if m.sum() < MIN_AREA_PX // 2:
            continue  # mapped outside the colour view or into the gripper zone
        top_h = float(np.percentile(height[m_d], 90))
        vs, us = np.nonzero(m)
        bbox = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
        # A tall blob, or a skin-coloured one (a hand resting on the table), is never a target
        sk = skin_fraction(rgb, m)
        if top_h > MAX_OBJ_HEIGHT_MM or sk >= 0.5:
            # Never a target; an intruder only if it reaches into the workspace (else static scenery)
            if _reaches_inside(bbox, m.shape):
                intruders.append({"bbox": bbox, "area": int(m.sum()),
                                  "reason": f"blob {top_h:.0f} mm tall, skin {sk:.0%}"})
            continue
        u0, v0, u1, v1 = bbox
        mu, mv = int((u1 - u0) * CROP_MARGIN) + 4, int((v1 - v0) * CROP_MARGIN) + 4
        crop = rgb[max(0, v0 - mv):min(H, v1 + mv + 1), max(0, u0 - mu):min(W, u1 + mu + 1)].copy()
        cen, yaw, L, Wd = _geometry(m)
        objs.append(Obj(id=len(objs), bbox=bbox, centroid_px=cen, yaw_deg=yaw, length_px=L, width_px=Wd,
                        height_mm=top_h, top_depth_mm=float(np.percentile(d[m_d], 10)), mask=m, crop=crop,
                        touches_edge=u0 <= 2 or v0 <= 2 or u1 >= W - 3 or v1 >= H - 3,
                        colour=colour_name(rgb, m)[0]))
    intruders += find_intruders(rgb, valid, model)
    return Segmentation(table_depth_mm=float(a * W / 2 + b * H / 2 + c), objects=objs, intruders=intruders,
                        align=model, plane=(a, b, c))


def segment(rgb: np.ndarray, depth: np.ndarray) -> tuple:
    """(table_depth_mm, objects), the EXEC_PLAN contract. Use segment_full() for intruders and alignment."""
    s = segment_full(rgb, depth)
    return s.table_depth_mm, s.objects


def colour_name(rgb: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Dominant colour word of the masked pixels and the fraction of pixels that have it.

    Saturated pixels vote by hue (degrees); low saturation splits into white / grey / black by value.
    Hue bins chosen by eye on sfhack4 frames (the yellow block is ~55 deg, the blue tape ~210 deg).
    """
    px = rgb[mask].astype(float) / 255.0
    if len(px) == 0:
        return "unknown", 0.0
    mx, mn = px.max(1), px.min(1)
    d = np.maximum(mx - mn, 1e-6)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    names = np.full(len(px), "grey", dtype=object)
    # Dark pixels carry a faint hue from white balance; a black charger read "blue" before this
    sat = (s > 0.35) & (mx > 0.3)
    names[~sat & (mx > 0.8)] = "white"
    names[mx <= 0.3] = "black"
    for lo, hi, nm in [(0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 165, "green"),
                       (165, 200, "cyan"), (200, 260, "blue"), (260, 330, "purple"), (330, 361, "red")]:
        names[sat & (hue >= lo) & (hue < hi)] = nm
    vals, counts = np.unique(names, return_counts=True)
    k = int(np.argmax(counts))
    return str(vals[k]), float(counts[k] / len(px))
