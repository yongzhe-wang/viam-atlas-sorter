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
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

MIN_HEIGHT_MM = 8.0  # objects stand at least this far above the fitted table plane
MAX_OBJ_HEIGHT_MM = 100.0  # taller is not a graspable object (blocks are < 80 mm): hand or arm
MIN_AREA_PX = 800  # ~12x12 mm at the 378 mm observe depth (about 2.4 px/mm)
# Pixel-area thresholds are tuned at the z=400 observe pose (table ~376 mm away). Higher sweep poses (z 480-550,
# table ~460-530 mm) shrink every object by (depth ratio)^2, so areas scale with the measured table depth;
# capped at 1 so z=400 frames are processed exactly as before.
REF_TABLE_DEPTH_MM = 380.0
CROP_MARGIN = 0.15  # crop = bbox grown by this fraction on each side
# depth -> colour pixel model: colour = scale * (depth - centre) + centre + (tu, tv)
IDENTITY = (1.0, 0.0, 0.0)  # align_color_depth on
# Unaligned depth has a wider field of view than colour, so it needs a scale, not just a shift. Fitted by
# overlaying the depth object mask on colour (laptop + phone at the image edges line up at 1.35; 1.25 and
# 1.45 visibly miss) and consistent with the one point measured by AGENT1.
# {FRAME f01 2026-10-29 overlay test} {PICK_PLACE_AUTO.PY "SAME BLOCK SEEN AT COLOUR (767,284) AND DEPTH (707,306)"}
# [CONFIDENCE: SINGLE-SRC 75% — one frame at ~378 mm; parallax makes the shift depth dependent]
UNALIGNED = (1.35, 37.0, -3.0)
# Fingertips seen by the wrist camera (colour grid). A held object needs no mask: it is inside the depth
# blind range. A full-width bottom band here once erased a block sitting in the square near the bottom.
# {FRAMES 2026-10-29 "CLOSED FINGERS (830,673)+(891,677)" "OPEN FINGERS (555,697)+(1181,696)"}
# [CONFIDENCE: CONFIRMED 85% — finger pixels measured by AGENT1; boxes padded ~60 px]
FINGER_BOXES = ((480, 640, 640, 720), (1100, 640, 1280, 720), (770, 630, 950, 720))
DEPTH_LEFT_INVALID_U = 60  # RealSense depth is invalid in a band at its left edge (stereo baseline)
# Hysteresis: an object is a connected region > GROW_HEIGHT_MM that contains a seed > MIN_HEIGHT_MM.
# A phone (~7 mm body, glossy) and a laptop came out as separate 9-10 mm fragments with one threshold;
# growing through their low bodies joins each into one object, which then touches the edge and is excluded.
GROW_HEIGHT_MM = 3.0
# Tall objects (60 mm cylinders) seen off-centre show top + side with an invalid-depth occlusion band in
# between, which split one cylinder into a 61 mm and a 28 mm piece on frame e01. Invalid pixels this close
# to an object bridge pieces for connectivity only; geometry still uses valid pixels.
BRIDGE_INVALID_PX = 10
# Top face = pixels within this of the object's top height. Off-centre, a standing object shows its sides
# too (a cube seen at a corner is a hexagon, fill ~0.75 like a cylinder); its top face is still a square.
# The user's photos confirm the cylinders stand upright and the green cube is a separate ~30 mm object.
TOP_BAND_MM = 8.0
EDGE_MARGIN_PX = 8  # an object this close to the image border is cut off: excluded, not a target
# Optional workspace polygon in colour pixels at the observe pose: [[u, v], ...]; absent = whole image
WORKSPACE_FILE = Path(__file__).resolve().parent.parent / "data" / "workspace.json"
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


def _mask_fingers(img: np.ndarray) -> None:
    """Zero the fingertip boxes in place (labels or bool masks on the colour grid)."""
    for u0, v0, u1, v1 in FINGER_BOXES:
        img[v0:v1, u0:u1] = 0


def load_workspace():
    """Workspace polygon [(u, v), ...] from data/workspace.json, or None (whole image)."""
    if WORKSPACE_FILE.exists():
        return [tuple(p) for p in json.loads(WORKSPACE_FILE.read_text())]
    return None


def _inside(poly, u: float, v: float) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    for (u1, v1), (u2, v2) in zip(poly, poly[1:] + poly[:1]):
        if (v1 > v) != (v2 > v) and u < (u2 - u1) * (v - v1) / (v2 - v1) + u1:
            inside = not inside
    return inside


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
    rect_fill: float = 1.0  # top face hull / min-area rectangle: square/rectangle ~1.0, disc ~0.81
    top_centroid_px: tuple = ()  # centroid of the top face: above the object's axis from any view angle
    colours: dict = field(default_factory=dict)  # share of each saturated colour family in the mask (>= 2%)
    # Merged see-through object (MERGE_FRAGMENTS) only: where to grip it (centre of the opaque piece with valid depth
    # nearest the object's centre) and that piece's camera distance; None: use the object's own geometry
    grip_px: tuple = None
    grip_depth_mm: float = None
    n_fragments: int = 1
    flat: bool = False  # LOW_OBJECTS: 3-12 mm above the table, found by colour and confirmed by depth (a sweet bag)


@dataclass
class Segmentation:
    table_depth_mm: float  # fitted plane at the image centre
    objects: list  # fully visible, inside the workspace: the only things the arm may target
    excluded: list  # [(Obj, reason)]: cut by the image border or outside the workspace (debug / dashboard)
    intruders: list  # [{"bbox", "area", "reason"}]: pause the arm while non-empty
    align: tuple  # depth -> colour model that was used (IDENTITY or UNALIGNED)
    plane: tuple  # (a, b, c): table depth = a*u + b*v + c on the depth grid
    cables: list = field(default_factory=list)  # SPLIT_BLOBS: [{"bbox", "length_mm", "width_mm"}] depth grid, never targets


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


def alignment_ious(rgb: np.ndarray, above: np.ndarray, table: np.ndarray) -> dict:
    """IoU of the depth object mask with the colour "not table" mask, under each depth -> colour model."""
    cue = non_table_colour(rgb, to_colour(table, UNALIGNED) & to_colour(table, IDENTITY))
    out = {}
    for name, model in (("identity", IDENTITY), ("unaligned", UNALIGNED)):
        m = to_colour(above, model)
        out[name] = float((m & cue).sum() / max((m | cue).sum(), 1))
    return out


# Decided once per process on the first frame with objects; see segment_full()
_ALIGN = {"model": None}


def _startup_alignment(rgb: np.ndarray, above: np.ndarray, table: np.ndarray) -> tuple:
    """
    align_color_depth is on (verified 2026-10-29: IoU 0.31 at identity vs 0.06 at the old shift), so
    identity is the model. Checked once on the first frame with objects: if the unaligned model fits
    clearly better, someone turned the switch off -> warn and fall back instead of mis-masking everything.
    {VIAM CENTERAL 2026-10-29 "ALIGN_COLOR_DEPTH IS NOW ON — DEPTH PIXELS MAP 1:1 TO COLOUR PIXELS"}
    [CONFIDENCE: CONFIRMED 90% — also seen on frame f02: per-frame check picked identity]
    """
    if _ALIGN["model"] is None and above.sum() >= MIN_AREA_PX:
        ious = alignment_ious(rgb, above, table)
        if ious["unaligned"] > ious["identity"] * 1.5:
            warnings.warn(f"depth looks UNALIGNED to colour {ious}; is align_color_depth off? using UNALIGNED")
            _ALIGN["model"] = UNALIGNED
        else:
            _ALIGN["model"] = IDENTITY
    return _ALIGN["model"] or IDENTITY


def rect_fill(mask: np.ndarray) -> float:
    """
    Convex-hull area over its minimum-area bounding rectangle: square/rectangle ~1.0, disc ~0.785.

    WHY: aspect ratio alone called a round cylinder top "square", so "green square" matched a green cylinder.
    The raw mask area was useless on real frames (ragged depth edges: a rectangular block scored 0.81, like a
    disc), so the hull is used; the min rectangle comes from rotating calipers along the hull edges.
    """
    from scipy.spatial import ConvexHull

    vs, us = np.nonzero(mask)
    if len(us) < 10:
        return 1.0
    pts = np.column_stack([us, vs]).astype(float)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0
    hp = pts[hull.vertices]
    best = math.inf
    for i in range(len(hp)):
        e = hp[(i + 1) % len(hp)] - hp[i]
        n = math.hypot(*e)
        if n == 0:
            continue
        ax = e / n
        a = hp @ ax
        b = hp @ np.array([-ax[1], ax[0]])
        best = min(best, np.ptp(a) * np.ptp(b))
    return float(hull.volume / best) if best > 0 else 1.0  # ConvexHull.volume is the area in 2-D


def _geometry(mask: np.ndarray, off: tuple = (0, 0)) -> tuple:
    """
    (centroid, yaw_deg, length_px, width_px) from mask PCA; a w x l rectangle has variance l^2/12.

    `off` = (u, v) of the mask's top-left corner when `mask` is a sub-array, so the centroid is in full-image pixels.
    """
    vs, us = np.nonzero(mask)
    us, vs = us + off[0], vs + off[1]
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


def area_scale(table_depth_mm: float) -> float:
    """Factor for pixel-area thresholds at this table depth: 1 at the observe pose, ~0.5 at z 550."""
    return min(1.0, (REF_TABLE_DEPTH_MM / max(table_depth_mm, 1.0)) ** 2)


def find_intruders(rgb: np.ndarray, valid: np.ndarray, model: tuple, scale: float = 1.0) -> list:
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
    _mask_fingers(hole)
    lab, n = ndimage.label(hole)
    H = hole.shape[0]
    out = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < INTRUDER_MIN_AREA_PX * scale:
            continue
        vs, us = np.nonzero(m)
        bbox = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
        # Attached to the bottom edge under the gripper = the held object itself, not a person
        if bbox[3] >= H - 3 and 450 <= us.mean() <= 1280:
            continue
        sk = skin_fraction(rgb, m)
        if sk >= SKIN_FRAC and _reaches_inside(bbox, m.shape):
            out.append({"bbox": bbox, "area": area, "reason": f"blind-range hole, skin {sk:.0%}"})
    return out


def segment_full(rgb: np.ndarray, depth: np.ndarray, align=None) -> Segmentation:
    """
    Objects, intruders and table depth from one colour + depth frame (both 1280x720).

    `align`: depth -> colour model; None = IDENTITY (align_color_depth on), checked once per process.
    """
    d = depth.astype(float)
    valid = (d > 0) & (d < 2000)
    a, b, c = fit_table(d)
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    height = np.where(valid, a * uu + b * vv + c - d, 0.0)

    above = ndimage.binary_opening(valid & (height > MIN_HEIGHT_MM) & (height < 300), iterations=2)
    table = valid & (np.abs(height) < 3)
    model = align if align is not None else _startup_alignment(rgb, above, table)

    # Grow seeds (> MIN_HEIGHT_MM) through low pixels (> GROW_HEIGHT_MM) so one thing is one object
    low = ndimage.binary_opening(valid & (height > GROW_HEIGHT_MM) & (height < 300), iterations=2) | above
    bridge = ~valid & ndimage.binary_dilation(above, iterations=BRIDGE_INVALID_PX)
    grp, n = ndimage.label(low | bridge)
    seeded = np.zeros(n + 1, bool)
    seeded[np.unique(grp[above])] = True
    seeded[0] = False
    lab_d = np.where(seeded[grp] & low, grp, 0)
    H, W = d.shape
    table_depth = float(a * W / 2 + b * H / 2 + c)
    cables = []
    if SPLIT_BLOBS:
        lab_d, n, cables = _split_cables(lab_d, n, INTRINSICS["fx"] / table_depth)
    # No fingertip mask for objects: the fingers are ~140 mm from the camera, inside the depth blind range, so they
    # never become objects; masking them erased a real 30 mm cube that a z=500 sweep saw in that corner
    lab_c = to_colour(lab_d, model)
    ascale = area_scale(table_depth)
    min_area = MIN_AREA_PX * ascale
    poly = load_workspace()
    objs, excluded, intruders = [], [], []
    # Work on each object's bounding sub-array: a full 1280x720 comparison per label (including hundreds of
    # unseeded noise labels) made segmentation grow with clutter; find_objects gives None for absent labels
    sl_d, sl_c = ndimage.find_objects(lab_d), ndimage.find_objects(lab_c)
    for i in range(1, n + 1):
        sd = sl_d[i - 1] if i - 1 < len(sl_d) else None
        sc = sl_c[i - 1] if i - 1 < len(sl_c) else None
        if sd is None or sc is None:
            continue  # unseeded, or mapped outside the colour view / onto the fingertips
        m_d = lab_d[sd] == i
        if m_d.sum() < min_area:
            continue
        m_s = lab_c[sc] == i
        if m_s.sum() < min_area // 2:
            continue
        cv0, cu0 = sc[0].start, sc[1].start
        top_h = float(np.percentile(height[sd][m_d], 90))
        vs, us = np.nonzero(m_s)
        bbox = (int(cu0 + us.min()), int(cv0 + vs.min()), int(cu0 + us.max()), int(cv0 + vs.max()))
        # A tall blob, or a skin-coloured one (a hand resting on the table), is never a target
        sk = skin_fraction(rgb[sc], m_s)
        if top_h > MAX_OBJ_HEIGHT_MM or sk >= 0.5:
            # Never a target; an intruder only if it reaches into the workspace (else static scenery)
            if _reaches_inside(bbox, (H, W)):
                intruders.append({"bbox": bbox, "area": int(m_s.sum()),
                                  "reason": f"blob {top_h:.0f} mm tall, skin {sk:.0%}"})
            continue
        u0, v0, u1, v1 = bbox
        mu, mv = int((u1 - u0) * CROP_MARGIN) + 4, int((v1 - v0) * CROP_MARGIN) + 4
        crop = rgb[max(0, v0 - mv):min(H, v1 + mv + 1), max(0, u0 - mu):min(W, u1 + mu + 1)].copy()
        m = np.zeros((H, W), bool)
        m[sc] = m_s
        cen, yaw, L, Wd = _geometry(m_s, (cu0, cv0))
        # Top face on the colour grid (falls back to the whole mask for low, flat objects)
        if model == IDENTITY:
            top_s = (lab_d[sc] == i) & (height[sc] >= top_h - TOP_BAND_MM) & m_s
        else:
            full_d = np.zeros((H, W), bool)
            full_d[sd] = m_d & (height[sd] >= top_h - TOP_BAND_MM)
            top_s = to_colour(full_d, model)[sc] & m_s
        if top_s.sum() < 200 * ascale:
            top_s = m_s
        tvs, tus = np.nonzero(top_s)
        e = EDGE_MARGIN_PX
        edge = u0 <= e or v0 <= e or u1 >= W - 1 - e or v1 >= H - 1 - e
        o = Obj(id=0, bbox=bbox, centroid_px=cen, yaw_deg=yaw, length_px=L, width_px=Wd, height_mm=top_h,
                top_depth_mm=float(np.percentile(d[sd][m_d], 10)), mask=m, crop=crop, touches_edge=edge,
                colour=colour_name(rgb[sc], m_s)[0], rect_fill=rect_fill(top_s),
                top_centroid_px=(float(cu0 + tus.mean()), float(cv0 + tvs.mean())),
                colours=colour_fractions(rgb[sc], m_s))
        # Only fully visible objects inside the workspace are targets; the rest is kept for display/debug
        if edge:
            excluded.append((o, "touches the image edge"))
        elif poly and not _inside(poly, *cen):
            excluded.append((o, "outside the workspace"))
        else:
            o.id = len(objs)
            objs.append(o)
    if MERGE_FRAGMENTS:
        objs, excluded = _merge_fragments(objs, excluded, rgb, valid, ascale, (H, W))
        objs = _absorb_slivers(objs, excluded, ascale, (H, W))
    if LOW_OBJECTS:
        objs, excluded = _find_low_objects(objs, excluded, rgb, d, valid, height, ascale)
    for k, (o, _) in enumerate(excluded):
        o.id = -1 - k  # negative ids never collide with targets
    intruders += find_intruders(rgb, valid, model, ascale)
    return Segmentation(table_depth_mm=table_depth, objects=objs, excluded=excluded,
                        intruders=intruders, align=model, plane=(a, b, c), cables=cables)


# ---- low objects: too flat for the depth threshold, found by colour and confirmed by depth ------------------
# A sweet bag stands 6.5 mm above the table: below MIN_HEIGHT_MM except on its creases, so it came out in pieces or not at
# all. The paper square curls up to several mm too, so depth alone cannot go lower: a low object must also differ from the
# table in colour, and tape and white paper never count. {SCENE2 OBSERVE "BAG HEIGHT P10/50/90 [-0.1, 5.2, 7.3]" "TABLE
# [-2.3, -1.4, -0.3]" "PAPER [-1.6, -0.5, 0.9]"} [CONFIDENCE: CONFIRMED 70% — one real bag; every detection on 106 frames checked]
LOW_OBJECTS = os.environ.get("SORTER_LOW_OBJECTS") == "1"  # off by default (VIAM CENTERAL: behind a flag)
LOW_MIN_MM = 2.5  # above the table's own level (the plane fit sits ~1.3 mm off); table noise is ~1 mm
LOW_MAX_MM = 12.0  # taller ones are the normal path's
LOW_DE_MIN = 12.0  # colour distance from the table, Lab with lightness at half weight (grain and sun change lightness)
LOW_MIN_AREA_PX = 3000  # ~23x23 mm at the reference depth
LOW_MIN_WIDTH_MM = 15.0  # a cable on the table is 7-10 mm
LOW_MIN_RECT = 0.6


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8) to CIE Lab."""
    c = rgb.astype(float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def _find_low_objects(objs: list, excluded: list, rgb: np.ndarray, d: np.ndarray, valid: np.ndarray, height: np.ndarray, ascale: float) -> tuple:
    """(objects, excluded) plus flat objects; an existing low piece of one (its creases) is replaced by the whole of it."""
    H, W = d.shape
    q = 2
    rgb2, h2, v2 = rgb[::q, ::q], height[::q, ::q], valid[::q, ::q]
    lab = _lab(rgb2)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    table = v2 & (np.abs(h2) < 4.0)
    tape = (lab[..., 2] < -12.0) & (chroma > 15.0)  # saturated blue
    paper = (lab[..., 0] > 68.0) & (chroma < 16.0)
    ref = table & ~tape & ~paper
    if ref.sum() < 1000:
        return objs, excluded
    level = float(np.median(h2[ref]))
    tone = np.median(lab[ref], axis=0)
    de = np.sqrt((0.5 * (lab[..., 0] - tone[0])) ** 2 + (lab[..., 1] - tone[1]) ** 2 + (lab[..., 2] - tone[2]) ** 2)
    rel = h2 - level
    never = ndimage.binary_dilation(tape | paper, iterations=6)
    others = np.zeros_like(v2)
    for o in objs + [e for e, _ in excluded]:
        if o.height_mm > LOW_MAX_MM:
            others |= o.mask[::q, ::q]
    never |= ndimage.binary_dilation(others, iterations=8)
    fg = v2 & (rel >= LOW_MIN_MM) & (de >= LOW_DE_MIN) & ~never
    fg = ndimage.binary_opening(fg, iterations=1)
    fg = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(fg, 6), iterations=5))[6:-6, 6:-6] & ~never
    labels, n = ndimage.label(fg)
    px_mm = INTRINSICS["fx"] / float(np.median(d[valid]))
    new = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        m2 = labels[sl] == i
        if m2.sum() * q * q < LOW_MIN_AREA_PX * ascale:
            continue
        h = float(np.median(rel[sl][m2 & v2[sl]]))
        if not LOW_MIN_MM <= h <= LOW_MAX_MM:
            continue
        mask = np.zeros((H, W), bool)
        small = np.zeros(v2.shape, bool)
        small[sl] = m2
        up = np.kron(small, np.ones((q, q), bool))[:H, :W]
        mask[:up.shape[0], :up.shape[1]] = up
        vs, us = np.nonzero(mask)
        u0, v0, u1, v1 = int(us.min()), int(vs.min()), int(us.max()), int(vs.max())
        fsl = (slice(v0, v1 + 1), slice(u0, u1 + 1))
        cen, yaw, L, Wd = _geometry(mask[fsl], (u0, v0))
        rf = rect_fill(mask[fsl])
        if Wd / px_mm < LOW_MIN_WIDTH_MM or rf < LOW_MIN_RECT:
            continue
        mu, mv = int((u1 - u0) * CROP_MARGIN) + 4, int((v1 - v0) * CROP_MARGIN) + 4
        e = EDGE_MARGIN_PX
        edge = u0 <= e or v0 <= e or u1 >= W - 1 - e or v1 >= H - 1 - e
        ok = mask & valid
        top = float(np.median(d[ok])) if ok.any() else float(np.median(d[valid]))
        new.append(Obj(id=0, bbox=(u0, v0, u1, v1), centroid_px=cen, yaw_deg=yaw, length_px=L, width_px=Wd, height_mm=h,
                       top_depth_mm=top, mask=mask, crop=rgb[max(0, v0 - mv):min(H, v1 + mv + 1), max(0, u0 - mu):min(W, u1 + mu + 1)].copy(),
                       touches_edge=edge, colour=colour_name(rgb[fsl], mask[fsl])[0], rect_fill=rf, top_centroid_px=cen,
                       colours=colour_fractions(rgb[fsl], mask[fsl]), flat=True))
    if not new:
        return objs, excluded

    def replaced(o):
        return o.height_mm <= LOW_MAX_MM and any((o.mask & f.mask).sum() >= 0.3 * o.mask.sum() for f in new)

    objs2 = [o for o in objs if not replaced(o)]
    excl2 = [(o, why) for o, why in excluded if not replaced(o)]
    for f in new:
        if f.touches_edge:
            excl2.append((f, "touches the image edge"))
        else:
            objs2.append(f)
    for k, o in enumerate(objs2):
        o.id = k
    return objs2, excl2


# ---- cables: a thin strip that joins a blob to the frame is not part of the object -------------------------
# The robot's cable lying against a screwdriver handle made one blob 97 mm wide: refused as too wide. Bodies are what
# survives removing everything narrower than SPLIT_BRIDGE_MM; a thin strip left over is a cable only if it runs to the
# image border or is longer than any part of an object. A screwdriver's shaft (thin, ends in the open) stays.
# {VIAM CENTERAL 2026-10-30 "TOO_WIDE 97 MM, MASK 把把手和相邻的线缆连成一块"} {SCAN 071 "BLACK 167X51 E" -> "97X53" (MOUSE + ITS
# WIRE), 009 "BLUE 91X67 E" -> "59X35" (BLOCK + SHAFT)} [CONFIDENCE: CONFIRMED 75% — real blobs by eye; not yet that screwdriver frame]
SPLIT_BLOBS = os.environ.get("SORTER_SPLIT_BLOBS") == "1"  # off by default (VIAM CENTERAL: behind a flag)
SPLIT_BRIDGE_MM = 14.0  # cable ~10 mm, nearer the camera when lifted; the thinnest graspable bodies are ~25 mm
SPLIT_EDGE_MIN_MM = 30.0  # a strip this long reaching the image border is a cable
SPLIT_LONG_MM = 160.0  # a strip this long is a cable wherever it ends (the longest shaft here is ~130 mm)
SPLIT_CONTACT_MAX = 0.5  # shared edge with the body / strip length: a wedge shaved off a cut object lies along it
SPLIT_BODY_MIN_MM = 18.0  # the part kept must be this thick: graspable bodies are 25+, see-through fragments 15-17 (scene2)
SPLIT_BORDER_MAX = 0.5  # stretch of image border it touches / strip length
SPLIT_WIDE_MM = 60.0  # a blob this wide cannot be gripped as it is, so it is split even if still cut by the frame


def _split_cables(lab: np.ndarray, n: int, px_mm: float) -> tuple:
    """
    (labels, n, cables): cable strips removed from a blob, one label per connected part of what remains.

    A blob changes only if it gains from it: its main part stops touching the frame (it becomes a whole target) or
    it was too wide to grasp. The opening also shaves wedges off objects cut by the frame; those lie along the body
    (contact as long as the strip), a cable leaves it end-on, so contact / length must be small.
    """
    H, W = lab.shape
    r = SPLIT_BRIDGE_MM / 2 * px_mm
    cables = []
    lab = lab.copy()
    e = EDGE_MARGIN_PX

    def at_edge(v0, u0, v1, u1):
        return v0 <= e or u0 <= e or v1 >= H - 1 - e or u1 >= W - 1 - e

    def width_mm(mask):
        vs, us = np.nonzero(mask)
        pts = np.column_stack([us, vs]).astype(float)
        minor = np.linalg.eigh(np.cov(pts.T))[1][:, 0]
        return float(np.ptp(pts @ minor)) / px_mm

    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < SPLIT_EDGE_MIN_MM * px_mm:
            continue
        # One empty pixel all round, so a mask on the image border still erodes from that side
        m = np.pad(lab[sl] == i, 1)
        if m.sum() < 4:
            continue
        dt = ndimage.distance_transform_edt(m)
        body = (ndimage.distance_transform_edt(dt <= r) <= r) & m  # opening: grow the core back inside the mask
        strips, k = ndimage.label(m & ~body, structure=np.ones((3, 3), bool))
        cut, found = np.zeros_like(m), []
        for j, ss in enumerate(ndimage.find_objects(strips), start=1):
            one = strips == j
            vs, us = np.nonzero(one[ss])
            if len(vs) < 3:
                continue
            pts = np.column_stack([us, vs]).astype(float)
            length = float(np.ptp(pts @ np.linalg.eigh(np.cov(pts.T))[1][:, -1])) / px_mm
            # Back to image coordinates (the pad added 1) to see whether it reaches the frame
            v0, v1 = sl[0].start + ss[0].start - 1, sl[0].start + ss[0].stop - 2
            u0, u1 = sl[1].start + ss[1].start - 1, sl[1].start + ss[1].stop - 2
            if not ((at_edge(v0, u0, v1, u1) and length >= SPLIT_EDGE_MIN_MM) or length >= SPLIT_LONG_MM):
                continue
            # A cable crosses the frame (its own width of contact); an object the frame cuts runs along it, and what
            # remained would pass for a whole little object (scene2 observe: the handle cut by the right border)
            uu, vv = us + sl[1].start + ss[1].start - 1, vs + sl[0].start + ss[0].start - 1
            along = max(len(np.unique(uu[(vv <= e) | (vv >= H - 1 - e)])), len(np.unique(vv[(uu <= e) | (uu >= W - 1 - e)])))
            if along / px_mm / max(length, 1.0) > SPLIT_BORDER_MAX:
                continue
            contact = float((ndimage.binary_dilation(one, iterations=2) & body).sum()) / 2.0 / px_mm  # mm of shared edge
            if contact / max(length, 1.0) > SPLIT_CONTACT_MAX:
                continue
            cut |= one
            found.append({"bbox": (u0, v0, u1, v1), "length_mm": round(length, 1), "width_mm": round(len(vs) / px_mm ** 2 / max(length, 1.0), 1)})
        if not found or not body.any():
            continue  # (a blob with no body at all may be a thin real object: only the frame cuts it, it stays excluded)
        rest, parts = ndimage.label(m & ~cut)
        sizes = ndimage.sum(m, rest, range(1, parts + 1))
        main = rest == 1 + int(np.argmax(sizes))
        vs, us = np.nonzero(main)
        if 2 * ndimage.distance_transform_edt(main).max() / px_mm < SPLIT_BODY_MIN_MM:
            continue  # nothing thick is left: a sliver of something the frame cuts, still useful as a partial sighting
        whole = not at_edge(sl[0].start + vs.min() - 1, sl[1].start + us.min() - 1, sl[0].start + vs.max() - 1, sl[1].start + us.max() - 1)
        if not whole and width_mm(m) <= SPLIT_WIDE_MM:
            continue  # still cut by the frame and not too wide: nothing gained
        cables += found
        lab[sl] = np.where(lab[sl] == i, 0, lab[sl])
        for p in range(1, parts + 1):
            piece = rest == p
            if 2 * ndimage.distance_transform_edt(piece).max() / px_mm < SPLIT_BRIDGE_MM:
                continue  # a thin leftover between strips: more cable, not a new target
            n += 1
            lab[sl] = np.where(piece[1:-1, 1:-1], n, lab[sl])
    return lab, n, cables


# ---- see-through objects: fragments of one object joined by its depth-less body ---------------------------
# A clear bottle has depth only on its cap and label; the clear body reads no depth at all, and that invalid region is
# the object's whole silhouette. Thin invalid rims (parallax shadows, ~20 px) and the table edge / floor (touching the
# image border) are not bodies. {BOTTLE CENTERED.NPZ "1 GREY H=45 BBOX=(633, 85, 706, 156)" "2 GREY H=63 BBOX=(587, 222,
# 774, 385)"; DEPTH_VIS: INVALID REGION = BOTTLE SILHOUETTE} [CONFIDENCE: SINGLE-SRC 70% — one real see-through object]
MERGE_FRAGMENTS = os.environ.get("SORTER_MERGE_FRAGMENTS") == "1"  # off by default (VIAM CENTERAL: behind a flag)
MERGE_CORE_R = 4  # opening radius on the quarter-resolution grid (16 px): keeps only thick depth-less bodies
MERGE_CORE_MIN_PX = 6000  # the body's own area, full-resolution pixels at the 380 mm reference depth
# Share of a fragment's surroundings (8 px ring) with no depth: bottle cap 0.65, label 0.84; the wrapper beside it 0.45
# (its parallax rim), a cable 0.15. {MERGE DEBUG "FRAG (633, 85, 706, 156) … INVALID SHARE 0.65"}
# [CONFIDENCE: SINGLE-SRC 65% — one see-through object measured]
MERGE_RING_MIN = 0.55
MERGE_RING_TOUCH = 0.05  # ... and this share must be a thick body, not just rims
MERGE_SOLIDITY_MIN = 0.85  # merged mask / its convex hull: one object is nearly convex, two joined ones are not


def _merge_fragments(objs: list, excluded: list, rgb: np.ndarray, valid: np.ndarray, ascale: float, shape: tuple) -> tuple:
    """
    (objects, excluded) with the fragments of each see-through object replaced by one Obj covering all of it.

    Thick depth-less bodies and the fragments they surround form a graph: a label splits a bottle's body in two, and
    cap - body - label - body is one connected group. A fragment joins only if most of its surroundings have no depth
    (a neighbouring opaque object touches a body on one side only).
    """
    from scipy.spatial import ConvexHull

    H, W = shape
    q = 4
    inv = ~valid[::q, ::q]
    disk = np.hypot(*np.mgrid[-MERGE_CORE_R:MERGE_CORE_R + 1, -MERGE_CORE_R:MERGE_CORE_R + 1]) <= MERGE_CORE_R
    lab, n = ndimage.label(ndimage.binary_opening(inv, structure=disk))
    bodies = []
    for i in range(1, n + 1):
        core = lab == i
        if core.sum() * q * q < MERGE_CORE_MIN_PX * ascale:
            continue
        if core[0].any() or core[-1].any() or core[:, 0].any() or core[:, -1].any():
            continue  # table edge, floor, or something cut by the frame: not a body we can see whole
        # The body: the core plus the invalid pixels the opening shaved off it, back on the full grid
        grown = ndimage.binary_dilation(core, iterations=MERGE_CORE_R + 1, mask=inv)
        body = np.kron(grown, np.ones((q, q), bool))[:H, :W] & ~valid
        if skin_fraction(rgb, body) < 0.5:  # a hand is find_intruders' business
            bodies.append(body)
    if not bodies:
        return objs, excluded
    frags = [(o, None) for o in objs] + list(excluded)
    # Union-find over bodies (0..b-1) and fragments (b..)
    b = len(bodies)
    parent = list(range(b + len(frags)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    joined = set()
    for k, (o, _) in enumerate(frags):
        u0, v0, u1, v1 = o.bbox
        sl = (slice(max(v0 - 12, 0), min(v1 + 13, H)), slice(max(u0 - 12, 0), min(u1 + 13, W)))
        m = o.mask[sl]
        ring = ndimage.binary_dilation(m, iterations=8) & ~m
        if not ring.any() or (~valid[sl])[ring].mean() < MERGE_RING_MIN:
            continue
        for j, body in enumerate(bodies):
            if body[sl][ring].mean() >= MERGE_RING_TOUCH:
                parent[find(b + k)] = find(j)
                joined.add(k)
    groups = {}
    for j in range(b):
        groups.setdefault(find(j), [[], []])[0].append(j)
    for k in joined:
        groups.setdefault(find(b + k), [[], []])[1].append(k)
    used, merged = set(), []
    for body_ids, parts in groups.values():
        if not parts:
            continue  # no opaque piece: nothing to grip, and it may be a reflection or a dark patch
        union = np.zeros((H, W), bool)
        for j in body_ids:
            union |= bodies[j]
        for k in parts:
            union |= frags[k][0].mask
        union = ndimage.binary_fill_holes(ndimage.binary_closing(union, iterations=6))
        vs, us = np.nonzero(union & ~ndimage.binary_erosion(union))
        if union.sum() / ConvexHull(np.column_stack([us, vs]).astype(float)).volume < MERGE_SOLIDITY_MIN:
            continue
        o = _merged_obj(union, [frags[k] for k in parts], rgb, shape)
        used.update(parts)
        merged.append((o, "touches the image edge" if o.touches_edge else None))
    if not merged:
        return objs, excluded
    keep = [f for k, f in enumerate(frags) if k not in used] + merged
    objs2, excl2 = [o for o, why in keep if why is None], [(o, why) for o, why in keep if why is not None]
    for k, o in enumerate(objs2):
        o.id = k
    return objs2, excl2


# A standing cylinder's side can segment apart from its top (076: a 19x8 mm sliver beside the blue cylinder). A known name
# is matched by label, but for an unknown one the sliver's crop may beat the object and the jaws go for the sliver.
# {OPENVOCAB "BLUE CYLINDER@076: WRONG (PICKED CYAN H=14)"; 076 "FRAGMENT 19X8 MM H=14 AREA_PX=913"}
# [CONFIDENCE: INFERRED 70% — every absorption on the 89 scan frames checked by eye]
MERGE_SLIVER_MAX_PX = 2500  # at most this big (at the 380 mm reference depth; a 30 mm cube is ~5000 px)
# ... and this close to the larger object. On 89 frames the distances are 6, 10, 18, 22, 27 px (shaft to handle, cylinder
# side to top, cup pieces: all one object by eye), then 70 and more. {SLIVERS_DIST "27.3PX 044 GREY" "70.0PX 000 GREY"}
MERGE_SLIVER_GAP_PX = 30
MERGE_SLIVER_RATIO = 3.0  # which is at least this many times its area


def _absorb_slivers(objs: list, excluded: list, ascale: float, shape: tuple) -> list:
    """Objects minus small pieces of a larger, taller neighbour of a similar colour (mask and bbox join the neighbour's)."""
    H, W = shape
    others = objs + [o for o, _ in excluded]
    gone = set()
    for o in sorted(objs, key=lambda x: x.mask.sum()):
        area = int(o.mask.sum())
        if area > MERGE_SLIVER_MAX_PX * ascale:
            continue
        u0, v0, u1, v1 = o.bbox
        g = MERGE_SLIVER_GAP_PX
        sl = (slice(max(v0 - g, 0), min(v1 + g + 1, H)), slice(max(u0 - g, 0), min(u1 + g + 1, W)))
        near = ndimage.binary_dilation(o.mask[sl], iterations=g)
        for b in others:
            if b is o or id(b) in gone or b.mask.sum() < MERGE_SLIVER_RATIO * area or b.height_mm <= o.height_mm:
                continue
            if not (_colour_alike(o.colour, b.colour) and b.mask[sl][near].any()):
                continue
            b.mask = b.mask | o.mask
            b.bbox = (min(b.bbox[0], u0), min(b.bbox[1], v0), max(b.bbox[2], u1), max(b.bbox[3], v1))
            b.n_fragments += o.n_fragments
            gone.add(id(o))
            break
    return [o for k, o in enumerate(o for o in objs if id(o) not in gone) if not setattr(o, "id", k)]


def _colour_alike(a: str, b: str) -> bool:
    """Same hue family, allowing the neighbours the shading of one object spans (the blue cylinder: top blue, side cyan)."""
    return a == b or {a, b} in ({"blue", "cyan"}, {"cyan", "green"}, {"yellow", "orange"}, {"orange", "red"}, {"grey", "white"}, {"grey", "black"})


def _merged_obj(mask: np.ndarray, parts: list, rgb: np.ndarray, shape: tuple) -> Obj:
    """One Obj for the whole silhouette: geometry and crop of all of it, height of its highest piece, grip on an opaque piece."""
    H, W = shape
    vs, us = np.nonzero(mask)
    u0, v0, u1, v1 = int(us.min()), int(vs.min()), int(us.max()), int(vs.max())
    mu, mv = int((u1 - u0) * CROP_MARGIN) + 4, int((v1 - v0) * CROP_MARGIN) + 4
    sl = (slice(v0, v1 + 1), slice(u0, u1 + 1))
    cen, yaw, L, Wd = _geometry(mask[sl], (u0, v0))
    pieces = [o for o, _ in parts]
    grip = min(pieces, key=lambda o: math.dist(o.top_centroid_px or o.centroid_px, cen))
    e = EDGE_MARGIN_PX
    return Obj(id=0, bbox=(u0, v0, u1, v1), centroid_px=cen, yaw_deg=yaw, length_px=L, width_px=Wd,
               height_mm=max(o.height_mm for o in pieces), top_depth_mm=min(o.top_depth_mm for o in pieces), mask=mask,
               crop=rgb[max(0, v0 - mv):min(H, v1 + mv + 1), max(0, u0 - mu):min(W, u1 + mu + 1)].copy(),
               touches_edge=u0 <= e or v0 <= e or u1 >= W - 1 - e or v1 >= H - 1 - e,
               # top_centroid_px is what grasp_px() and the table map use: the grip, since the body has no depth to stand on
               colour=colour_name(rgb[sl], mask[sl])[0], rect_fill=rect_fill(mask[sl]),
               top_centroid_px=tuple(grip.top_centroid_px or grip.centroid_px),
               colours=colour_fractions(rgb[sl], mask[sl]), grip_px=tuple(grip.top_centroid_px or grip.centroid_px),
               grip_depth_mm=grip.top_depth_mm, n_fragments=len(pieces))


# ---- pixel <-> table geometry (wrist camera looking straight down) ----------------------------------------
# Measured with get_properties() on sfhack4 2026-10-29; a frame's own "intrinsics" overrides these
INTRINSICS = {"fx": 909.93, "fy": 909.36, "cx": 645.48, "cy": 365.37}
# Jaw (flange) axis relative to the camera axis, camera coordinates, from the hand-measured fingertip pixel
# (864, 685) at 137 mm: ((864-645.5)*137/909.9, (685-365.4)*137/909.4). Config cam offset is 3-4 cm off.
# {PICK_PLACE_AUTO.PY "JAW_TIP_PX = (864.0, 685.0)" "TIP_BELOW_CAM = 155.0 - CAM_T[2]"}
# [CONFIDENCE: CONFIRMED 80% — independent check: AGENT1's own pixel->world conversion agrees within 1-15 mm]
JAW_IN_CAM_MM = (32.9, 48.1)
# Image +u is world -y and image +v is world -x at theta 0 (arm +y moved the scene +u, arm +x moved it +v).
# Wrist rotation direction checked on the arm: with -1 the four objects of theta_test 079 (theta -90) land on their
# 077 (theta 0) positions up to a constant offset; with +1 they scatter by 150-300 mm.
# {079.JSON "THETA": -90.03} {THETA_CHECK SIGN -1 "GREEN (218, 8)" VS 077 "GREEN (198, 33)"; SIGN +1 "GREEN (220, -286)"}
# [CONFIDENCE: CONFIRMED 90% — one theta pair, 4 objects; the +90 frame 078 had no fully visible shared object]
THETA_SIGN = -1.0
# The wrist turns about the pose point, not about the jaw axis that world_xy() maps to the flange xy: the jaw axis
# sits TOOL_AXIS_MM from the rotation axis (world frame at theta 0). Fitted so 079 matches 077 (residuals <= 2.5 mm).
# Zero effect at theta 0, so the map / look / travel convention (map xy = theta-0 flange xy over the point) is kept.
# Independent check at theta +90: the blue block in 078 lands at (417, -173); the theta-0 frame 080 two minutes
# later sees it at (419, -175) (uncorrected 078 said (442, -154)).
# {THETA_FIT "J = [21.74 -3.25]" "RED RESID [-2.5 0.]"} {080 "EDGE BLUE H=30 … XY=(419,-175)"}
# [CONFIDENCE: CONFIRMED 85% — fitted on theta -90, confirmed within 3 mm at theta +90 on an edge-cut sighting]
TOOL_AXIS_MM = (21.7, -3.3)


def _rot(t: float, x: float, y: float) -> tuple:
    """Rotate (x, y) by t radians (counter-clockwise in the arm base frame)."""
    return x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)


def world_xy(pose: dict, intr: dict, u: float, v: float, depth_mm: float) -> tuple:
    """
    Table xy (arm base frame, mm) of the point seen at pixel (u, v) at camera distance depth_mm.

    Pinhole back-projection to camera coordinates, then the offset from the jaw axis rotated into the
    world; a point exactly on the jaw axis maps to the flange xy at theta 0. At other wrist angles the
    offset is taken about the real rotation axis (TOOL_AXIS_MM), so every angle agrees with theta 0.
    """
    xc = (u - intr["cx"]) * depth_mm / intr["fx"]
    yc = (v - intr["cy"]) * depth_mm / intr["fy"]
    dx, dy = JAW_IN_CAM_MM[1] - yc, JAW_IN_CAM_MM[0] - xc  # world offset at theta 0
    t = math.radians(THETA_SIGN * float(pose.get("theta", 0.0)))
    # Offset from the rotation axis turns with the wrist; subtracting TOOL_AXIS_MM keeps theta 0 unchanged
    rx, ry = _rot(t, dx + TOOL_AXIS_MM[0], dy + TOOL_AXIS_MM[1])
    return pose["x"] + rx - TOOL_AXIS_MM[0], pose["y"] + ry - TOOL_AXIS_MM[1]


def project_xy(pose: dict, intr: dict, table_xy, depth_mm: float) -> tuple:
    """Pixel (u, v) where a table point appears at camera distance depth_mm: the inverse of world_xy()."""
    t = math.radians(THETA_SIGN * float(pose.get("theta", 0.0)))
    # Undo world_xy(): offset from the rotation axis, rotated back by -t, then back to the jaw-axis offset
    rx, ry = _rot(-t, table_xy[0] - pose["x"] + TOOL_AXIS_MM[0], table_xy[1] - pose["y"] + TOOL_AXIS_MM[1])
    dx, dy = rx - TOOL_AXIS_MM[0], ry - TOOL_AXIS_MM[1]
    yc, xc = JAW_IN_CAM_MM[1] - dx, JAW_IN_CAM_MM[0] - dy
    return intr["cx"] + xc * intr["fx"] / depth_mm, intr["cy"] + yc * intr["fy"] / depth_mm


LOOK_MIN_REACH_MM = 150.0  # the executor refuses flange xy closer than 120 mm to the base; keep a margin


def look_flange_xy(table_xy: tuple) -> tuple:
    """
    Flange xy (theta 0) that puts the table point `table_xy` near the image centre.

    The image centre sees the point JAW_IN_CAM_MM away from the jaw axis, i.e. flange + (48.1, 32.9) at theta 0.
    Too close to the base (the green cube needed (100, 78), 127 mm) is pushed radially out to LOOK_MIN_REACH_MM;
    at z 400 the view spans ~+-150 mm in x, so the point stays in frame (AGENT1 shot the cube from (150, 110)).
    """
    fx, fy = table_xy[0] - JAW_IN_CAM_MM[1], table_xy[1] - JAW_IN_CAM_MM[0]
    r = math.hypot(fx, fy)
    if 0 < r < LOOK_MIN_REACH_MM:
        fx, fy = fx * LOOK_MIN_REACH_MM / r, fy * LOOK_MIN_REACH_MM / r
    return fx, fy


SQUARE_MIN_AREA_PX = 15000  # the white interior is ~290x300 px at the observe pose; scaled with depth like objects


def find_drop_square(rgb: np.ndarray, depth: np.ndarray, plane: tuple = None) -> dict:
    """
    The drop zone: white interior enclosed by the blue tape. {centroid_px, bbox, area, depth_mm[, partial, cut]} or None.

    Blue tape by hue/saturation, as measured by AGENT1 (tape reads ~(0,64,116), PIL hue ~147).
    {PICK_PLACE_AUTO.PY "(HSV[..., 0] >= 130) & (HSV[..., 0] <= 165) & (HSV[..., 1] > 120) & (HSV[..., 2] > 50)"}
    [CONFIDENCE: CONFIRMED 80% — AGENT1's servo uses the same mask on this paper square]
    The hole must be mostly white paper (low saturation, bright). A hole fully in view wins; otherwise a border-cut
    one ringed by tape comes back with partial=True (visible part only).
    """
    from PIL import Image

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    d = depth.astype(float)
    a, b, c = plane or fit_table(d)  # reuse segment_full's plane when the caller has it
    H, W = d.shape
    min_area = SQUARE_MIN_AREA_PX * area_scale(float(a * W / 2 + b * H / 2 + c))
    # Strict tape mask first; one looser retry (lower saturation/brightness, wider hue) because the real z=500 sweep
    # found no square although geometry and a simulated higher view both say it was fully in view: exposure at a
    # different height is the remaining suspect. The hole checks (enclosed, large, mostly white) stay the same.
    masks = [(hsv[..., 0] >= hlo) & (hsv[..., 0] <= hhi) & (hsv[..., 1] > smin) & (hsv[..., 2] > vmin)
             for hlo, hhi, smin, vmin in ((130, 165, 120, 50), (120, 172, 70, 40))]
    # Closing on an edge-padded mask: plain binary_closing erodes 3 px off the image border (border_value 0), which
    # opened the tape where it runs off the frame and joined a border-cut hole to the whole table (080)
    masks = [ndimage.binary_closing(np.pad(m, 4, mode="edge"), iterations=3)[4:-4, 4:-4] for m in masks]
    # Objects are never tape. The silver knife reads 52-69 % "tape" by colour, so lying in the square it cut the
    # interior in two and only the part above it was found (live observe frame, 16 mm off). First every raised
    # pixel is "not tape" (an object on the paper joins the interior); if that finds nothing, raised pixels are
    # walls instead (an object lying across the tape would otherwise join the interior to the table).
    # {KNIFE TAPE SHARE 081 "STRICT 0.69 LOOSE 0.84", 083 "0.61 / 0.83"; LIVE SNAPSHOT "HIT BBOX (336, 16, 742, 199)"}
    # [CONFIDENCE: CONFIRMED 80% — synthetic objects on 076 in tests/test_drop_square.py; no real frame of it yet]
    vv, uu = np.mgrid[0:H, 0:W]
    raised = (d > 0) & (a * uu + b * vv + c - d > MIN_HEIGHT_MM)
    # In the second try the walls also take what is neither paper nor tape: very dark pixels and no depth. The
    # robot's cable hanging in front of the lens crossed the square at the place look pose; depth sees past it
    # (375 mm, the table) and it is not blue, so it joined the interior to the table and no square was found.
    # {SQUARE_CENTERED.NPZ "DARK (CABLE) PIXELS 141694 … DEPTH>0 MEDIAN ON CABLE 375.0" "PX (500,300) V=27"}
    # [CONFIDENCE: CONFIRMED 85% — that frame now finds the square; no change on the 89 scan frames]
    occluder = raised | (d <= 0) | (hsv[..., 2] < SQUARE_DARK_V)
    # (walls, tape, rim): with occluders as walls a bright patch fenced by objects (008: a transparent cup) also looks
    # enclosed, so there the hole's rim must still be mostly real tape (SQUARE_RIM_TAPE)
    walls = [(m & ~raised, m, None) for m in masks] + [(m | occluder, m, SQUARE_RIM_TAPE) for m in masks]
    for wall, tape, rim in walls:
        hit = _enclosed_white_hole(hsv, wall, min_area, (a, b, c), tape=tape, rim_min=rim)
        if hit:
            return hit
    # Not fully in view: frame 080 (observe pose, after a place pushed the paper ~40 mm) had the hole cut by the
    # top edge, so the delivered yellow block was not "in the square" on the map. Accept a border-cut hole whose
    # in-image rim is tape, flagged partial: its centroid/bbox are only the visible part, never a place target.
    # {SQ_76_80 "080: WHITE INTERIOR V 0-270 (TOUCHES THE TOP EDGE)" VS 076 "V 30-360"}
    # [CONFIDENCE: CONFIRMED 85% — seen on 080/082/084; partial rim check mirrors AGENT1's find_square ring test]
    # Strict tape mask only: with the loose one, transparent cups (bluish reflections, paper-like brightness) came
    # back as border-cut squares on 000/016/032/056 (masks[0]: strict, objects not tape; masks[2]: strict, objects walls)
    for wall, tape, _ in (walls[0], walls[2]):
        hit = _enclosed_white_hole(hsv, wall, min_area * SQUARE_PARTIAL_AREA_FRAC, (a, b, c), border_cut=True, tape=tape)
        if hit:
            return hit
    return None


# Paper reads light blue to this camera: PIL saturation median 52-60 (010-076), 66 (080), 103 (082), so the old
# "S < 60" white test failed whole exposures. Tape is S > 120 by the mask itself; brightness tells paper from wood.
# {PAPER S P50: 076 "60", 080 "69", 082 "103"; WOOD "V P10/50 [86 130]" … "[117 128]" VS V_WHITE 150}
# [CONFIDENCE: CONFIRMED 85% — measured on 9 frames; wood V>v_white share 0.00-0.09, paper 0.95-1.00]
SQUARE_WHITE_S = 120
# A hole whose bbox comes this close to the image border is treated as cut there (partial)
# {060 "BBOX (376, 427, 818, 718)" "ROWS 700-719 BLUE FRAC … [0.72 … 0.88]"} [CONFIDENCE: CONFIRMED 85% — 060 vs 061/084]
SQUARE_EDGE_PX = 6
SQUARE_DARK_V = 60  # PIL brightness below this is an occluder (cable, fingers), never paper (V ~200) or tape (~110)
SQUARE_PARTIAL_AREA_FRAC = 0.4  # a border-cut hole may show less than the full interior
# Border-cut hole checks, from 5 real (010/015/019/043/050) vs 7 fake partials (cups, tape strips, a tape roll):
# real rim 1.00 and rect_fill 0.92-0.99; fakes rect_fill 0.67-0.89 or paper-like V on < 80 % of the region
# [CONFIDENCE: CONFIRMED 80% — 12 labelled cases, all separated; thresholds sit in the gap]
SQUARE_RIM_BLUE = 0.9  # share of the in-image rim that must be tape
# Hole found with occluders as walls: its rim must still be at least this much tape. Real squares 0.78 (the cable
# frame) to 1.0; the cup patch on 008 0.0. {RIM SHARE "SQUARE_CENTERED … 0.78" "008 … 0.0" "076 … 1.0" "020 … 0.84"}
# [CONFIDENCE: CONFIRMED 80% — 6 frames measured; 0.5 sits in the wide gap]
SQUARE_RIM_TAPE = 0.5
SQUARE_PARTIAL_RECT = 0.9  # rect_fill of the hole-filled region: a clipped rectangle stays a rectangle
SQUARE_PARTIAL_WHITE_V = 0.8  # share of bright (paper) pixels; 0.6 for a fully seen hole


def _enclosed_white_hole(hsv: np.ndarray, blue: np.ndarray, min_area: float, plane: tuple, border_cut: bool = False,
                         tape: np.ndarray = None, rim_min: float = None):
    """
    Largest region not in the wall mask `blue`, >= min_area and mostly white paper; or None.

    border_cut False: the region must not touch the image border (the whole hole in view); still flagged partial
    when its bbox is within SQUARE_EDGE_PX of the border. rim_min: share of its rim that must be `tape` (None: no check).
    border_cut True: only regions touching the border, and SQUARE_RIM_BLUE of the rest of their rim must be `tape`;
    the result carries "partial": True and "cut" (the image sides it runs off). tape None: the walls are the tape.
    """
    a, b, c = plane
    lab, n = ndimage.label(~blue)
    if not n:
        return None
    H, W = blue.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    # "White" relative to this frame's exposure: at 60 % brightness the paper fell below a fixed V>150 and the square
    # was lost at every height in simulation; capped at 150 so normal exposures behave exactly as before
    v_white = min(150.0, 0.8 * float(np.percentile(hsv[::4, ::4, 2], 95)))
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or (i in border) != border_cut:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_area or (best and area <= best["area"]):
            continue
        bright = SQUARE_PARTIAL_WHITE_V if border_cut else 0.6
        white = (hsv[sl][..., 1][m] < SQUARE_WHITE_S).mean() > 0.6 and (hsv[sl][..., 2][m] > v_white).mean() > bright
        if not white:
            continue
        vs, us = np.nonzero(m)
        bbox = (int(sl[1].start + us.min()), int(sl[0].start + vs.min()), int(sl[1].start + us.max()), int(sl[0].start + vs.max()))
        e = SQUARE_EDGE_PX
        cut = [s for s, hit in (("top", bbox[1] <= e), ("bottom", bbox[3] >= H - 1 - e),
                                ("left", bbox[0] <= e), ("right", bbox[2] >= W - 1 - e)) if hit]
        # The interior (~11x18 cm) is smaller than the view, so it can run off one side or one corner only;
        # a region off opposite sides is the table/paper around the tape (032-039 false positives)
        if border_cut and (len(cut) > 2 or {"top", "bottom"} <= set(cut) or {"left", "right"} <= set(cut)):
            continue
        need = SQUARE_RIM_BLUE if border_cut else rim_min
        if need is not None:
            # Outer rim inside the image (10 px ring around the hole-filled region, slice padded so the ring is
            # not clipped) must be mostly tape; holes are filled so objects lying in the region do not count as rim
            pad = (slice(max(sl[0].start - 12, 0), min(sl[0].stop + 12, H)), slice(max(sl[1].start - 12, 0), min(sl[1].stop + 12, W)))
            mp = ndimage.binary_fill_holes(lab[pad] == i)
            band = ndimage.binary_dilation(mp, iterations=10) & ~mp
            ring = (blue if tape is None else tape)[pad][band]
            if not band.any() or ring.mean() < need or (border_cut and rect_fill(mp) < SQUARE_PARTIAL_RECT):
                continue
        u, v = sl[1].start + us.mean(), sl[0].start + vs.mean()
        best = {"centroid_px": (float(u), float(v)), "area": area, "bbox": bbox,
                "depth_mm": float(a * u + b * v + c), "_i": i, "_sl": sl}  # flat paper: the table plane at its centre
        if cut:
            # Border-cut, or "enclosed" only by a few edge rows: on 060 the vignetted bottom rows of the paper read as
            # tape, so a hole running off the bottom looked whole and its centroid sat 20 mm off on the map
            best.update(partial=True, cut=cut)
    if best is not None:
        best["hull_px"], best["hull_cut"] = _hole_hull(lab, best.pop("_i"), best.pop("_sl"), plane)
    return best


def _hole_hull(lab: np.ndarray, i: int, sl: tuple, plane: tuple) -> tuple:
    """
    Convex hull of the hole (objects on the paper filled in): ([[u, v, depth_mm], ...] counter-clockwise,
    [cut per edge i -> i+1]). An edge is cut when both ends lie on the same image side (within SQUARE_EDGE_PX):
    that side of the hull is the frame, not tape. table_map fits the full rectangle from the other edges.
    """
    from scipy.spatial import ConvexHull

    a, b, c = plane
    H, W = lab.shape
    pad = (slice(max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)), slice(max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)))
    m = ndimage.binary_fill_holes(lab[pad] == i)
    vs, us = np.nonzero(m & ~ndimage.binary_erosion(m))  # the rim is enough for the hull
    pts = np.column_stack([us + pad[1].start, vs + pad[0].start]).astype(float)
    hp = pts[ConvexHull(pts).vertices]
    e = SQUARE_EDGE_PX
    side = [{s for s, hit in (("top", v <= e), ("bottom", v >= H - 1 - e), ("left", u <= e), ("right", u >= W - 1 - e))
             if hit} for u, v in hp]
    cut = [bool(side[k] & side[(k + 1) % len(hp)]) for k in range(len(hp))]
    return [[float(u), float(v), float(a * u + b * v + c)] for u, v in hp], cut


def segment(rgb: np.ndarray, depth: np.ndarray) -> tuple:
    """(table_depth_mm, objects), the EXEC_PLAN contract. Use segment_full() for intruders and alignment."""
    s = segment_full(rgb, depth)
    return s.table_depth_mm, s.objects


HUE_BINS = [(0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 165, "green"),
            (165, 200, "cyan"), (200, 260, "blue"), (260, 330, "purple"), (330, 361, "red")]


def _hue_sat(rgb: np.ndarray, mask: np.ndarray) -> tuple:
    """(hue degrees, saturated?, max channel) per masked pixel, 0-1 scale."""
    px = rgb[mask].astype(float) / 255.0
    mx, mn = px.max(1), px.min(1)
    d = np.maximum(mx - mn, 1e-6)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    # Dark pixels carry a faint hue from white balance; a black charger read "blue" before this
    return hue, (s > 0.35) & (mx > 0.3), mx


def colour_fractions(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    Share of the mask in each saturated colour family (only families >= 2%).

    WHY: the dominant colour of both screwdrivers is "black" (handles), so only the secondary colour tells the
    blue one (blue 0.33-0.38) from the red one (red 0.31, blue 0.00); stray hue on black objects stays <= 0.06.
    """
    if not mask.any():
        return {}
    hue, sat, _ = _hue_sat(rgb, mask)
    out = {}
    for lo, hi, nm in HUE_BINS:
        out[nm] = out.get(nm, 0.0) + float((sat & (hue >= lo) & (hue < hi)).mean())
    return {k: round(v, 3) for k, v in out.items() if v >= 0.02}


def colour_name(rgb: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Dominant colour word of the masked pixels and the fraction of pixels that have it.

    Saturated pixels vote by hue (degrees); low saturation splits into white / grey / black by value.
    Hue bins chosen by eye on sfhack4 frames (the yellow block is ~55 deg, the blue tape ~210 deg).
    """
    if not mask.any():
        return "unknown", 0.0
    hue, sat, mx = _hue_sat(rgb, mask)
    names = np.full(len(hue), "grey", dtype=object)
    names[~sat & (mx > 0.8)] = "white"
    names[mx <= 0.3] = "black"
    for lo, hi, nm in HUE_BINS:
        names[sat & (hue >= lo) & (hue < hi)] = nm
    vals, counts = np.unique(names, return_counts=True)
    k = int(np.argmax(counts))
    return str(vals[k]), float(counts[k] / len(hue))
