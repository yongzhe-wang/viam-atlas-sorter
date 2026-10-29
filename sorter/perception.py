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


@dataclass
class Segmentation:
    table_depth_mm: float  # fitted plane at the image centre
    objects: list  # fully visible, inside the workspace: the only things the arm may target
    excluded: list  # [(Obj, reason)]: cut by the image border or outside the workspace (debug / dashboard)
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
    # No fingertip mask for objects: the fingers are ~140 mm from the camera, inside the depth blind range, so they
    # never become objects; masking them erased a real 30 mm cube that a z=500 sweep saw in that corner
    lab_c = to_colour(lab_d, model)
    H, W = d.shape
    table_depth = float(a * W / 2 + b * H / 2 + c)
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
    for k, (o, _) in enumerate(excluded):
        o.id = -1 - k  # negative ids never collide with targets
    intruders += find_intruders(rgb, valid, model, ascale)
    return Segmentation(table_depth_mm=table_depth, objects=objs, excluded=excluded,
                        intruders=intruders, align=model, plane=(a, b, c))


# ---- pixel <-> table geometry (wrist camera looking straight down) ----------------------------------------
# Measured with get_properties() on sfhack4 2026-10-29; a frame's own "intrinsics" overrides these
INTRINSICS = {"fx": 909.93, "fy": 909.36, "cx": 645.48, "cy": 365.37}
# Jaw (flange) axis relative to the camera axis, camera coordinates, from the hand-measured fingertip pixel
# (864, 685) at 137 mm: ((864-645.5)*137/909.9, (685-365.4)*137/909.4). Config cam offset is 3-4 cm off.
# {PICK_PLACE_AUTO.PY "JAW_TIP_PX = (864.0, 685.0)" "TIP_BELOW_CAM = 155.0 - CAM_T[2]"}
# [CONFIDENCE: CONFIRMED 80% — independent check: AGENT1's own pixel->world conversion agrees within 1-15 mm]
JAW_IN_CAM_MM = (32.9, 48.1)
# Image +u is world -y and image +v is world -x at theta 0 (arm +y moved the scene +u, arm +x moved it +v).
# Wrist rotation direction is not verified on hardware: use theta 0 for surveys and look poses.
THETA_SIGN = -1.0


def world_xy(pose: dict, intr: dict, u: float, v: float, depth_mm: float) -> tuple:
    """
    Table xy (arm base frame, mm) of the point seen at pixel (u, v) at camera distance depth_mm.

    Pinhole back-projection to camera coordinates, then the offset from the jaw axis rotated into the
    world; a point exactly on the jaw axis maps to the flange xy.
    """
    xc = (u - intr["cx"]) * depth_mm / intr["fx"]
    yc = (v - intr["cy"]) * depth_mm / intr["fy"]
    dx, dy = JAW_IN_CAM_MM[1] - yc, JAW_IN_CAM_MM[0] - xc  # world offset at theta 0
    t = math.radians(THETA_SIGN * float(pose.get("theta", 0.0)))
    return (pose["x"] + dx * math.cos(t) - dy * math.sin(t), pose["y"] + dx * math.sin(t) + dy * math.cos(t))


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


def find_drop_square(rgb: np.ndarray, depth: np.ndarray) -> dict:
    """
    The drop zone: white interior enclosed by the blue tape, fully in view. {centroid_px, bbox, area, depth_mm} or None.

    Blue tape by hue/saturation, as measured by AGENT1 (tape reads ~(0,64,116), PIL hue ~147).
    {PICK_PLACE_AUTO.PY "(HSV[..., 0] >= 130) & (HSV[..., 0] <= 165) & (HSV[..., 1] > 120) & (HSV[..., 2] > 50)"}
    [CONFIDENCE: CONFIRMED 80% — AGENT1's servo uses the same mask on this paper square]
    The hole must not touch the image border and must be mostly white paper (low saturation, bright).
    """
    from PIL import Image

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    blue = (hsv[..., 0] >= 130) & (hsv[..., 0] <= 165) & (hsv[..., 1] > 120) & (hsv[..., 2] > 50)
    blue = ndimage.binary_closing(blue, iterations=3)
    lab, n = ndimage.label(~blue)
    if not n:
        return None
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    d = depth.astype(float)
    a, b, c = fit_table(d)
    H, W = d.shape
    min_area = SQUARE_MIN_AREA_PX * area_scale(float(a * W / 2 + b * H / 2 + c))
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or i in border:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_area or (best and area <= best["area"]):
            continue
        white = (hsv[sl][..., 1][m] < 60).mean() > 0.6 and (hsv[sl][..., 2][m] > 150).mean() > 0.6
        if not white:
            continue
        vs, us = np.nonzero(m)
        u, v = sl[1].start + us.mean(), sl[0].start + vs.mean()
        best = {"centroid_px": (float(u), float(v)), "area": area,
                "bbox": (int(sl[1].start + us.min()), int(sl[0].start + vs.min()), int(sl[1].start + us.max()), int(sl[0].start + vs.max())),
                "depth_mm": float(a * u + b * v + c)}  # flat paper: the table plane at its centre
    return best


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
