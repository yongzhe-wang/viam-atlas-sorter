"""
Grasp primitives, chosen from an object's geometry and never its name, so an object nobody has seen gets one too.

WHY: the user: "we need to implement different grasp method for tape and other". Clamping the tape roll's wall crushed
its paper core and slipped; a long object needs its handle; a tall one must not be pushed over.
{VIAM CENTERAL 2026-10-30 "原则：按几何选抓法，不按名字，这样没见过的物体也能用"}
[CONFIDENCE: SINGLE-SRC 80% — the rule is the user's and coordinator's; only "top" and "ring_wall" have run on the arm]

choose_primitives(o) -> candidates, best first ([] = too wide to grasp):
  ring (hole in the mask)            -> ring_inside (if the hole fits the jaws), ring_wall
  tall and small (> 50 mm high)      -> standing, top
  long (length / width > 2.5)        -> handle, centroid
  anything else                      -> top
target_px(primitive, ...) -> where the jaws go in a frame; pick_inside() is the internal expansion grip.
Escalation: a primitive that failed twice gives way to the next candidate. GRASP_PRIMITIVES=1 turns the new ones on;
without it the candidates are the two already proven on the arm (ring_wall, top), exactly the previous behaviour.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402

from sorter import perception  # noqa: E402

MAX_OPEN_MM = 78.0  # jaws open ~84 mm; keep a margin so the fingers clear the sides
LONG_ASPECT = 2.5  # length / width above this is a long object: gripped across its short axis, at its thickest part
TALL_MM = 50.0  # higher than this with a small footprint: gripped at half height so it is not pushed over
TALL_MAX_FOOTPRINT_MM = 80.0
ROUND_RECT_FILL = 0.9  # perception.Obj.rect_fill below this is round (disc ~0.81): the wrist angle does not matter
# Hole diameters the internal grip fits: the closed fingers must enter with room to spare, and the jaws (84 mm at most)
# must reach the wall. The tape roll's core measures 73 mm (ring frames: inner radius 36.7 mm).
INSIDE_MIN_ID_MM, INSIDE_MAX_ID_MM = 45.0, 80.0
EXPAND_MARGIN_MM = 4.0  # the jaws are told to open this far past the wall, so they press on it
FAILS_TO_ESCALATE = 2
ESCALATION_TTL_S = 180.0


def enabled() -> bool:
    """GRASP_PRIMITIVES=1: all primitives. Default off: the internal grip has not run on the arm yet."""
    return os.environ.get("GRASP_PRIMITIVES", "0") == "1"


RING_HOLE_FRAC = 0.2  # share of an object's hole-filled mask that is hole: a ring (tape roll), grasped by its wall


RING_MAX_R_MM = 75.0  # a "ring" wider than this is a merged blob, not a tape roll (outer radius ~59 mm)


def ring_geometry(o, wall_mm: float = None):
    """
    ((cu, cv) ring centre px, radius px of the middle of the wall) if `o` is ring-like, else None.

    wall_mm: the object is known to be a ring (ObjTracker remembers it from the found frame) with this wall radius:
    the centre comes from the hole-filled mask even when this frame's hole looks small. The black tape's servo did
    not converge (65 s): frames that missed the hole switched the target from the wall to the top centroid, ~45 mm
    away, and back.
    """
    u0, v0, u1, v1 = o.bbox
    sub = o.mask[v0:v1 + 1, u0:u1 + 1]
    filled = ndimage.binary_fill_holes(sub)
    hole = filled & ~sub
    if not filled.any() or (wall_mm is None and hole.sum() < RING_HOLE_FRAC * filled.sum()):
        return None
    vs, us = np.nonzero(filled)
    if wall_mm is not None:
        r = wall_mm * perception.INTRINSICS["fx"] / max(o.top_depth_mm, 1.0)
    else:
        r_out = math.sqrt(filled.sum() / math.pi)
        # Too big for a ring: the mask merged the ring with something (the reset aimed 99 mm off the tape's centre)
        if r_out * o.top_depth_mm / perception.INTRINSICS["fx"] > RING_MAX_R_MM:
            return None
        r = 0.5 * (r_out + math.sqrt(hole.sum() / math.pi))
    return (u0 + float(us.mean()), v0 + float(vs.mean())), r


def ring_world(o, pose: dict, k: dict):
    """(ring centre table xy, outer radius mm, wall radius mm) of a ring-like object, or None."""
    ring = ring_geometry(o)
    if ring is None:
        return None
    (cu, cv), r_mid = ring
    u0, v0, u1, v1 = o.bbox
    filled = ndimage.binary_fill_holes(o.mask[v0:v1 + 1, u0:u1 + 1])
    mm = o.top_depth_mm / k["fx"]
    c = perception.world_xy(pose, k, cu, cv, o.top_depth_mm)
    return (float(c[0]), float(c[1])), math.sqrt(filled.sum() / math.pi) * mm, r_mid * mm


# A grasp point is moved to at least MIN_GRASP_R_MM + this from the base when possible, and approach refuses one
# closer: near the base the arm's pose looked unsafe to the people at the table (the E-stop was pressed at r ~220)
# {VIAM CENTERAL 2026-10-29 "DEMO 期间把 MIN_GRASP_R_MM + 15 改成 + 40（=245），宁可报 TOO_CLOSE"}
REACH_MARGIN_MM = 40.0
GRASP_END_MARGIN_MM = 15.0  # a grasp point moved along an object stays this far from its end


def along_axis(o, pose: dict, k: dict, u: float, v: float) -> tuple:
    """
    (u, v, yaw_deg) for a grasp at pixel (u, v) on `o`; closer to the base than MIN_GRASP_R_MM + REACH_MARGIN_MM it
    moves outward along the long axis, as far as needed and at most to GRASP_END_MARGIN_MM from the object's end.
    """
    d = o.top_depth_mm
    want = ppa.MIN_GRASP_R_MM + REACH_MARGIN_MM
    g = np.array(perception.world_xy(pose, k, u, v, d))
    if np.hypot(*g) >= want:
        return float(u), float(v), float(o.yaw_deg)
    a = math.radians(o.yaw_deg)
    tip = np.array(perception.world_xy(pose, k, u + 20.0 * math.cos(a), v - 20.0 * math.sin(a), d))
    axis = (tip - g) / (np.linalg.norm(tip - g) or 1.0)
    if axis @ g < 0:
        axis = -axis
    half = o.length_px * d / k["fx"] / 2.0
    step = min(max(0.0, half - GRASP_END_MARGIN_MM), max(0.0, want - float(np.hypot(*g))))
    for s in np.arange(0.0, step + 0.1, 2.0):
        if np.hypot(*(g + s * axis)) >= want:
            step = s
            break
    p = g + step * axis
    pu, pv = perception.project_xy(pose, k, tuple(p), d)
    return float(pu), float(pv), float(o.yaw_deg)


def grasp_px(o, pose: dict = None, k: dict = None, wall_mm: float = None) -> tuple:
    """
    (u, v, yaw_deg): where the jaws close on `o` in this frame; yaw is the image angle align_wrist turns vertical
    (the jaws close across it, along image u).

    Default: the top-face centroid (a long object's handle / thickest part) and the object's long axis.
    {VIAM CENTERAL 2026-10-29 "FOR ELONGATED OBJECTS, CHOOSE THE GRASP POINT AT THE HANDLE/THICKEST SECTION"}
    [CONFIDENCE: INFERRED 75% — perception.Obj.top_centroid_px is the top band within TOP_BAND_MM]
    Too close to the base there (< ppa.MIN_GRASP_R_MM + REACH_MARGIN_MM): moved outward along the long axis, at
    most to GRASP_END_MARGIN_MM from the object's end.
    Ring (ring_geometry: the black tape roll, whose top centroid is in its hole and whose outside is wider than
    the jaws): on the middle of its wall, the point nearest the base among those at least MIN_GRASP_R_MM +
    REACH_MARGIN_MM from it (the far side when the near one is out of reach); yaw = the wall's tangent there, so
    the jaws close across the wall. Needs the frame's pose and intrinsics.
    {VIAM CENTERAL 2026-10-29 "RING 的抓取点在环壁中线上取 R >= MIN_GRASP_R + 15 MM 的点里离 BASE 最近的那个"}
    [CONFIDENCE: INFERRED 70% — the tape at r 182 mm on its near side was refused; its far side (~250) is reachable]
    """
    if pose is None or k is None:
        u, v = o.top_centroid_px or o.centroid_px
        return float(u), float(v), float(o.yaw_deg)
    ring = ring_geometry(o, wall_mm)
    d = o.top_depth_mm
    want = ppa.MIN_GRASP_R_MM + REACH_MARGIN_MM

    def image_yaw(p1, p2) -> float:
        (u1, v1), (u2, v2) = perception.project_xy(pose, k, p1, d), perception.project_xy(pose, k, p2, d)
        return math.degrees(math.atan2(-(v2 - v1), u2 - u1)) % 180.0

    if ring is None:
        return along_axis(o, pose, k, *(o.top_centroid_px or o.centroid_px))
    (cu, cv), r_px = ring
    c = np.array(perception.world_xy(pose, k, cu, cv, d))
    r_mm = r_px * d / k["fx"]
    phis = np.radians(np.arange(0.0, 360.0, 5.0))
    pts = c[None, :] + r_mm * np.stack([np.cos(phis), np.sin(phis)], axis=1)
    rs = np.hypot(pts[:, 0], pts[:, 1])
    ok = rs >= want
    # Nearest the base among the reachable wall points; none reachable: the farthest (approach refuses it)
    i = int(np.argmin(np.where(ok, rs, np.inf))) if ok.any() else int(np.argmax(rs))
    p = pts[i]
    tangent = np.array([-math.sin(phis[i]), math.cos(phis[i])])
    pu, pv = perception.project_xy(pose, k, tuple(p), d)
    return float(pu), float(pv), image_yaw(tuple(p - 10.0 * tangent), tuple(p + 10.0 * tangent))


def ring_inner_mm(o, fx: float) -> float:
    """Diameter of a ring's hole in mm (0 if it has none), from pixels at its top-surface depth."""
    u0, v0, u1, v1 = o.bbox
    sub = o.mask[v0:v1 + 1, u0:u1 + 1]
    hole = ndimage.binary_fill_holes(sub) & ~sub
    return 2.0 * math.sqrt(hole.sum() / math.pi) * o.top_depth_mm / fx


def choose_primitives(o, fx: float) -> list:
    """Candidate primitives for object `o`, best first, from its geometry alone; [] if it is too wide to grasp."""
    mm = o.top_depth_mm / fx
    length, width = o.length_px * mm, o.width_px * mm
    if ring_geometry(o) is not None:
        # A ring's outside width does not matter: it is not gripped across it
        inside = enabled() and INSIDE_MIN_ID_MM <= ring_inner_mm(o, fx) <= INSIDE_MAX_ID_MM
        return ["ring_inside", "ring_wall"] if inside else ["ring_wall"]
    if width > MAX_OPEN_MM:
        return []
    if not enabled():
        return ["top"]
    if o.height_mm > TALL_MM and length < TALL_MAX_FOOTPRINT_MM:
        return ["standing", "top"]
    if length / max(width, 1.0) > LONG_ASPECT:
        return ["handle", "centroid"]
    return ["top"]


def target_px(primitive, o, pose: dict, k: dict, wall_mm: float = None) -> tuple:
    """(u, v, yaw_deg): where the jaws go for this primitive in this frame. top, handle, standing, ring_wall: grasp_px."""
    if primitive == "ring_inside":
        ring = ring_geometry(o, wall_mm)
        if ring is not None:
            # Into the hole: its centre. The yaw is unused (keeps_wrist_at_zero)
            return float(ring[0][0]), float(ring[0][1]), 90.0
    if primitive == "centroid" and pose is not None and k is not None:
        return along_axis(o, pose, k, *o.centroid_px)
    return grasp_px(o, pose, k, wall_mm)


def keeps_wrist_at_zero(primitive, o) -> bool:
    """True when the wrist angle does not matter for this grip, so the arm should not turn it at all."""
    return primitive == "ring_inside" or (primitive == "standing" and getattr(o, "rect_fill", 1.0) < ROUND_RECT_FILL)


def grip_fn(primitive):
    """How far below the object's top the fingertips close, from its height; None = ppa.pick's default."""
    # Halfway down a tall object, so closing on it does not tip it over
    return (lambda height: height / 2.0) if primitive == "standing" else None


class Escalation:
    """Which primitive next: candidates that failed FAILS_TO_ESCALATE times go last; all failed: the same order again."""

    def __init__(self):
        self.fails, self.tried, self.ts = {}, [], time.time()

    def order(self, candidates: list) -> list:
        fresh = [c for c in candidates if self.fails.get(c, 0) < FAILS_TO_ESCALATE]
        return fresh + [c for c in candidates if c not in fresh]

    def attempt(self, primitive: str) -> None:
        self.tried.append(primitive)
        self.ts = time.time()

    def failed(self, primitive: str) -> None:
        self.fails[primitive] = self.fails.get(primitive, 0) + 1
        self.ts = time.time()


_ESCALATIONS = {}


def escalation_for(text: str) -> Escalation:
    """The running Escalation of a request: the module retries a failed grasp with a new execute() call."""
    esc = _ESCALATIONS.get(text)
    if esc is None or time.time() - esc.ts > ESCALATION_TTL_S:
        esc = _ESCALATIONS[text] = Escalation()
    return esc


def forget_escalation(text: str) -> None:
    _ESCALATIONS.pop(text, None)


async def pick_inside(rig, times: dict, finder, check=None, slipped=None, inner_mm: float = 73.0) -> tuple:
    """
    Internal expansion grip for a ring: jaws closed, down into its hole, open against the inner wall, lift.
    Returns (height, grip depth) like ppa.pick; raises ppa.ObjectLost if the picture shows it stayed behind.

    WHY: clamping the wall crushed the tape's paper core and slipped; pressing outward from inside does not crush it,
    and centres the ring on the jaw axis. {VIAM CENTERAL 2026-10-30 "JAW 合拢后伸进环洞 ... 然后张开，向外撑住纸芯内壁"}
    [CONFIDENCE: INFERRED 60% — NOT YET RUN ON THE ARM; the gripper's behaviour on contact while opening is untested]
    The jaws stay open for the servo and the checks, the same view of the fingers as every other pick, and close only
    before the descent. Released by closing (ppa.release_jaws, grip_mode "internal").
    """
    t = time.monotonic()
    times["at"] = "servo"
    rig.picked_at, rig.grip_mode = None, "external"  # set again only after this pick's expansion
    await rig.open()
    p = await rig.pose()
    await rig.goto(p.x, p.y, ppa.ALIGN_Z)
    print("1/5 refine over the ring's hole (internal grip)")
    await rig.servo(finder, True, "ring")
    t = ppa.mark(times, "servo", t)
    print("2/5 measure + into the hole + expand")
    times["at"] = "descend"
    z0, table, top = await rig.measure(finder)
    if top is None:
        raise RuntimeError("ring top not found in depth map")
    height = table - top
    if not 10.0 <= height <= 80.0:
        raise RuntimeError(f"implausible ring height {height:.0f} mm")
    grip = min(ppa.GRIP_DEPTH, height / 2)
    if check is not None:
        ok, why = await check()
        print(f"  continuity check: {why}")
        if not ok:
            raise ppa.PickFailed(f"lost the target, refusing to grab something else: {why}")
    await rig.wait_clear(ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, top))
    hard = ppa.MIN_GRASP_Z_THIN if height < ppa.THIN_MM else ppa.MIN_GRASP_Z
    z = max(z0 - (top + grip - ppa.TIP_BELOW_CAM), rig.floor_z, hard)
    print(f"  table={table:.0f}mm top={top:.0f}mm height={height:.0f}mm hole {inner_mm:.0f}mm -> into the hole, z={z:.0f}")
    await rig.grab()  # nothing between the jaws yet: closed, they fit the hole
    try:
        await rig.descend(z, settle=False)  # the expansion follows, no frame
    except ppa.EStop:
        raise
    except ppa.MoveRefused as e:
        # Refused: the arm did not move. Back to open jaws, as every other failed pick leaves them
        q = await rig.pose()
        await rig.lift()
        await rig.open()
        if math.hypot(q.x, q.y) < ppa.NEAR_BASE_R_MM:
            raise ppa.TooClose(f"{ppa.too_close_msg(q.x, q.y)} ({e})") from e
        raise ppa.PickFailed(f"descent into the ring refused at ({q.x:.0f},{q.y:.0f}): {e}") from e
    t = ppa.mark(times, "descend", t)
    times["at"] = "grab"
    target = inner_mm + EXPAND_MARGIN_MM
    q = await rig.pose()
    # Noted before the command: if it is cut short the ring may be on the jaws, and recover() must close them to let go
    rig.picked_at, rig.grip_mode = (q.x, q.y, rig.theta), "internal"
    rig.expand_stop_mm = await rig.expand(target)
    print(f"  expand toward {target:.0f} mm: jaws stopped at {rig.expand_stop_mm} mm")
    # Same lift as ppa.pick: the first 30 mm as a short move, then up to where the slip check's frame is taken
    await rig.goto(q.x, q.y, min(q.z + 30.0, ppa.ALIGN_Z), settle=False)
    await rig.goto(q.x, q.y, ppa.ALIGN_Z)
    t = ppa.mark(times, "grab", t)
    if slipped is not None:
        q = await rig.pose()
        rgb, depth = await rig.frame()
        gone, why = slipped(rgb, depth, {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta})
        print(f"  after the lift: picture: {why}")
        if gone:
            # Stays open, not closed: if the picture were wrong the ring is still held, not dropped from up here
            rig.picked_at, rig.grip_mode = None, "external"
            raise ppa.ObjectLost(f"ring stayed behind on the lift ({why})")
    ppa.save_state(True, height, grip, mode="internal")
    return height, grip
