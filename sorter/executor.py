"""
Pick ANY object on the table (sorter.perception.Obj) and put it in the blue tape square; clear the square.

WHY: the demo is "grasp green square" with whatever is on the table, and the square must be emptied
between runs without a person. The arm mechanics (servo, depth heights, reconnect, intruder pause,
square search) live in scripts/pick_place_auto.py; this module adds object choice and tracking for
arbitrary objects, the in-square check, and the reset.

Flow, execute(text, target=None, target_xy=None): object from a given frame (target), else search_and_find
(current frame -> hint xy, e.g. the table map -> sweep poses nearest-first, stopping at the first frame that
shows it) -> square from any frame seen so far (else the last known square, else search while carrying) ->
ObjTracker -> ppa.pick -> ppa.place_in_square -> objects_in_hole check -> Store.log_attempt -> observe pose.
sweep_table(): theta 0 frames over the whole reachable table, only for an explicit "scan the table" command.
Flow, clear_square(): find square -> each object in the hole: pick (<= 2 tries) -> place_at random free spot;
failed pick -> wipe (<= 2 passes) -> verify hole empty by depth -> Store.log_attempt(mode="reset").
Contract: docs/EXEC_PLAN.md (Command flow, Reset).
"""
import inspect
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402  arm library (Rig, pick, place_in_square, place_at, ...)

from sorter import clip_embed, drop_square, perception  # noqa: E402

MAX_OPEN_MM = 78.0  # jaws open ~84 mm; keep a margin so the fingers clear the sides
COLOUR_TOL = 60.0  # RGB distance that still counts as the object at close range (grey/black/white objects)
HUE_TOL = 9  # PIL hue units (~13 deg) for saturated objects: hue barely changes when the block looks brighter up close
SAT_REF = 90  # PIL saturation above which an object is "coloured" and tracked by hue
WIPE_CLEAR_MM = 10.0  # fingertips this far above the table during a wipe
WIPE_STEP_MM = 20.0  # wipe in short moves = slow, and interruptible between steps
WORKER_ID = "sfhack4-worker"  # heartbeat / frame id; AGENT 2's Viam module uses "module:<name>"
# What the arm is doing right now, for heartbeats: the current attempt's times dict ("at" = phase) + last error
STATE = {"phase": "idle", "times": None, "error": None}
PHASES = {"square": "observe", "find": "observe", "check": "place", "start": "observe"}
# Flange x/y over the square's centre from the latest locate/place, and when; used when a frame shows no square
LAST_SQUARE = {"xy": None, "ts": 0.0}
SQUARE_CACHE_S = 600.0
# Whole-table sweep: theta 0 poses at SWEEP_Z. The first layout (z 500, rows at flange x 280/460) saw x >= 138
# only, so the green cube near the base (166, 118) was cut by the image edge and dropped, and no frame held the
# whole square. Raised to z 580 (user: "raise a little higher"): camera ~562 mm above the table, one frame covers
# ~445 mm in x by ~790 mm in y, centred ~(48, 33) mm from the flange (perception.JAW_IN_CAM_MM). Rows 170/400 and
# columns -180/+180 overlap ~210 / ~430 mm: green cube, KIND bar corner and the whole square (~(370, 84),
# 137 x 188 mm) each lie fully inside at least one of the 4 frames.
SWEEP_Z = 580.0
SWEEP_XY = [(x, y) for x in (170.0, 400.0) for y in (-180.0, 180.0)]
# Every sweep frame is saved here for offline debugging (AGENT3); SORTER_DATA_DIR overrides (module data dir)
SWEEP_DIR = Path(os.environ.get("SORTER_DATA_DIR") or Path(__file__).resolve().parent.parent / "data") / "sweeps"


def phase_now(rig) -> str:
    """Heartbeat phase: link/person states first, then the step in progress, else idle."""
    if rig.status:
        return rig.status
    at = (STATE.get("times") or {}).get("at")
    return PHASES.get(at, at) if at else STATE["phase"]


def publish_frame(store, rgb: np.ndarray, seg, phase: str, scores: dict = None) -> None:
    """Latest frame + boxes for the dashboard: targets, ignored (excluded) objects, and intruders."""
    if not hasattr(store, "put_frame"):
        return
    objs = [{"id": o.id, "bbox": list(o.bbox), "label": o.colour, "height_mm": round(o.height_mm),
             **({"score": round(scores[o.id], 3)} if scores and o.id in scores else {})} for o in seg.objects]
    objs += [{"id": o.id, "bbox": list(o.bbox), "label": why, "ignored": True} for o, why in seg.excluded]
    meta = {"objects": objs, "intruders": [{"bbox": list(i["bbox"]), "reason": i["reason"]} for i in seg.intruders],
            "phase": phase}
    try:
        store.put_frame(WORKER_ID, jpeg(rgb), meta)
    except Exception as e:
        print(f"put_frame failed: {e}")


async def apply_limits(rig) -> None:
    """Obstacle keep-outs (ppa.apply_workspace_limits) applied to the sweep/search poses; drop spots check ppa.allowed."""
    await ppa.apply_workspace_limits(rig)
    SWEEP_XY[:] = [q for q in SWEEP_XY if ppa.allowed(*q)]
    print(f"  workspace: {len(SWEEP_XY)} search poses")


def intr(rig) -> dict:
    return {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy}


def pose_dict(p) -> dict:
    return {"x": p.x, "y": p.y, "z": p.z, "theta": p.theta}


def start_recording(rig, store, doc: dict) -> None:
    """
    Give the attempt its Store id and record every frame the arm code already grabs into that attempt's clip.

    WHY: the dashboard's Replay plays these frames (AGENT 2's put_clip_frame); no extra photos are taken, so
    recording costs only JPEG encoding. No-op when the Store has no clip support.
    """
    rig.on_frame = None
    if not (hasattr(store, "new_attempt_id") and hasattr(store, "put_clip_frame")):
        return
    aid = store.new_attempt_id()
    doc["_id"] = aid
    t0 = time.monotonic()

    def on_frame(phase: str, rgb: np.ndarray, meta: dict) -> None:
        m = dict(meta, t=round(time.monotonic() - t0, 2))
        if rig.last_pose is not None:
            m["pose"] = {k: round(v, 1) for k, v in pose_dict(rig.last_pose).items()}
        store.put_clip_frame(aid, phase, jpeg(rgb), m)

    rig.on_frame = on_frame


def finish(rig, store, doc: dict) -> None:
    """Log the attempt under its recording id (Store keeps a given _id) and stop recording."""
    rig.on_frame = None
    doc["_id"] = store.log_attempt(dict(doc))


async def back_to_observe(rig) -> None:
    """
    End every task at the observe pose (HOME_XY, ALIGN_Z, theta 0) with whatever it holds already handled.

    WHY: the module's idle loop takes its frames there (within +-15 mm / +-1 deg), so the next command's photo
    is current. Never raises: a failed return is reported by the next command, not by this one.
    """
    try:
        await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
    except Exception as e:
        print(f"  could not return to the observe pose: {e}")


def square_from_frame(rgb: np.ndarray, depth: np.ndarray, pose: dict, intrinsics: dict):
    """
    Square hole centre seen in one theta-0 frame: {"px": [u, v], "xy": [x, y]}, or None if not enclosed.

    xy is the flange x/y that puts the jaws over the hole centre (perception.world_xy at the table depth),
    so a place can go straight there instead of searching. Also refreshes LAST_SQUARE.
    """
    if depth is None or abs(float(pose.get("theta", 0.0))) > 1.0:
        return None
    # find_square remembers the hole's pixel size for occluded views while carrying at ALIGN_Z; a search frame
    # from z 580 sees the hole ~30% smaller, so keep the size it had unless this frame is at ALIGN_Z too
    size = ppa.SQUARE_SIZE.get("wh")
    hit = ppa.find_square(rgb)
    if abs(float(pose.get("z", ppa.ALIGN_Z)) - ppa.ALIGN_Z) > 15.0:
        ppa.SQUARE_SIZE.pop("wh", None)
        if size is not None:
            ppa.SQUARE_SIZE["wh"] = size
    if hit is None:
        return None
    table, _ = ppa.table_and_near(depth)
    x, y = (float(v) for v in perception.world_xy(pose, intrinsics, hit[0], hit[1], table))
    LAST_SQUARE.update(xy=(x, y), ts=time.time())
    return {"px": [int(round(hit[0])), int(round(hit[1]))], "xy": [round(x, 1), round(y, 1)]}


def record_sightings(store, rgb, depth, seg, pose: dict, intrinsics: dict, source: str) -> list:
    """
    Tell the object memory what this frame saw (AGENT3's where_is follows moved objects); never fails.
    Returns the recognise_all() records (one CLIP batch) so best() can reuse them; [] on failure.
    """
    try:
        recs = clip_embed.recognise_all(rgb, depth, store, seg=seg)
    except Exception as e:
        print(f"  recognition failed: {e}")
        return []
    try:
        clip_embed.record_sightings(store, recs, pose, intrinsics, source=source)
    except Exception as e:
        print(f"  sightings not recorded: {e}")
    return recs


RING_HOLE_FRAC = 0.2  # share of an object's hole-filled mask that is hole: a ring (tape roll), grasped by its wall


def ring_geometry(o):
    """((cu, cv) ring centre px, radius px of the middle of the wall) if `o` is ring-like, else None."""
    u0, v0, u1, v1 = o.bbox
    sub = o.mask[v0:v1 + 1, u0:u1 + 1]
    filled = ndimage.binary_fill_holes(sub)
    hole = filled & ~sub
    if not filled.any() or hole.sum() < RING_HOLE_FRAC * filled.sum():
        return None
    vs, us = np.nonzero(filled)
    r_out, r_in = math.sqrt(filled.sum() / math.pi), math.sqrt(hole.sum() / math.pi)
    return (u0 + float(us.mean()), v0 + float(vs.mean())), 0.5 * (r_out + r_in)


def grasp_px(o, pose: dict = None, k: dict = None) -> tuple:
    """
    (u, v, yaw_deg): where the jaws close on `o` in this frame; yaw is the image angle align_wrist turns vertical
    (the jaws close across it, along image u).

    Default: the top-face centroid (a long object's handle / thickest part) and the object's long axis.
    {VIAM CENTERAL 2026-10-29 "FOR ELONGATED OBJECTS, CHOOSE THE GRASP POINT AT THE HANDLE/THICKEST SECTION"}
    [CONFIDENCE: INFERRED 75% — perception.Obj.top_centroid_px is the top band within TOP_BAND_MM]
    Ring (ring_geometry: the black tape roll, whose top centroid is in its hole and whose outside is wider than
    the jaws): the middle of its wall on the side nearest the arm base, yaw = the wall's tangent there, so the
    jaws close across the wall. Needs the frame's pose and intrinsics for the base direction.
    {VIAM CENTERAL 2026-10-29 "FOR RING-LIKE OBJECTS ... GRASP THE WALL: GRASP POINT ON THE RING AT THE SIDE NEAREST THE ARM"}
    [CONFIDENCE: INFERRED 60% — the ring mask (table visible in the hole) is not yet seen on the arm]
    """
    ring = ring_geometry(o) if pose is not None and k is not None else None
    if ring is None:
        u, v = o.top_centroid_px or o.centroid_px
        return float(u), float(v), float(o.yaw_deg)
    (cu, cv), r_px = ring
    # Image direction toward the base: the ring centre and a point 10 mm nearer the base, both projected
    cx, cy = perception.world_xy(pose, k, cu, cv, o.top_depth_mm)
    n = math.hypot(cx, cy) or 1.0
    pu, pv = perception.project_xy(pose, k, (cx - 10.0 * cx / n, cy - 10.0 * cy / n), o.top_depth_mm)
    du, dv = pu - cu, pv - cv
    L = math.hypot(du, dv) or 1.0
    # Tangent = radial turned 90 deg; same angle convention as perception (atan2(-dv, du), [0, 180))
    return cu + du / L * r_px, cv + dv / L * r_px, math.degrees(math.atan2(-du / L, -dv / L)) % 180.0


class ObjTracker:
    """
    Finder (rgb -> (u, v, yaw, area) or None) that follows ONE object of any colour, and only that object.

    WHY (2026-10-29, "grasp blue screwdriver" put the BLUE BLOCK in the square): (1) at close range the hue
    tracker matched only the handle's blue parts and centred the jaws on the blue end cap, so the grab missed;
    (2) the re-alignment after the miss started from that close-range pixel, and in the far frame the nearest
    object to it was the blue block; (3) nothing checked identity before closing. So:
    - far and close positions are kept apart: far frames are matched against the last FAR position only;
    - far frames (depth valid): every object incl. edge-cut ones (a turned screwdriver's shaft leaves the frame);
      when more than one is within reach of the last position, best() with the request text decides; with no
      candidate in reach it re-acquires by recognition over the whole frame;
    - close frames (depth blind < ~280 mm): pixels of ANY of the object's dominant colours (a blue+black handle
      is matched as a whole, not just its blue cap), nearest blob to the last position;
    - identity is fixed in the found frame (approach() -> self.ref); executor.still_target() checks continuity
      before every descent and falls back to recognition (ppa.pick check=).
    """

    MAX_JUMP_PX = 260.0

    def __init__(self, rig, obj, rgb: np.ndarray, text: str = None, store=None):
        self.rig, self.text, self.store = rig, text, store
        self.far_uv = obj.centroid_px  # last position in a far frame
        self.close_uv = None  # last position in a close frame; cleared when the arm goes back up
        self.close_target = None  # jaw pixel for the close servo, set by ppa.pick before it starts
        self.ref = None  # {"xy", "area"} of the object best() accepted (set by approach); still_target() compares
        self.uv = self.far_uv
        self._learn(rgb, obj.mask)

    def back_to_far(self) -> None:
        """Called by ppa.pick after a lift: the next frame is a far one, compare it with the last FAR sighting."""
        self.close_uv = None
        self.uv = self.far_uv

    def _learn(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        """
        Close-range colour model from the object's latest far view: one hue for its saturated part (hue barely
        changes when the object looks brighter up close) and one RGB median for its unsaturated part (black/grey/
        white), each kept if it covers >= 15% of the object. A blue+black screwdriver gets both.
        """
        hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))[mask].astype(float)
        px = rgb[mask].astype(float)
        sat = (hsv[:, 1] > SAT_REF) & (hsv[:, 2] > 50)
        self.hue = float(np.median(hsv[sat, 0])) if sat.mean() >= 0.15 else None
        self.rgb_ref = np.median(px[~sat], axis=0) if (~sat).mean() >= 0.15 else None
        if self.hue is None and self.rgb_ref is None:
            self.rgb_ref = np.median(px, axis=0)

    def _close_mask(self, rgb: np.ndarray) -> np.ndarray:
        m = np.zeros(rgb.shape[:2], bool)
        if self.hue is not None:
            hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
            dh = np.abs(hsv[..., 0] - int(self.hue))
            dh = np.minimum(dh, 256 - dh)  # hue is circular
            m |= (dh <= HUE_TOL) & (hsv[..., 1] > SAT_REF * 0.6) & (hsv[..., 2] > 50)
        if self.rgb_ref is not None:
            m |= np.linalg.norm(rgb.astype(float) - self.rgb_ref, axis=2) < COLOUR_TOL
        m[ppa.FINGER_ROW:] = False
        return ndimage.binary_closing(ndimage.binary_opening(m, iterations=2), iterations=4)

    def _far(self, rgb: np.ndarray, depth: np.ndarray):
        seg = perception.segment_full(rgb, depth)
        objs = list(seg.objects) + [o for o, _ in seg.excluded]
        if not objs:
            return None, None

        def dist(o):
            return math.hypot(o.centroid_px[0] - self.far_uv[0], o.centroid_px[1] - self.far_uv[1])

        near = [o for o in objs if dist(o) <= self.MAX_JUMP_PX]
        pick = min(near, key=dist) if near else None
        if len(near) > 1 and self.text:
            # Several objects in reach (the screwdriver next to the blue block): identity by name, not distance
            named = self._best(near)
            if named is not None:
                pick = named
        if pick is None:
            pick = self._best(objs)  # lost: re-acquire by recognition over the whole frame
            if pick is not None and dist(pick) > 2 * self.MAX_JUMP_PX:
                pick = None
            if pick is not None:
                print("  re-acquired the target by recognition")
        return pick, objs

    def _best(self, objs: list):
        if not self.text or not objs:
            return None
        try:
            return clip_embed.best(objs, self.text, self.store).get("obj")
        except Exception as e:
            print(f"  recognition failed: {e}")
            return None

    def __call__(self, rgb: np.ndarray):
        depth = self.rig.last_depth
        if depth is not None and (depth > 0).mean() > 0.6:
            o, _ = self._far(rgb, depth)
            if o is None:
                return None
            self.far_uv = self.uv = o.centroid_px
            self.close_uv = None
            if not o.touches_edge:
                self._learn(rgb, o.mask)
            # Servo onto the grasp point (handle / ring wall, grasp_px); matching between frames stays on the
            # overall centroid (far_uv)
            gu, gv, yaw = grasp_px(o, self.rig.at, intr(self.rig))
            return (gu, gv, yaw, int(o.mask.sum()))
        ref = self.close_uv or self.far_uv
        lab, n = ndimage.label(self._close_mask(rgb))
        cands, self._labels = [], []
        for k in range(1, n + 1):
            vs, us = np.nonzero(lab == k)
            if len(us) < 1500:
                continue
            self._labels.append(k)
            cu, cv = us.mean(), vs.mean()
            w, vec = np.linalg.eigh(np.cov(np.vstack([us - cu, vs - cv])))
            cands.append((cu, cv, math.degrees(math.atan2(-vec[1, 1], vec[0, 1])) % 180.0, len(us)))
        if not cands:
            return None
        k_best = min(range(len(cands)), key=lambda i: math.hypot(cands[i][0] - ref[0], cands[i][1] - ref[1]))
        best = cands[k_best]
        if math.hypot(best[0] - ref[0], best[1] - ref[1]) > self.MAX_JUMP_PX:
            return None
        # A long object does not fit the close view: the fingertip band (rows >= FINGER_ROW) and the image edges
        # crop it, the visible part's centroid slides toward its far end, and chasing it walked the jaws onto the
        # screwdriver's end cap. When the blob is cut off, trust the far alignment: report the jaw pixel itself.
        vs, us = np.nonzero(lab == self._labels[k_best])
        cut = vs.max() >= ppa.FINGER_ROW - 3 or vs.min() <= 2 or us.min() <= 2 or us.max() >= rgb.shape[1] - 3
        if cut and self.close_target is not None:
            print("  close view cuts the object off: keeping the far alignment")
            return (self.close_target[0], self.close_target[1], best[2], best[3])
        self.close_uv = self.uv = (best[0], best[1])
        return best


CONT_XY_MM = 35.0  # continuity: the object under the jaws is within this of the verified one's table xy
CONT_AREA = (0.35, 2.5)  # ...and its footprint this fraction of the verified one (a long shaft cut off ~halves it)


async def under_jaws(rig) -> tuple:
    """(object nearest the jaw-axis pixel at its own top depth or None, every object in view, frame pose)."""
    rgb, depth = await rig.frame()
    pose = pose_dict(await rig.pose())
    seg = perception.segment_full(rgb, depth)
    objs = list(seg.objects) + [o for o, _ in seg.excluded]

    def off_axis(o):
        # The servo centres the grasp point (ObjTracker), so measure from it: a long object's overall centroid
        # sits out on its shaft, a ring's in its hole, both > 80 px from the jaws
        au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, o.top_depth_mm)
        gu, gv, _ = grasp_px(o, pose, intr(rig))
        return math.hypot(gu - au, gv - av)

    under = min(objs, key=off_axis) if objs else None
    return (under if under is not None and off_axis(under) <= 80.0 else None), objs, pose


def footprint(rig, o, pose: dict) -> dict:
    """Table xy of the object's grasp point and its area scaled to REF_TABLE_DEPTH_MM, for continuity checks."""
    gu, gv, _ = grasp_px(o, pose, intr(rig))
    xy = perception.world_xy(pose, intr(rig), gu, gv, o.top_depth_mm)
    return {"xy": (float(xy[0]), float(xy[1])),
            "area": float(o.mask.sum()) * (o.top_depth_mm / perception.REF_TABLE_DEPTH_MM) ** 2}


async def still_target(rig, text: str, store, tracker) -> tuple:
    """
    (ok, why) just before a descent, after the wrist turn: still the object best() accepted in the found frame?

    WHY (2026-10-29, "blue screwdriver" refused at theta -50 with its handle centred under the jaws): with the
    wrist turned, the fingertip band cut the shaft off and recognition of the handle alone failed. So here only
    continuity: the object nearest the jaw pixel stands where the verified one stood (CONT_XY_MM, full pose incl.
    theta) with a similar footprint (CONT_AREA) as approach() recorded from the found frame (tracker.ref). If not
    provable (a missed grab moved it, or no reference), full recognition on this frame decides, so a different
    object (the blue block, clips_sd2) is still refused.
    """
    under, objs, pose = await under_jaws(rig)
    if under is None:
        return False, "nothing under the jaws"
    ref, why = getattr(tracker, "ref", None), "no verified footprint"
    if ref is not None:
        now = footprint(rig, under, pose)
        d = math.hypot(now["xy"][0] - ref["xy"][0], now["xy"][1] - ref["xy"][1])
        ratio = now["area"] / max(ref["area"], 1.0)
        if d <= CONT_XY_MM and CONT_AREA[0] <= ratio <= CONT_AREA[1]:
            return True, f"same object under the jaws ({d:.0f} mm from the verified spot, area x{ratio:.2f})"
        why = f"object under the jaws {d:.0f} mm from the verified spot, area x{ratio:.2f}"
    ok, why2 = identity(under, objs, text, store)
    if ok:
        tracker.ref = footprint(rig, under, pose)
        return True, f"{why2} (re-identified: {why})"
    return False, f"{why2} ({why})"


def identity(under, objs: list, text: str, store) -> tuple:
    """(ok, why): best() over all objects in view picks `under`, or picks nothing and accepts `under` alone."""
    found = clip_embed.best(objs, text, store)
    win = found.get("obj")
    if win is not None:
        same = math.hypot(win.centroid_px[0] - under.centroid_px[0], win.centroid_px[1] - under.centroid_px[1]) < 40.0
        if same:
            return True, "target under the jaws"
        return False, f"the object under the jaws is not '{text}' ('{found.get('recognised') or win.colour}' is elsewhere)"
    alone = clip_embed.best([under], text, store)
    if alone.get("obj") is not None:
        return True, "target under the jaws (only candidate)"
    return False, f"the object under the jaws does not look like '{text}'"


# An object is "in the square" when this many of its raised pixels lie on the white interior (at the 380 mm
# reference depth; scaled with depth like perception's areas). Was 70% of the footprint: the blue screwdriver lying
# across the yellow block segments as one blob mostly outside the interior, and a reset said "done" in 3.7 s.
# {VIAM CENTERAL 2026-10-29 "IN THE SQUARE IF ANY OF ITS RAISED PIXELS ... COVER THE WHITE INTERIOR BY ... >= 1500 PX"}
# [CONFIDENCE: SINGLE-SRC 80% — coordinator's report of the stacked case; threshold from them]
IN_SQUARE_PX = 1500
SQUARE_CLEAR_RAISED_PX = 600  # reset is done only when the interior has fewer raised pixels than this (noise)


def hole_mask(rgb: np.ndarray, depth: np.ndarray = None):
    """
    Bool mask of the white interior inside the blue tape (incl. anything standing on it), or None if not in view.

    WHY (2026-10-29): a yellow block standing half on the tape border counted as "in the square": the hole was
    the non-blue region enclosed by tape, and the block's own pixels joined it. Now the interior is the convex
    hull of the region's white paper pixels, which stops at the tape's inner edge; anything inside the paper
    (even a blue or white object) is still covered by the hull.
    With `depth`, a hole cut by the image border (the moved paper at the observe pose, frames 080/082) falls back
    to perception.find_drop_square's partial hole: its visible bbox, so a delivered object there is still "in the
    square" and never a grasp candidate. Its centre is not the hole's: never a place target.
    {AGENT3 2026-10-29 "THE FRAME PATH CANNOT EXCLUDE THE YELLOW BLOCK IN THE SQUARE"}
    [CONFIDENCE: SINGLE-SRC 80% — AGENT3: partial detector 0 false positives on 89 frames]
    """
    blue = ndimage.binary_closing(ppa.blue_mask(rgb), iterations=3)
    lab, n = ndimage.label(~blue)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    best = None
    for i in range(1, n + 1):
        if i in border:
            continue
        size = int((lab == i).sum())
        if size > 2000 and (best is None or size > best[1]):
            best = (i, size)
    if best is None:
        return partial_hole(rgb, depth)
    region = ndimage.binary_fill_holes(lab == best[0])
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    # Paper reads light blue to this camera (PIL S median 52-103 by exposure); tape is S > 120 by its own mask
    # {PERCEPTION.PY "PAPER READS LIGHT BLUE TO THIS CAMERA: PIL SATURATION MEDIAN 52-60 (010-076), 66 (080), 103 (082)"}
    # [CONFIDENCE: CONFIRMED 85% — AGENT3's measurement; the old S < 60 failed whole exposures]
    grey = region & (hsv[..., 1] < perception.SQUARE_WHITE_S)
    if grey.sum() < 1000:
        return region
    # Paper relative to its own brightness (exposure changes); objects are a minority of the region
    paper = grey & (hsv[..., 2] > 0.7 * np.median(hsv[..., 2][grey]))
    vs, us = np.nonzero(paper)
    if len(us) < 1000:
        return region
    pts = np.column_stack([us, vs])[:: max(1, len(us) // 4000)]
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        return region
    img = Image.new("L", (rgb.shape[1], rgb.shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(map(int, p)) for p in hull], fill=1)
    return np.asarray(img, dtype=bool)


def partial_hole(rgb: np.ndarray, depth: np.ndarray):
    """Visible part (bbox) of a border-cut drop square from perception.find_drop_square, or None; never fails."""
    if depth is None:
        return None
    try:
        sq = perception.find_drop_square(rgb, depth)
    except Exception as e:
        print(f"  partial square check failed: {e}")
        return None
    if not sq or not sq.get("partial"):
        return None
    u0, v0, u1, v1 = (int(round(c)) for c in sq["bbox"])
    m = np.zeros(rgb.shape[:2], bool)
    m[max(v0, 0):v1 + 1, max(u0, 0):u1 + 1] = True
    return m


def in_hole(o, hole: np.ndarray) -> bool:
    """Object's raised pixels cover >= IN_SQUARE_PX of the square's white interior (any shape, stacked or overhanging)."""
    table = o.top_depth_mm + o.height_mm
    return (o.mask & hole).sum() >= IN_SQUARE_PX * perception.area_scale(table)


def raised_in_hole(depth: np.ndarray, hole: np.ndarray) -> int:
    """Pixels of the white interior standing > perception.MIN_HEIGHT_MM above the fitted table plane."""
    a, b, c = perception.fit_table(depth.astype(float))
    h, w = depth.shape
    plane = a * np.arange(w)[None, :] + b * np.arange(h)[:, None] + c
    return int((hole & (depth > 0) & (plane - depth > perception.MIN_HEIGHT_MM)).sum())


def frame_square(rgb: np.ndarray, depth: np.ndarray, pose: dict, k: dict):
    """
    The drop square from this one frame (AGENT3's drop_square on perception.find_drop_square, which tells raised
    objects from tape by depth): {xy, size_mm, yaw_deg, corners, partial, confidence, ...} or None; never fails.
    """
    if depth is None:
        return None
    try:
        hit = perception.find_drop_square(rgb, depth)
        return drop_square.update(None, drop_square.observe(hit, pose, k), time.time()) if hit else None
    except Exception as e:
        print(f"  square estimate failed: {e}")
        return None


def square_mask(sq: dict, pose: dict, k: dict, shape: tuple, depth_mm: float) -> np.ndarray:
    """The square's estimated interior (its four corners projected at table depth) as a bool image mask."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    pts = [perception.project_xy(pose, k, c, depth_mm) for c in sq["corners"]]
    ImageDraw.Draw(img).polygon([(float(u), float(v)) for u, v in pts], fill=1)
    return np.asarray(img, dtype=bool)


def objects_in_hole(rgb: np.ndarray, depth: np.ndarray, pose: dict = None, k: dict = None) -> tuple:
    """
    (hole visible?, [Obj inside the hole], segmentation, interior mask) from a frame taken above the square at
    z >= 400. Inside = in_hole(). Must be looked at from ALIGN_Z: below ~280 mm depth is blind and a block in
    the square is invisible. Edge-cut objects are counted too (a full square can push one to the border).
    With the frame's pose and intrinsics the interior is drop_square's full estimate (frame_square), whatever
    lies in it: a silver knife reads as tape (52-69% of its pixels, AGENT3) and split the colour-based hole, so a
    reset failed with "square: not visible". Without them, or with no full estimate, the colour-based hole_mask.
    """
    seg = perception.segment_full(rgb, depth) if depth is not None else None
    hole = None
    if pose is not None and k is not None and depth is not None:
        sq = frame_square(rgb, depth, pose, k)
        if sq is not None and not sq.get("partial"):
            hole = square_mask(sq, pose, k, rgb.shape, float(np.median(table_plane(depth))))
    if hole is None:
        hole = hole_mask(rgb, depth)
    if hole is None or seg is None:
        return False, [], seg, None
    objs = list(seg.objects) + [o for o, _ in seg.excluded]
    return True, [o for o in objs if in_hole(o, hole)], seg, hole


# Clear room a release needs, around the drop point at theta 0 (the jaws close along image u): the fingers open to
# ~84 mm, so a strip +-SPOT_JAW_MM along u and +-SPOT_FINGER_MM across it, plus the object's own grasped section
SPOT_JAW_MM = 45.0
SPOT_FINGER_MM = 12.0
SPOT_BODY_MAX_MM = 40.0  # half-length of the object counted at most: a long object's rest hangs over the border
RELEASE_MATCH_MM = 60.0  # the check looks for the placed object's grasp point this close to the release point


def table_plane(depth: np.ndarray) -> np.ndarray:
    """Fitted table depth per pixel (perception.fit_table)."""
    a, b, c = perception.fit_table(depth.astype(float))
    h, w = depth.shape
    return a * np.arange(w)[None, :] + b * np.arange(h)[:, None] + c


def square_spot(rig, size_mm: tuple = (40.0, 40.0)):
    """
    ppa.place_in_square's choose_spot: (flange xy over the free point of the white interior nearest its centre,
    why), or (None, why) if no point of the interior has the room a release needs: the finger strip (+-SPOT_JAW_MM
    along image u, +-SPOT_FINGER_MM along v) and the held object's grasped section (size_mm = (across the jaws,
    along them), half-length capped at SPOT_BODY_MAX_MM).

    WHY (2026-10-29): the blue screwdriver was put down on top of the yellow block already in the square. Seen
    from the carry-offset view (hole up-left, held object at the bottom edge): occupied = raised above the fitted
    table or unreadable; the held object (blind / raised and connected to the bottom row) counts as unknown, not
    free. The free point keeps the jaws' clearance from every occupied pixel and from the tape.
    {VIAM CENTERAL 2026-10-29 "DEPTH-FREE CELL NEAREST THE CENTRE WITH CLEARANCE FOR THE FOOTPRINT, OR FAIL WITH 'SQUARE IS FULL'"}
    [CONFIDENCE: INFERRED 70% — offline on synthetic depth only]
    """
    def choose(rgb: np.ndarray, depth: np.ndarray, pose: dict) -> tuple:
        if depth is None:
            return None, "no depth"
        # The full interior only (no border-cut fallback): the release needs all of it in view
        hole = hole_mask(rgb)
        if hole is None:
            raise ppa.SquareNotFound("the square's full interior is not in view at the carry-offset pose")
        plane = table_plane(depth)
        up = (depth > 0) & (plane - depth > perception.MIN_HEIGHT_MM)
        blind = depth == 0
        lab, _ = ndimage.label(up | blind)
        held = np.isin(lab, np.unique(lab[-1][lab[-1] > 0]))
        busy = up | blind | held
        free = hole & ~busy
        table = float(np.median(plane[hole]))
        mm = rig.fx / table  # px per mm on the table

        def room(half_u_mm: float, half_v_mm: float) -> np.ndarray:
            # Pixels whose whole (2 half_v + 1) x (2 half_u + 1) rectangle is free; outside the image is not free
            size = (2 * int(half_v_mm * mm) + 1, 2 * int(half_u_mm * mm) + 1)
            return ndimage.minimum_filter(free.astype(np.uint8), size=size, mode="constant", cval=0).astype(bool)
        across, along = size_mm
        ok = room(SPOT_JAW_MM, SPOT_FINGER_MM) & room(across / 2, min(along / 2, SPOT_BODY_MAX_MM))
        if not ok.any():
            return None, (f"no room for the release on the white interior ({2 * SPOT_JAW_MM:.0f} mm finger strip "
                          f"+ {across:.0f} x {min(along, 2 * SPOT_BODY_MAX_MM):.0f} mm object)")
        vs, us = np.nonzero(hole)
        cu, cv = us.mean(), vs.mean()
        vs, us = np.nonzero(ok)
        i = int(np.argmin((us - cu) ** 2 + (vs - cv) ** 2))
        u, v = float(us[i]), float(vs[i])
        xy = perception.world_xy(pose, intr(rig), u, v, float(plane[int(v), int(u)]))
        off = math.hypot(u - cu, v - cv) * table / rig.fx
        busy_px = int((hole & up).sum())
        return (float(xy[0]), float(xy[1])), f"free spot {off:.0f} mm from the centre ({busy_px} px occupied)"
    return choose


def placed_ok(rig, rgb: np.ndarray, depth: np.ndarray, text: str, store) -> tuple:
    """
    (hole seen?, success?, why, segmentation) right after a place, from the retreat pose above the release point.

    Success = the object at the release point (grasp point within RELEASE_MATCH_MM of the jaw axis) has its grasp
    point on the white interior and is the target: best() over it accepts it, or it is not recognised as anything
    else. Other objects in the square do not matter, and a long object hanging over the border passes by its
    grasped section. Before: "wrong object in the hole" for the blue screwdriver (handle in, shaft out, the yellow
    block also inside).
    {VIAM CENTERAL 2026-10-29 "SUCCESS WHEN THE TARGET IS IN THE HOLE (BY RECOGNITION PLUS POSITION), REGARDLESS OF OTHER OBJECTS"}
    [CONFIDENCE: SINGLE-SRC 80% — rule from the coordinator; recognition of the placed object not yet seen on the arm]
    """
    seg = perception.segment_full(rgb, depth) if depth is not None else None
    hole = hole_mask(rgb, depth)
    if hole is None or seg is None:
        return False, False, "hole not visible", seg
    pose = pose_dict(rig.last_pose) if rig.last_pose is not None else dict(rig.at)
    k = intr(rig)
    table = float(np.median(table_plane(depth)[hole]))
    au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, table)
    best_o, best_d = None, RELEASE_MATCH_MM * rig.fx / table
    for o in list(seg.objects) + [o for o, _ in seg.excluded]:
        gu, gv, _ = grasp_px(o, pose, k)
        inside = 0 <= int(gv) < hole.shape[0] and 0 <= int(gu) < hole.shape[1] and hole[int(gv), int(gu)]
        d = math.hypot(gu - au, gv - av)
        if inside and d < best_d:
            best_o, best_d = o, d
    if best_o is None:
        return True, False, "nothing at the release point in the square", seg
    found = clip_embed.best([best_o], text, store)
    if found.get("obj") is not None:
        return True, True, "target in the square", seg
    if found.get("recognised") is None:
        return True, True, "placed in the square (not re-recognised)", seg
    return True, False, f"the object in the square is recognised as {found['recognised']!r}", seg


def jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def width_mm(obj, fx: float) -> float:
    """Object width across the jaws (short side) in mm, from pixels at its top-surface depth."""
    return obj.width_px * obj.top_depth_mm / fx


async def look_here(rig, text: str, store, source: str = "demo") -> dict:
    """
    One frame from the current pose -> best match for `text` among objects outside the square.

    Returns clip_embed.best()'s dict plus "rgb", "depth", "pose", and "square" (square_from_frame) when the
    hole is in view. Objects cut by the image border or standing in the square are never candidates.
    """
    rgb, depth = await rig.frame()
    pose = pose_dict(await rig.pose())
    rig.record("observe", rgb)
    seg = perception.segment_full(rgb, depth)
    publish_frame(store, rgb, seg, "observe")
    # One CLIP batch for the sightings and for best(): it used to embed every crop twice per look
    recs = record_sightings(store, rgb, depth, seg, pose, intr(rig), source) if abs(pose["theta"]) <= 1.0 else []
    hole = hole_mask(rgb, depth)
    sq = (live_map(rig) or {}).get("square")
    k = intr(rig)

    def delivered(o) -> bool:
        # In the square in this frame, or inside the map's estimated full interior (any rotation, even when only
        # part of the paper is in view: AGENT3's drop_square), by the grasp point's table xy
        if hole is not None and in_hole(o, hole):
            return True
        if not sq:
            return False
        gu, gv, _ = grasp_px(o, pose, k)
        return drop_square.contains(sq, perception.world_xy(pose, k, gu, gv, o.top_depth_mm))
    objs = [o for o in seg.objects if not o.touches_edge and not delivered(o)]
    ids = {id(o) for o in objs}
    found = (clip_embed.best(objs, text, store, recognised=[r for r in recs if id(r["obj"]) in ids])
             if objs else {"obj": None, "reason": "nothing seen", "searchable": True})
    found.update(rgb=rgb, depth=depth, pose=pose, square=square_from_frame(rgb, depth, pose, intr(rig)))
    return found


def hint_for(text: str, store):
    """Table xy where the object memory last saw what `text` names (clip_embed.where_is), or None; never fails."""
    try:
        w = clip_embed.where_is(store, text)
    except Exception as e:
        print(f"  no last sighting: {e}")
        return None
    return tuple(w["xy"]) if w else None


async def search_and_find(rig, text: str, store, hint_xy=None, progress=None, skip_current: bool = False) -> dict:
    """
    Look for `text`, stopping at the FIRST frame whose best() accepts an object; no whole-table sweep first.

    WHY: the user wants "just search, and if found just start the grasp; do NOT have an initial step". The old
    module flow swept all 4 poses, built the map, then went back to look again before picking.
    {VIAM CENTERAL 2026-10-29 "ON THE FIRST ACCEPTANCE STOP AND HAND THAT TARGET, WITH ITS FRAME POSE, STRAIGHT TO PICK"}
    [CONFIDENCE: CONFIRMED 90% — user correction relayed by the coordinator]
    Order: the current frame (no move) -> the hint (live map / last sighting, looked at from ALIGN_Z with it
    near the image centre) -> SWEEP_XY at SWEEP_Z, each next pose the nearest to where the arm is. One frame
    per pose; every frame still reaches the live map (Rig.frame -> on_map_frame). An ambiguous match stops at
    once (another view does not make two equal matches unequal). "Not found" only after every pose was seen.
    skip_current: the caller just looked from here (module idle frame), start at the hint.
    Returns look_here()'s dict + "path" (current / hint / sweep) and "views"; when the found frame has no
    square but an earlier one did, "square" is that one and "square_from" is "search".
    progress(k, n, path, found) after each look (AGENT 2's module status: "searching... pose k/n", "found at
    pose k"); n = 1 + planned poses. Called plainly; an awaitable it returns is awaited.
    """
    hint_xy = hint_xy if hint_xy is not None else hint_for(text, store)
    poses = []
    if hint_xy is not None:
        fx, fy = perception.look_flange_xy(tuple(hint_xy))
        # Same reach limits as every other look: keep-outs and the 120 mm base clearance
        if ppa.reachable(fx, fy) and math.hypot(fx, fy) >= 120.0:
            poses.append(("hint", fx, fy, ppa.ALIGN_Z))
    # Only poses the arm can reach at SWEEP_Z (ppa.reachable): a far corner is skipped, not refused mid-search
    left = [q for q in SWEEP_XY if ppa.reachable(*q, SWEEP_Z) and math.hypot(*q) >= 150.0]
    p = await rig.pose()
    at = (p.x, p.y) if not poses else poses[0][1:3]
    while left:
        # Greedy nearest-next: from the hint pose (or the current pose) to the closest unseen sweep pose
        q = min(left, key=lambda c: math.hypot(c[0] - at[0], c[1] - at[1]))
        left.remove(q)
        poses.append(("sweep", q[0], q[1], SWEEP_Z))
        at = q
    square, views, last = None, 0, {"obj": None, "reason": "nothing seen"}
    first = [] if skip_current else [("current", None, None, None)]
    for path, x, y, z in first + poses:
        if path != "current":
            try:
                await ppa.travel(rig, x, y, theta=0.0, z=z)
            except ppa.MoveRefused as e:
                print(f"  search: skipping ({x:.0f},{y:.0f},{z:.0f}): {e}")
                continue
        found = await look_here(rig, text, store, source="look" if path == "hint" else "demo")
        views += 1
        found.update(path=path, views=views)
        square = found.get("square") or square
        if progress is not None:
            try:
                r = progress(views, len(first) + len(poses), path, found.get("obj") is not None)
                if inspect.isawaitable(r):
                    await r
            except Exception as e:
                print(f"  search progress callback failed: {e}")
        if found.get("obj") is not None:
            if found.get("square") is None and square is not None:
                found.update(square=square, square_from="search")
            print(f"  search: found at view {views} ({path})")
            return found
        last = found
    last["reason"] = f"no {text!r} on the table after {views} views ({last.get('reason')})"
    return last


def wrist_for(pose: dict, k: dict, u: float, v: float, yaw: float, depth: float) -> float:
    """
    Wrist theta (deg, within +-90) that turns the image axis `yaw` (at pixel u, v of a frame taken at `pose`)
    vertical, where the jaws close across it: the axis' two ends in table xy (world_xy), re-projected for every
    candidate theta (project_xy); the smallest |theta| among the best fits wins.
    """
    du, dv = 30.0 * math.cos(math.radians(yaw)), -30.0 * math.sin(math.radians(yaw))
    a = perception.world_xy(pose, k, u - du, v - dv, depth)
    b = perception.world_xy(pose, k, u + du, v + dv, depth)
    best = None
    for t in range(-90, 91):
        q = {"x": 0.0, "y": 0.0, "z": ppa.ALIGN_Z, "theta": float(t)}
        (ua, va), (ub, vb) = perception.project_xy(q, k, a, depth), perception.project_xy(q, k, b, depth)
        img = math.degrees(math.atan2(-(vb - va), ub - ua)) % 180.0
        err = abs(((90.0 - img + 90.0) % 180.0) - 90.0)
        key = (round(err), abs(t))
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1]


async def approach(rig, found: dict, tracker) -> None:
    """
    Go straight to the pre-grasp pose computed from the found frame, in ONE move with the wrist already turned:
    jaws over the grasp point (grasp_px) at ALIGN_Z, long axis (or ring wall tangent) vertical in the image;
    move the tracker's seed pixel with it and record the object's footprint (tracker.ref) for still_target().
    No new look.

    WHY: the yellow block's far servo started 530 px off and took 8 iterations (~31 s) of 60 mm steps, and the
    wrist turn cost a separate rotation + re-servo (~3.8 s on the blue screwdriver). The frame already gives the
    grasp point's table xy (perception.world_xy) and the axis direction; the camera model gives the flange xy and
    theta that put the jaws over it. best() accepted the whole object in that frame, so identity is fixed there.
    {VIAM CENTERAL 2026-10-29 "COMPUTE THE TARGET'S WORLD XY FROM THE FRAME ... GO STRAIGHT TO ITS LOOK POSE IN ONE MOVE"}
    {VIAM CENTERAL 2026-10-29 "APPLY THE YAW IN THE SAME MOVE AS THE FIRST TRANSLATION"}
    [CONFIDENCE: INFERRED 75% — camera model (THETA_SIGN, TOOL_AXIS_MM) checked on theta_test frames, not on a grasp]
    Falls back to theta 0 and the pose that centres it in view (look_flange_xy), then the frame's own xy, when the
    jaw-over pose is out of reach. Raises ppa.TooClose under MIN_GRASP_R_MM.
    """
    pose = found["pose"]
    obj, k = found["obj"], intr(rig)
    u, v, yaw = grasp_px(obj, pose, k)
    xy = perception.world_xy(pose, k, u, v, obj.top_depth_mm)
    # The jaws cannot reach down this close to the base: refuse before any servo time is spent
    if math.hypot(*xy) < ppa.MIN_GRASP_R_MM:
        raise ppa.TooClose(ppa.too_close_msg(*xy))
    top = obj.top_depth_mm - (pose["z"] - ppa.ALIGN_Z)  # camera height above it shrinks by the z change
    theta = wrist_for(pose, k, u, v, yaw, obj.top_depth_mm)
    # Flange that puts the jaw axis (axis pixel at that depth) over xy at this theta: the offset is translation-free
    au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, top)
    jx, jy = perception.world_xy({"x": 0.0, "y": 0.0, "z": ppa.ALIGN_Z, "theta": theta}, k, au, av, top)
    fx, fy = xy[0] - jx, xy[1] - jy
    if not ppa.allowed(fx, fy) or math.hypot(fx, fy) < 120.0:
        theta = 0.0
        fx, fy = perception.look_flange_xy(xy)
        if not ppa.allowed(fx, fy) or math.hypot(fx, fy) < 120.0:
            fx, fy = pose["x"], pose["y"]  # every closer pose out of reach: go down from where it was seen
    p = await rig.pose()
    # Already there (e.g. a retry from the same frame): skip the zero-length move and its settle time
    if math.hypot(p.x - fx, p.y - fy) > 3.0 or abs(p.z - ppa.ALIGN_Z) > 3.0 or abs(p.theta - theta) > 1.0:
        await ppa.travel(rig, fx, fy, theta=theta)
    seed = perception.project_xy({"x": fx, "y": fy, "z": ppa.ALIGN_Z, "theta": theta}, k, xy, top)
    tracker.far_uv = tracker.uv = (float(seed[0]), float(seed[1]))
    tracker.ref = footprint(rig, obj, pose)
    print(f"  approach: grasp point ({xy[0]:.0f},{xy[1]:.0f}) -> flange ({fx:.0f},{fy:.0f}) theta {theta:.0f}, "
          f"seed px ({seed[0]:.0f},{seed[1]:.0f})")


def save_frame(out: Path, k: int, fr: dict) -> None:
    """Sweep frame k to out/: <k>.jpg, <k>_depth.npy (uint16 mm), <k>.json (id, pose, intrinsics, square)."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fr["rgb"]).save(out / f"{k:02d}.jpg", quality=92)
        np.save(out / f"{k:02d}_depth.npy", fr["depth"])
        meta = {key: fr[key] for key in ("id", "pose", "intrinsics", "square") if key in fr}
        (out / f"{k:02d}.json").write_text(json.dumps(meta))
    except OSError as e:
        print(f"  could not save sweep frame {k}: {e}")


async def sweep_table(rig, progress=None) -> list:
    """
    Frames over the whole reachable table for sorter.table_map.map_from_frames (AGENT3); ends at observe.
    Only for an explicit "scan the table" command: a grasp never sweeps first (search_and_find stops early).

    Each frame: {"id", "rgb", "depth" (uint16 mm, aligned), "pose", "intrinsics"} plus "square" when the hole
    is fully in view. progress(k, n) is called after each pose (dashboard: "scanning table... k/n").
    """
    poses = [q for q in SWEEP_XY if ppa.reachable(*q, SWEEP_Z) and math.hypot(*q) >= 150.0]
    frames, n = [], len(poses)
    t0 = time.monotonic()
    out = SWEEP_DIR / time.strftime("%Y%m%d-%H%M%S")
    try:
        for k, (x, y) in enumerate(poses, 1):
            await ppa.travel(rig, x, y, theta=0.0, z=SWEEP_Z)
            rgb, depth = await rig.frame()
            pose = pose_dict(await rig.pose())
            rig.record("sweep", rgb, k=k, n=n)
            fr = {"id": f"sweep-{int(time.time() * 1000)}-{k}", "rgb": rgb,
                  "depth": np.clip(depth, 0, 65535).astype(np.uint16), "pose": pose, "intrinsics": intr(rig)}
            sq = square_from_frame(rgb, depth, pose, fr["intrinsics"])
            if sq:
                fr["square"] = sq
            frames.append(fr)
            save_frame(out, k, fr)
            STATE["sweep"] = {"k": k, "n": n}
            if progress:
                progress(k, n)
    finally:
        STATE["sweep"] = None
        await back_to_observe(rig)
    print(f"sweep: {len(frames)} frames in {time.monotonic() - t0:.1f} s -> {out}")
    return frames


async def execute(rig, text: str, store, mode: str = "demo", target=None, target_xy=None, progress=None,
                  skip_current: bool = False) -> dict:
    """
    One command: put the object described by `text` into the blue square. Returns the logged attempt doc.

    target: clip_embed.best() dict + "rgb", "depth", "pose", "ts" from a frame the caller already took at the
      observe pose (theta within +-1 deg): no observe move, pick at once. Ignored if its theta is off.
    target_xy: a hint, table x/y where it was last seen (live map): search_and_find looks there right after the
      current frame, then keeps searching if it is gone. None: the object memory's last sighting is the hint.
    progress(k, n, path, found): passed to search_and_find (module status "searching... pose k/n").
    skip_current: the caller has just looked from the current pose (the module's idle frame): the search starts
      at the hint instead of taking that view again (~3.6 s on the arm).
    The square comes from the found frame, else any frame of this search, else known_square() (the newer of
    LAST_SQUARE and the live map's), else it is searched only while carrying. Always ends at the observe pose.
    Never raises for an ordinary failure: status is done / not_found / ambiguous / too_wide / too_close /
    square_full / failed.
    """
    doc = {"ts": time.time(), "mode": mode, "target_text": text, "success": False, "status": "failed",
           "fail_phase": None, "fail_reason": None}
    times = {}
    STATE.update(times=times, error=None)
    start_recording(rig, store, doc)
    t0 = time.monotonic()
    try:
        times["at"] = "find"
        found = None
        if target is not None and target.get("obj") is not None and abs(float((target.get("pose") or {}).get("theta", 99.0))) <= 1.0:
            found, doc["find_path"] = target, "frame"
            tp = target["pose"]
            # No move back to where that frame was taken: approach() goes from here straight to the object
            if found.get("square") is None:
                found["square"] = square_from_frame(found["rgb"], found.get("depth"), tp, intr(rig))
        else:
            found = await search_and_find(rig, text, store, hint_xy=target_xy, progress=progress,
                                          skip_current=skip_current)
            doc["find_path"], doc["find_views"] = found.get("path"), found.get("views")
        obj = found.get("obj")
        if obj is None:
            doc.update(status="not_found", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        doc.update(label=found.get("recognised") or found.get("label") or obj.colour or "object", colour=obj.colour,
                   score=round(float(found.get("score", 0.0)), 3),
                   margin=round(float(found.get("margin", 0.0)), 3), img_before=jpeg(found["rgb"]))
        if found.get("ambiguous"):
            doc.update(status="ambiguous", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        # A ring is grasped by its wall (grasp_px), so its outside width does not matter
        if width_mm(obj, rig.fx) > MAX_OPEN_MM and ring_geometry(obj) is None:
            doc.update(status="too_wide", fail_phase="find", fail_reason=f"{width_mm(obj, rig.fx):.0f} mm across")
            return doc
        # Where to look for the square while carrying: a full view from this task's frames, then recent
        # sightings newest first, then a search; place_in_square releases only after a full view of the interior
        square_xy, doc["square_path"] = None, "search"
        if found.get("square"):
            square_xy, doc["square_path"] = tuple(found["square"]["xy"]), found.get("square_from", "frame")
        sightings = square_sightings(rig)
        if square_xy is None and sightings:
            doc["square_path"] = sightings[0][1]
        tracker = ObjTracker(rig, obj, found["rgb"], text=text, store=store)
        # One move (translation + wrist yaw) to the pre-grasp pose computed from the found frame, where best()
        # already accepted the whole object: identity is fixed there (tracker.ref); pick() refines and checks
        # continuity before the descent
        await approach(rig, found, tracker)
        height, grip = await ppa.pick(rig, times, tracker, check=lambda: still_target(rig, text, store, tracker))
        doc["height_mm"], doc["grasp"] = round(height, 1), {"grip_depth": round(grip, 1), "theta": rig.theta}
        size = (width_mm(obj, rig.fx), obj.length_px * obj.top_depth_mm / rig.fx)
        # Look first where the found frame saw part of the paper, then the recent sightings; search the observe
        # pose before any z 580 pose (most squares are in that view)
        partial = partial_square_xy(found, rig)
        centre = await ppa.place_in_square(rig, height, grip, times, square_xy=square_xy,
                                           choose_spot=square_spot(rig, size),
                                           more_guesses=([partial] if partial else []) + [xy for xy, _ in sightings],
                                           search_poses=[(*ppa.HOME_XY, ppa.ALIGN_Z)]
                                           + [(x, y, SWEEP_Z) for x, y in SWEEP_XY])
        LAST_SQUARE.update(xy=tuple(centre), ts=time.time())
        q = await rig.pose()
        times["at"] = "check"
        rgb, depth = await rig.frame()
        rig.record("check", rgb)
        # Released at a free spot, not necessarily the centre: the square's position comes from this view
        square_from_frame(rgb, depth, pose_dict(q), intr(rig))
        seen, right, why, seg = placed_ok(rig, rgb, depth, text, store)
        if seg is not None:
            publish_frame(store, rgb, seg, "place")
        doc["img_after"] = jpeg(rgb)
        doc["success"] = right
        doc["status"] = "done" if right else "failed"
        doc["check"] = why
        if not right:
            doc.update(fail_phase="check", fail_reason=why)
    except ppa.SquareFull as e:
        # Nothing free in the square: put it back where it came from and say so
        doc.update(status="square_full", fail_phase="place", fail_reason=str(e))
        await recover(rig, doc)
    except ppa.TooClose as e:
        # A refusal, not a failure: nothing was grabbed; tell the person to move it out
        doc.update(status="too_close", fail_phase=times.get("at", "find"), fail_reason=str(e))
        await recover(rig, doc)
    except Exception as e:
        doc.update(status="failed", fail_phase=times.get("at", "start"), fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    finally:
        times.pop("at", None)
        doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
        finish(rig, store, doc)
        STATE.update(times=None, error=doc["fail_reason"] if doc["status"] == "failed" else None)
        await back_to_observe(rig)
    return doc


async def recover(rig, doc: dict) -> None:
    """
    After a failure: lift; if still holding something, put it back where it was picked.

    WHY (2026-10-29): a reset found no free spot and this put the yellow block down "right there", over the
    last refused spot, half on the square's tape. The pick spot is the one place known to fit this object, so
    the gripper opens only there (lowered to the measured table); with no pick spot known it keeps holding.
    """
    try:
        await rig.lift()
        if await rig.holding():
            xy = await put_back(rig, doc.get("height_mm", 30.0), (doc.get("grasp") or {}).get("grip_depth", 15.0))
            doc["put_back_at"] = [round(xy[0], 1), round(xy[1], 1)]
            doc["fail_reason"] = f"{doc.get('fail_reason')} | put back where it was picked ({xy[0]:.0f},{xy[1]:.0f})"
    except Exception as e:
        doc["fail_reason"] = f"{doc.get('fail_reason')} | recovery failed: {e}"


async def put_back(rig, height: float, grip: float) -> tuple:
    """Carry the held object back to rig.picked_at and release it there at table height; raises if unknown."""
    if rig.picked_at is None:
        raise RuntimeError("still holding: pick spot unknown, not opening the gripper")
    x, y, theta = rig.picked_at
    await ppa.travel(rig, x, y, theta=theta)
    await ppa.release_here(rig, height, grip)
    return x, y


DROP_CLEAR_MM = (83.0, 60.0)  # clearance around a drop spot: SPOT_CLEAR_PX at the observe depth, then relaxed
DROP_GRID_MM = 60.0  # candidate spacing over the reachable table
DROP_REACH_MM = (170.0, 560.0)  # flange distance from the base for a release near the table
# Table area known to be table: survey free spots spanned x 240..535, y -342..430; objects stood at x >= 166
DROP_X, DROP_Y = (150.0, 560.0), (-440.0, 440.0)
DROP_TRIES = 6  # live looks per clearance before relaxing
SQUARE_AVOID_MM = 180.0  # never drop within this of the square's centre (hole ~140 x 190 mm, tape, paper)
SQUARE_FAR_MM = 350.0  # beyond this from the square, nearer to the arm wins


def square_sightings(rig) -> list:
    """
    [(xy, source)] where the square was seen, newest first, each < SQUARE_CACHE_S old: LAST_SQUARE (executor
    frames, a full view: "cache") and the live map's square (every module frame: "map", or "map-partial" when not
    all four tape sides were seen yet; its xy is then drop_square's estimate of the full centre, a guess to look at,
    never a place target).

    WHY (2026-10-29): the user moved the paper next to the screwdrivers and a grasp still used square_path
    "cache" at the old spot; the live map had already seen it elsewhere. The paper moves at will, so these are
    only where place_in_square looks first; it releases only after a full view of the interior.
    {AGENT3 2026-10-29 "PARTIAL == TRUE ... XY / SIZE_MM ARE ONLY THE VISIBLE PART, NOT THE HOLE CENTRE"}
    [CONFIDENCE: SINGLE-SRC 85% — AGENT3's map format]
    """
    seen = []
    if LAST_SQUARE["xy"] is not None:
        seen.append((float(LAST_SQUARE["ts"]), tuple(LAST_SQUARE["xy"]), "cache"))
    sq = (live_map(rig) or {}).get("square")
    if sq and sq.get("xy"):
        seen.append((float(sq.get("seen_ts") or 0.0), tuple(sq["xy"]), "map-partial" if sq.get("partial") else "map"))
    now = time.time()
    return [(xy, src) for ts, xy, src in sorted(seen, reverse=True) if now - ts <= SQUARE_CACHE_S]


def partial_square_xy(found: dict, rig):
    """
    Full-centre estimate (AGENT3's drop_square) of a square the found frame shows only cut off, or None; never fails.

    WHY (2026-10-29): after a restart the map had no square and the black tape frame showed the paper only in part,
    so the carry went straight to the z 580 search (and its far start was out of reach). This is where to look first.
    """
    if found.get("square") or found.get("depth") is None:
        return None
    try:
        hit = perception.find_drop_square(found["rgb"], found["depth"])
        if not hit:
            return None
        sq = drop_square.derive(drop_square.update(None, drop_square.observe(hit, found["pose"], intr(rig)), time.time()))
        return tuple(sq["xy"]) if sq and sq.get("xy") else None
    except Exception as e:
        print(f"  partial square estimate failed: {e}")
        return None


def known_square(rig) -> tuple:
    """(xy, source) of the newest square sighting (square_sightings), or (None, None)."""
    seen = square_sightings(rig)
    return seen[0] if seen else (None, None)


def live_map(rig):
    """The module's live table map via rig.live_map(), or None; never fails."""
    try:
        return rig.live_map() if getattr(rig, "live_map", None) else None
    except Exception as e:
        print(f"  live map unavailable: {e}")
        return None


def drop_candidates(rig, clear_mm: float, square_xy, avoid: list) -> list:
    """
    Flange xy drop spots over the whole reachable table, best first.

    Skips keep-outs, the square (SQUARE_AVOID_MM) and anything within clear_mm of an `avoid` (x, y, radius)
    (mapped objects, spots already refused). Score: distance from the square up to SQUARE_FAR_MM, minus 0.3 x
    travel from the arm, so spots far from the square come first and nearby ones break ties.
    """
    p = rig.last_pose
    here = (p.x, p.y) if p is not None else ppa.HOME_XY
    out = []
    for x in np.arange(DROP_X[0], DROP_X[1] + 1.0, DROP_GRID_MM):
        for y in np.arange(DROP_Y[0], DROP_Y[1] + 1.0, DROP_GRID_MM):
            x, y = float(x), float(y)
            if not DROP_REACH_MM[0] <= math.hypot(x, y) <= DROP_REACH_MM[1] or not ppa.allowed(x, y):
                continue
            d_sq = math.hypot(x - square_xy[0], y - square_xy[1]) if square_xy is not None else SQUARE_FAR_MM
            if d_sq < SQUARE_AVOID_MM:
                continue
            if any(math.hypot(x - ax, y - ay) < clear_mm + ar for ax, ay, ar in avoid):
                continue
            out.append((min(d_sq, SQUARE_FAR_MM) - 0.3 * math.hypot(x - here[0], y - here[1]), x, y))
    return [(x, y) for _, x, y in sorted(out, reverse=True)]


async def drop_somewhere(rig, height: float, grip: float) -> tuple:
    """
    Put the held object at a free spot; returns (x, y). Raises (still holding) if none passes.

    WHY (2026-10-29): a reset on the cluttered table found nothing: candidates came from a stale survey list
    and every spot needed ~83 mm clear. Now candidates cover the whole reachable table, skip what the live map
    knows is there, and each is checked live (place_at -> spot_free on the current depth); after DROP_TRIES
    refusals the clearance relaxes to 60 mm.
    """
    square_xy, _ = known_square(rig)
    m = live_map(rig)
    avoid = [(float(e["xy"][0]), float(e["xy"][1]), 0.5 * max(e.get("size_mm") or [0.0, 0.0]))
             for e in (m or {}).get("objects", []) if e.get("xy")]
    for clear_mm in DROP_CLEAR_MM:
        refused = []
        for _ in range(DROP_TRIES):
            cands = drop_candidates(rig, clear_mm, square_xy, avoid + refused)
            if not cands:
                break
            x, y = cands[0]
            print(f"reset: trying ({x:.0f},{y:.0f}) with {clear_mm:.0f} mm clear")
            if await ppa.place_at(rig, x, y, height, grip, clear_mm=clear_mm):
                return x, y
            refused.append((x, y, 0.0))
    raise RuntimeError("no free spot to drop the object; still holding it")


# Frame-based drop spots for the reset (drop_spots_in_frame). The knife (160 x 25 mm) found no 166 mm free disc on
# the cluttered table and went back into the square, though a ~100 x 200 mm patch below the square was empty.
# {VIAM CENTERAL 2026-10-29 "空位候选用 LIVE MAP 加当前帧的 FREE MASK ... 长物体允许旋转 WRIST，让长轴沿空隙方向"}
# [CONFIDENCE: INFERRED 70% — offline on real frames; not run on the arm]
DROP_ANGLES = (0.0, 45.0, 90.0, 135.0)  # image angles tried for the object's long axis
DROP_MARGIN_MM = 12.0  # clear table around the footprint
SQUARE_KEEP_MM = 60.0  # never drop within this of the square's interior (tape ring ~40 mm + paper edge)
DISCARD_XY = [(280.0, 0.0), (280.0, -120.0), (280.0, 120.0)]  # last resort, below/left of the observe view
DISCARD_CHECK_MM = 30.0  # the discard spot only has to be clear this close to the jaws


FINGER_W_MM = 20.0  # finger width along the object's long axis
FINGER_T_MM = 8.0  # finger thickness beyond the full jaw opening


def drop_spots_in_frame(rgb: np.ndarray, depth: np.ndarray, pose: dict, k: dict, size_mm: tuple,
                        sq: dict = None, near_xy: tuple = None, n: int = 4) -> list:
    """
    Up to n drop spots seen free in this empty-gripper frame: [(table xy, image angle of the long axis, why, frame
    pose)], nearest to near_xy first.

    Free = nothing raised above the fitted table, nothing far below it (the floor past the table's edge read as free
    on 084) and depth readable (each opened by 2 px against noise); flat tape,
    paper or wood grain do not block (only raised things can stop the object or the opening fingers); the arm's
    hanging cable is raised, so it blocks. The square's interior grown by SQUARE_KEEP_MM is never free. Two
    rectangles must be free around the spot, at DROP_ANGLES for the long axis, every 16 px on an 8 px sample grid:
    the object (length capped at 200 mm, plus DROP_MARGIN_MM) and the finger strip across its middle (the full jaw
    opening + finger thickness, FINGER_W_MM wide). The first version also counted bright wood as paper and took
    the finger strip as the object's width: 0 spots on 076/080/084 for the knife.
    """
    if depth is None:
        return []
    h, w = depth.shape
    plane = table_plane(depth)
    table = float(np.median(plane))
    mm = k["fx"] / table
    up = (depth > 0) & (plane - depth > ppa.SPOT_MIN_H_MM)
    off = (depth > 0) & (depth - plane > 8.0)  # below the table plane: the floor past the table's edge (084)
    busy = (ndimage.binary_opening(up, iterations=2) | ndimage.binary_opening(depth == 0, iterations=2)
            | ndimage.binary_opening(off, iterations=2))
    free = ~busy
    free[ppa.FINGER_ROW:] = False
    if sq is not None and sq.get("corners"):
        keep = square_mask(sq, pose, k, rgb.shape, table)
        free &= ~ndimage.binary_dilation(keep, iterations=int(SQUARE_KEEP_MM * mm))
    across, along = size_mm
    rects = [((min(along, 200.0) + 2 * DROP_MARGIN_MM) * mm, (across + 2 * DROP_MARGIN_MM) * mm),
             ((FINGER_W_MM + 2 * DROP_MARGIN_MM) * mm, (2 * SPOT_JAW_MM + 2 * FINGER_T_MM) * mm)]
    cu, cv = np.meshgrid(np.arange(0, w, 16), np.arange(0, h, 16))
    cu, cv = cu.ravel(), cv.ravel()
    found = []
    for ang in DROP_ANGLES:
        a = math.radians(ang)
        lu, lv = math.cos(a), -math.sin(a)  # long axis in (u, v); image yaw convention atan2(-dv, du)
        fits = np.ones(len(cu), bool)
        for L, W in rects:
            su, sv = np.meshgrid(np.arange(-L / 2, L / 2 + 1, 8.0), np.arange(-W / 2, W / 2 + 1, 8.0))
            ou = (su * lu - sv * lv).ravel()  # across axis = long axis turned 90 deg: (-lv, lu)
            ov = (su * lv + sv * lu).ravel()
            uu = (cu[:, None] + ou[None, :]).round().astype(int)
            vv = (cv[:, None] + ov[None, :]).round().astype(int)
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            ok = np.zeros(uu.shape, bool)
            ok[inside] = free[vv[inside], uu[inside]]
            fits &= ok.all(axis=1)
        for i in np.nonzero(fits)[0]:
            u, v = float(cu[i]), float(cv[i])
            xy = perception.world_xy(pose, k, u, v, float(plane[int(v), int(u)]))
            found.append((tuple(float(c) for c in xy), ang, u, v))
    if not found:
        return []
    ref = near_xy or (pose["x"], pose["y"])
    found.sort(key=lambda f: math.hypot(f[0][0] - ref[0], f[0][1] - ref[1]))
    out = []
    for xy, ang, u, v in found:
        # Spread the picks: skip spots within 40 mm of one already chosen
        if all(math.hypot(xy[0] - o[0][0], xy[1] - o[0][1]) > 40.0 for o in out):
            out.append((xy, ang, f"free {along:.0f} x {across:.0f} mm footprint at image angle {ang:.0f} ({u:.0f},{v:.0f})",
                        pose))
        if len(out) >= n:
            break
    return out


async def drop_at(rig, height: float, grip: float, spots: list) -> tuple:
    """
    Put the held object at the first spot of drop_spots_in_frame() that still looks clear at the jaws, with the
    wrist turned so its long axis (image-vertical since align_wrist) lies along the spot's angle; then the
    DISCARD_XY spots at theta 0. Returns the xy used; raises (still holding) if none.
    {VIAM CENTERAL 2026-10-29 "实在找不到时，退到一个固定的 DISCARD 区 ... 不要放回方框"}
    """
    k = intr(rig)
    table = ppa.TABLE_DEPTH_MM
    tries = [(xy, ang, fp) for xy, ang, _, fp in spots] + [(xy, None, None) for xy in DISCARD_XY]
    for xy, ang, fp in tries:
        theta = 0.0
        if ang is not None:
            d = table + (fp["z"] - ppa.ALIGN_Z)  # the table's depth in that frame (z 580 views are ~180 mm farther)
            u, v = perception.project_xy(fp, k, xy, d)
            theta = wrist_for(fp, k, u, v, ang, d)
        # Flange that puts the jaws over xy at this theta (the jaw offset turns with the wrist)
        au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, table)
        jx, jy = perception.world_xy({"x": 0.0, "y": 0.0, "z": ppa.ALIGN_Z, "theta": theta}, k, au, av, table)
        fx, fy = xy[0] - jx, xy[1] - jy
        if not ppa.reachable(fx, fy) or math.hypot(fx, fy) < 150.0:
            continue
        try:
            await ppa.travel(rig, fx, fy, theta=theta)
        except ppa.MoveRefused as e:
            print(f"  drop: ({xy[0]:.0f},{xy[1]:.0f}) refused: {e}")
            continue
        rgb, depth = await rig.frame()
        if depth is None:
            continue
        t, _ = ppa.table_and_near(depth)
        hint = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, t)
        # Only what is right under the jaws now (a hand, something moved there): the footprint was checked above
        free, why = ppa.spot_free(rgb, depth, hint, radius=DISCARD_CHECK_MM * rig.fx / t)
        rig.record("drop", rgb, xy=[round(xy[0]), round(xy[1])], theta=round(theta), free=free, why=why,
                   discard=ang is None)
        if not free and "near the tape square" not in why:
            print(f"  drop: ({xy[0]:.0f},{xy[1]:.0f}) not clear now: {why}")
            continue
        await ppa.release_here(rig, height, grip)
        return xy
    raise RuntimeError("no free spot to drop the object (frame spots and discard area); still holding it")


async def wipe(rig) -> None:
    """
    Push whatever is in the hole out sideways with the closed gripper (fallback when picking fails).

    One slow straight pass along world y (sideways to the arm: never toward the base or away into the room),
    fingertips WIPE_CLEAR_MM above the measured table, toward the side whose depth shows fewer objects.
    Starts with the arm centred over the hole (after the square servo). Not yet run on hardware.
    """
    p = await rig.pose()
    z1, table, _ = await rig.measure()
    rgb, depth = await rig.frame()
    near = (depth > 0) & (depth < table - 8)
    w = near.shape[1]
    # Image +u is world -y (measured on sfhack4), so the image's right half is the -y side
    left_busy, right_busy = int(near[:, : w // 3].sum()), int(near[:, 2 * w // 3:].sum())
    direction = 1.0 if left_busy <= right_busy else -1.0  # +y = toward the image's left
    z_wipe = max(z1 - (table - ppa.TIP_BELOW_CAM - WIPE_CLEAR_MM), rig.floor_z)
    start, end = p.y - direction * 90.0, p.y + direction * 180.0
    print(f"  wipe: y {start:.0f} -> {end:.0f} at z={z_wipe:.0f}")
    await ppa.travel(rig, p.x, start)
    await rig.grab()  # close the jaws into a flat pusher; returns False with nothing inside, as expected
    await rig.wait_clear()
    await rig.descend(z_wipe)
    y = start
    while (end - y) * direction > 0:
        y = y + direction * min(WIPE_STEP_MM, abs(end - y))
        await rig.goto(p.x, y, z_wipe)
    await rig.lift()
    await rig.open()


async def go_over_square(rig) -> tuple:
    """
    Empty gripper: leave the arm at ALIGN_Z, theta 0, with the square's full interior centred in view (the pose
    perception.look_flange_xy gives for its centre); returns (centre table xy, frame pose). Raises SquareNotFound.

    WHY (2026-10-29): a reset failed with "square: not visible": search_square + a servo on find_square, which
    reads the silver knife in the square as tape (AGENT3: 52-69% of its pixels) and pulls the square into the
    fingertip rows. Now as for the place (62e3a76): drop_square's estimate from the current frame, then the recent
    sightings, then the observe pose before any z 580 pose; each estimate gets one move to centre it, and a second
    look must show the full interior there (not partial, centre within 15 mm).
    {VIAM CENTERAL 2026-10-29 "请改成用 AGENT3 的 DROP_SQUARE 估计或 LIVE MAP 的 SQUARE，跟 PLACE 那边用同一套"}
    [CONFIDENCE: INFERRED 75% — offline on frame 084 and a fake rig; not run on the arm]
    """
    k = intr(rig)

    async def look():
        rgb, depth = await rig.frame()
        q = pose_dict(await rig.pose())
        return frame_square(rgb, depth, q, k), q

    async def centre_on(xy):
        # Up to two moves: centre the estimate, then confirm a full, stable view of the interior there
        for _ in range(2):
            fx, fy = perception.look_flange_xy(tuple(xy))
            if not ppa.reachable(fx, fy):
                return None
            p = await rig.pose()
            if math.hypot(p.x - fx, p.y - fy) > 5.0 or abs(p.z - ppa.ALIGN_Z) > 3.0 or abs(p.theta) > 1.0:
                try:
                    await ppa.travel(rig, fx, fy, theta=0.0)
                except ppa.MoveRefused as e:
                    print(f"  square: look pose refused: {e}")
                    return None
            sq, q = await look()
            if sq is None:
                return None
            if not sq.get("partial") and math.hypot(sq["xy"][0] - xy[0], sq["xy"][1] - xy[1]) <= 15.0:
                return tuple(sq["xy"]), q
            xy = sq["xy"]
        return None

    first, _ = await look()
    guesses = ([first["xy"]] if first else []) + [xy for xy, _ in square_sightings(rig)]
    for g in guesses:
        hit = await centre_on(g)
        if hit:
            LAST_SQUARE.update(xy=hit[0], ts=time.time())
            return hit
    poses = [(*ppa.HOME_XY, ppa.ALIGN_Z)] + [(x, y, SWEEP_Z) for x, y in SWEEP_XY]
    poses = [q for q in poses if ppa.reachable(*q)]
    for x, y, z in poses:
        try:
            await ppa.travel(rig, x, y, theta=0.0, z=z)
        except ppa.MoveRefused as e:
            print(f"  square search: skipping ({x:.0f},{y:.0f},{z:.0f}): {e}")
            continue
        sq, _ = await look()
        if sq is not None:
            hit = await centre_on(sq["xy"])
            if hit:
                LAST_SQUARE.update(xy=hit[0], ts=time.time())
                return hit
    raise ppa.SquareNotFound("square: no full view of its interior anywhere on the table")


async def clear_square(rig, store) -> dict:
    """
    Empty the blue square. Returns the logged attempt doc (mode "reset"); success = interior empty by depth.

    Locates the square with go_over_square() (drop_square, no servo on the tape), counts what stands on its
    estimated interior (objects_in_hole with the frame pose), picks each out (<= 2 tries) and puts it at a free
    spot seen free in an empty-gripper view (drop_spots_in_frame, else DISCARD_XY: never back into the square); a failed pick -> wipe (<= 2 passes). Raises nothing for ordinary failures; status
    "failed" means it parked and the next trial must not start.
    """
    doc = {"ts": time.time(), "mode": "reset", "target_text": "__reset__", "success": False, "status": "failed",
           "removed": 0, "wipes": 0, "fail_phase": None, "fail_reason": None}
    t0 = time.monotonic()
    STATE.update(times={"at": "reset"}, error=None)
    start_recording(rig, store, doc)
    k = intr(rig)
    dropped = []
    try:
        # The view the reset starts from (normally the observe pose, empty gripper): more table to drop on
        rgb0, depth0 = await rig.frame()
        first = (rgb0, depth0, pose_dict(await rig.pose()))
        centre, _ = await go_over_square(rig)
        for _round in range(6):
            rgb, depth = await rig.frame()
            q = pose_dict(await rig.pose())
            rig.record("reset", rgb)
            seen, inside, seg, hole = objects_in_hole(rgb, depth, q, k)
            if seg is not None:
                publish_frame(store, rgb, seg, "reset")
            if not seen:
                raise RuntimeError("square interior not in view at its look pose")
            # Done only when nothing stands on the interior at all (a stack segments as one overhanging blob)
            raised = raised_in_hole(depth, hole)
            doc["raised_px"] = raised
            if not inside and raised < SQUARE_CLEAR_RAISED_PX:
                doc.update(success=True, status="done", square_xy=[round(centre[0], 1), round(centre[1], 1)])
                break
            if not inside:
                print(f"reset: no object segmented but {raised} raised px on the interior: wiping")
                if doc["wipes"] >= 2:
                    raise RuntimeError(f"square still not empty after 2 wipes ({raised} raised px)")
                await ppa.travel(rig, *centre, theta=0.0)  # wipe starts with the jaws over the centre
                await wipe(rig)
                doc["wipes"] += 1
                centre, _ = await go_over_square(rig)
                continue
            obj = max(inside, key=lambda o: o.height_mm)  # the top of a stack first
            print(f"reset: {len(inside)} object(s) in the square; removing one ({obj.colour}, {obj.height_mm:.0f} mm)")
            picked = False
            # Where it can go, from views with an empty gripper: this one and the starting view; never where an
            # earlier object of this reset went
            size = (width_mm(obj, rig.fx), obj.length_px * obj.top_depth_mm / rig.fx)
            sq_now = frame_square(rgb, depth, q, k)
            spots = (drop_spots_in_frame(rgb, depth, q, k, size, sq_now, near_xy=centre)
                     + drop_spots_in_frame(*first, k, size, sq_now, near_xy=centre))
            spots = sorted((sp for sp in spots if all(math.hypot(sp[0][0] - d[0], sp[0][1] - d[1]) > 80.0 for d in dropped)),
                           key=lambda sp: math.hypot(sp[0][0] - centre[0], sp[0][1] - centre[1]))
            doc.setdefault("drop_spots", []).append([sp[2] for sp in spots[:4]])
            if width_mm(obj, rig.fx) <= MAX_OPEN_MM or ring_geometry(obj) is not None:
                for _try in range(2):
                    try:
                        height, grip = await ppa.pick(rig, None, ObjTracker(rig, obj, rgb))
                        xy = await drop_at(rig, height, grip, spots[:4])
                        dropped.append(xy)
                        doc.setdefault("dropped_at", []).append([round(xy[0], 1), round(xy[1], 1)])
                        picked = True
                        doc["removed"] += 1
                        break
                    except ppa.PickFailed as e:
                        print(f"reset: pick failed ({e})")
                    await rig.lift()
            if not picked:
                if doc["wipes"] >= 2:
                    raise RuntimeError("square still not empty after 2 wipes")
                await ppa.travel(rig, *centre, theta=0.0)
                await wipe(rig)
                doc["wipes"] += 1
            # Back over the square to look again (the paper may have been pushed)
            centre, _ = await go_over_square(rig)
        if not doc["success"]:
            doc.update(fail_phase="verify", fail_reason="objects still in the square")
        doc["img_after"] = jpeg((await rig.frame())[0])
    except Exception as e:
        doc.update(fail_phase=doc.get("fail_phase") or "reset", fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    doc["duration_s"] = round(time.monotonic() - t0, 1)
    finish(rig, store, doc)
    STATE.update(times=None, error=None if doc["success"] else doc["fail_reason"])
    await back_to_observe(rig)
    return doc


async def resume(rig, store):
    """
    After a restart: if the gripper still holds something, put it in the square first. Returns the doc or None.

    WHY: the SDK can abort the process mid-carry; the supervisor restarts it with the object still held.
    Height/grip come from data/arm_state.json (pick_place_auto.save_state); defaults fit the yellow block.
    """
    st = ppa.load_state()
    if not await rig.holding():
        if st.get("holding"):
            ppa.save_state(False)
        return None
    print("resume: gripper is holding something; placing it in the square first")
    height, grip = st.get("height") or 32.0, st.get("grip") or 15.0
    doc = {"ts": time.time(), "mode": "resume", "target_text": "held after restart", "success": False,
           "status": "failed", "fail_phase": None, "fail_reason": None, "height_mm": height}
    times = {}
    STATE.update(times=times, error=None)
    start_recording(rig, store, doc)
    t0 = time.monotonic()
    try:
        await ppa.place_in_square(rig, height, grip, times)
        times["at"] = "check"
        rgb, depth = await rig.frame()
        seen, inside, _, _ = objects_in_hole(rgb, depth)
        doc["success"] = bool(seen and inside)
        doc["status"] = "done" if doc["success"] else "failed"
    except Exception as e:
        doc.update(fail_phase=times.get("at", "start"), fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    times.pop("at", None)
    doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
    finish(rig, store, doc)
    STATE.update(times=None, error=doc["fail_reason"])
    return doc
