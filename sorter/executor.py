"""
Pick ANY object on the table (sorter.perception.Obj) and put it in the blue tape square; clear the square.

WHY: the demo is "grasp green square" with whatever is on the table, and the square must be emptied
between runs without a person. The arm mechanics (servo, depth heights, reconnect, intruder pause,
square search) live in scripts/pick_place_auto.py; this module adds object choice and tracking for
arbitrary objects, the in-square check, and the reset.

Flow, execute(text): observe -> perception.segment_full -> clip_embed.best -> ObjTracker -> ppa.pick ->
ppa.place_in_square -> objects_in_hole check -> Store.log_attempt.
Flow, clear_square(): find square -> each object in the hole: pick (<= 2 tries) -> place_at random free spot;
failed pick -> wipe (<= 2 passes) -> verify hole empty by depth -> Store.log_attempt(mode="reset").
Contract: docs/EXEC_PLAN.md (Command flow, Reset).
"""
import io
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402  arm library (Rig, pick, place_in_square, place_at, ...)

from sorter import clip_embed, perception  # noqa: E402

MAX_OPEN_MM = 78.0  # jaws open ~84 mm; keep a margin so the fingers clear the sides
COLOUR_TOL = 45.0  # RGB distance to the object's median colour that still counts as the object (close range)
# Where to look for objects, flange x/y mm: home first, then the square-search grid
OBSERVE_XY = [ppa.HOME_XY] + ppa.SEARCH_XY
# Random drop area for reset, flange x/y mm (see run_trials.RESET_*); place_at refuses occupied/square spots
RESET_X = (230.0, 380.0)
RESET_Y = (-220.0, 220.0)
WIPE_CLEAR_MM = 10.0  # fingertips this far above the table during a wipe
WIPE_STEP_MM = 20.0  # wipe in short moves = slow, and interruptible between steps


class ObjTracker:
    """
    Finder (rgb -> (u, v, yaw, area) or None) that follows one object of any colour.

    Far (depth valid, e.g. at ALIGN_Z): nearest object from perception.segment_full.
    Close (depth blind < ~280 mm, during the close-range servo): nearest blob of pixels near the object's
    own median colour. Gives up (None) if the nearest candidate jumped more than MAX_JUMP_PX, i.e. the
    object was taken or moved by a person; the caller then re-acquires.
    Limitation: at close range a table-coloured object cannot be told from the table.
    """

    MAX_JUMP_PX = 260.0

    def __init__(self, rig, obj, rgb: np.ndarray):
        self.rig = rig
        self.uv = obj.centroid_px
        self.colour = np.median(rgb[obj.mask].astype(float), axis=0)

    def _candidates(self, rgb: np.ndarray) -> list:
        depth = self.rig.last_depth
        if depth is not None and (depth > 0).mean() > 0.6:
            seg = perception.segment_full(rgb, depth)
            return [(o.centroid_px[0], o.centroid_px[1], o.yaw_deg, int(o.mask.sum())) for o in seg.objects]
        m = np.linalg.norm(rgb.astype(float) - self.colour, axis=2) < COLOUR_TOL
        m[ppa.FINGER_ROW:] = False
        m = ndimage.binary_opening(m, iterations=2)
        lab, n = ndimage.label(m)
        out = []
        for k in range(1, n + 1):
            vs, us = np.nonzero(lab == k)
            if len(us) < 1500:
                continue
            cu, cv = us.mean(), vs.mean()
            w, vec = np.linalg.eigh(np.cov(np.vstack([us - cu, vs - cv])))
            out.append((cu, cv, math.degrees(math.atan2(-vec[1, 1], vec[0, 1])) % 180.0, len(us)))
        return out

    def __call__(self, rgb: np.ndarray):
        cands = self._candidates(rgb)
        if not cands:
            return None
        best = min(cands, key=lambda c: math.hypot(c[0] - self.uv[0], c[1] - self.uv[1]))
        if math.hypot(best[0] - self.uv[0], best[1] - self.uv[1]) > self.MAX_JUMP_PX:
            return None
        self.uv = (best[0], best[1])
        return best


def hole_mask(rgb: np.ndarray):
    """Bool mask of the tape hole incl. anything standing in it (even a blue object), or None if not in view."""
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
        return None
    # Fill so that an object inside the hole that is itself blue still counts as inside
    return ndimage.binary_fill_holes(lab == best[0])


def objects_in_hole(rgb: np.ndarray, depth: np.ndarray) -> tuple:
    """(hole visible?, [Obj inside the hole]) from a frame taken above the square. Inside = >= half the mask."""
    hole = hole_mask(rgb)
    if hole is None or depth is None:
        return False, []
    _, objs = perception.segment(rgb, depth)
    return True, [o for o in objs if (o.mask & hole).sum() >= 0.5 * o.mask.sum()]


def jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def width_mm(obj, fx: float) -> float:
    """Object width across the jaws (short side) in mm, from pixels at its top-surface depth."""
    return obj.width_px * obj.top_depth_mm / fx


async def find_target(rig, text: str, store) -> dict:
    """
    Look from the current pose, then OBSERVE_XY, for the object best matching `text` (outside the square).

    Returns clip_embed.best()'s dict plus "rgb" of the frame it was found in; obj None if no match anywhere.
    """
    p = await rig.pose()
    poses = [None] + sorted(OBSERVE_XY, key=lambda q: math.hypot(q[0] - p.x, q[1] - p.y))
    last = {"obj": None, "reason": "nothing seen"}
    for xy in poses:
        if xy is not None:
            await ppa.travel(rig, *xy)
        rgb, depth = await rig.frame()
        seg = perception.segment_full(rgb, depth)
        hole = hole_mask(rgb)
        # Never pick what is already in the square, or something cut by the image border
        objs = [o for o in seg.objects if not o.touches_edge and (hole is None or (o.mask & hole).sum() < 0.5 * o.mask.sum())]
        if not objs:
            continue
        pick = clip_embed.best(objs, text, store)
        pick["rgb"] = rgb
        if pick.get("obj") is not None:
            return pick
        last = pick
    return last


async def execute(rig, text: str, store, mode: str = "demo") -> dict:
    """
    One command: put the object described by `text` into the blue square. Returns the logged attempt doc.

    Never raises for an ordinary failure: status is done / not_found / ambiguous / too_wide / failed.
    """
    doc = {"ts": time.time(), "mode": mode, "target_text": text, "success": False, "status": "failed",
           "fail_phase": None, "fail_reason": None}
    times = {}
    t0 = time.monotonic()
    try:
        times["at"] = "find"
        found = await find_target(rig, text, store)
        obj = found.get("obj")
        if obj is None:
            doc.update(status="not_found", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        doc.update(label=obj.colour or "object", score=round(float(found.get("score", 0.0)), 3),
                   margin=round(float(found.get("margin", 0.0)), 3), img_before=jpeg(found["rgb"]))
        if found.get("ambiguous"):
            doc.update(status="ambiguous", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        if width_mm(obj, rig.fx) > MAX_OPEN_MM:
            doc.update(status="too_wide", fail_phase="find", fail_reason=f"{width_mm(obj, rig.fx):.0f} mm across")
            return doc
        tracker = ObjTracker(rig, obj, found["rgb"])
        height, grip = await ppa.pick(rig, times, tracker)
        doc["height_mm"], doc["grasp"] = round(height, 1), {"grip_depth": round(grip, 1), "theta": rig.theta}
        await ppa.place_in_square(rig, height, grip, times)
        times["at"] = "check"
        rgb, depth = await rig.frame()
        seen, inside = objects_in_hole(rgb, depth)
        doc["img_after"] = jpeg(rgb)
        doc["success"] = bool(seen and inside)
        doc["status"] = "done" if doc["success"] else "failed"
        if not doc["success"]:
            doc.update(fail_phase="check", fail_reason="nothing seen inside the hole" if seen else "hole not visible")
    except Exception as e:
        doc.update(status="failed", fail_phase=times.get("at", "start"), fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    finally:
        times.pop("at", None)
        doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
        doc["_id"] = store.log_attempt({k: v for k, v in doc.items() if k != "_id"})
    return doc


async def recover(rig, doc: dict) -> None:
    """After a failure: lift; if still holding something, put it down right there so nothing stays mid-air."""
    try:
        await rig.lift()
        if await rig.holding():
            await ppa.release_here(rig, doc.get("height_mm", 30.0), (doc.get("grasp") or {}).get("grip_depth", 15.0))
    except Exception as e:
        doc["fail_reason"] = f"{doc.get('fail_reason')} | recovery failed: {e}"


async def drop_somewhere(rig, height: float, grip: float) -> None:
    """Put the held object at a random free spot (not near the square, no object within ~90 mm)."""
    for _ in range(8):
        x, y = random.uniform(*RESET_X), random.uniform(*RESET_Y)
        if not 150.0 <= math.hypot(x, y) <= 600.0:
            continue
        print(f"reset: trying ({x:.0f},{y:.0f})")
        if await ppa.place_at(rig, x, y, height, grip):
            return
    raise RuntimeError("no free spot to drop the object; still holding it")


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


async def clear_square(rig, store) -> dict:
    """
    Empty the blue square. Returns the logged attempt doc (mode "reset"); success = hole empty by depth.

    Raises nothing for ordinary failures; status "failed" means it parked and the next trial must not start.
    """
    doc = {"ts": time.time(), "mode": "reset", "target_text": "__reset__", "success": False, "status": "failed",
           "removed": 0, "wipes": 0, "fail_phase": None, "fail_reason": None}
    t0 = time.monotonic()
    try:
        await ppa.search_square(rig)
        await rig.servo(ppa.find_square, False, "square")
        for _round in range(6):
            rgb, depth = await rig.frame()
            seen, inside = objects_in_hole(rgb, depth)
            if not seen:
                raise RuntimeError("square not visible after servo")
            if not inside:
                doc.update(success=True, status="done")
                break
            obj = inside[0]
            print(f"reset: {len(inside)} object(s) in the square; removing one ({obj.colour}, {obj.height_mm:.0f} mm)")
            picked = False
            if width_mm(obj, rig.fx) <= MAX_OPEN_MM:
                for _try in range(2):
                    try:
                        height, grip = await ppa.pick(rig, None, ObjTracker(rig, obj, rgb))
                        await drop_somewhere(rig, height, grip)
                        picked = True
                        doc["removed"] += 1
                        break
                    except ppa.PickFailed as e:
                        print(f"reset: pick failed ({e})")
                    await rig.lift()
            if not picked:
                if doc["wipes"] >= 2:
                    raise RuntimeError("square still not empty after 2 wipes")
                await ppa.search_square(rig)
                await rig.servo(ppa.find_square, False, "square")
                await wipe(rig)
                doc["wipes"] += 1
            # Back over the square to look again
            await ppa.search_square(rig)
            await rig.servo(ppa.find_square, False, "square")
        if not doc["success"]:
            doc.update(fail_phase="verify", fail_reason="objects still in the square")
        doc["img_after"] = jpeg((await rig.frame())[0])
    except Exception as e:
        doc.update(fail_phase=doc.get("fail_phase") or "reset", fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    doc["duration_s"] = round(time.monotonic() - t0, 1)
    doc["_id"] = store.log_attempt({k: v for k, v in doc.items() if k != "_id"})
    return doc
