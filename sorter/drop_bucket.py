"""
Put the held object into the orange bucket: where it is (found frame, live map, or a look around while carrying), then over
its mouth high enough that the object clears the rim, and let go.

WHY: the user replaced the blue square with a ~180 mm tall Halloween bucket and moves it around; nothing is lowered into it.
{VIAM CENTERAL 2026-10-30 "找到桶 → 抓起 → 飞到桶口上方 Z400 → 松手"}
[CONFIDENCE: SINGLE-SRC 70% — offline only; the bucket's position comes from AGENT3's sorter.bucket on two real frames]
Flow: executor.execute -> (after the pick) known_bucket() -> place_in_bucket() -> release_jaws over the mouth.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402

from sorter import bucket  # noqa: E402

RIM_CLEAR_MM = 25.0  # the held object's bottom passes this far above the rim
MOUTH_MARGIN_MM = 35.0  # the jaws let go this far inside the rim at least: the object falls in, not onto the rim
BUCKET_MAX_AGE_S = 120.0  # a map sighting older than this may be from before the bucket was moved


def enabled() -> bool:
    """SORTER_TARGET=bucket (sorter.bucket's default): a bucket in view is the drop target."""
    return bucket.TARGET == "bucket"


def from_frame(rgb, depth, pose: dict, k: dict):
    """The bucket in one frame ({xy, radius_mm, rim_height_mm, partial, confidence}), or None; never fails."""
    if rgb is None or depth is None:
        return None
    try:
        return bucket.find_bucket(rgb, depth, pose, k)
    except Exception as e:
        print(f"  bucket estimate failed: {e}")
        return None


def known_bucket(found: dict, live: dict, k: dict):
    """Where the bucket is without moving: the found frame, else the live map's recent entry; None if neither has one."""
    b = from_frame(found.get("rgb"), found.get("depth"), found.get("pose"), k) if found else None
    if b is not None:
        return dict(b, source="frame")
    m = (live or {}).get("bucket")
    if m and time.time() - float(m.get("seen_ts") or 0.0) <= BUCKET_MAX_AGE_S:
        return dict(m, source="map")
    return None


def release_z(b: dict, height: float, grip: float) -> float:
    """
    Flange z at which the held object's bottom is RIM_CLEAR_MM over the rim: at least ALIGN_Z. At ALIGN_Z the fingertips are
    TABLE_DEPTH_MM - TIP_BELOW_CAM (~241 mm) over the table, and the object hangs (height - grip) below them.
    """
    tips_at_align = ppa.TABLE_DEPTH_MM - ppa.TIP_BELOW_CAM
    need = float(b.get("rim_height_mm") or bucket.RIM_HEIGHT_MM) + RIM_CLEAR_MM + (height - grip)
    return max(ppa.ALIGN_Z, ppa.ALIGN_Z + need - tips_at_align)


def release_xy(b: dict, z: float):
    """
    Flange xy (theta 0: the jaw axis is the flange xy) inside the mouth by MOUTH_MARGIN_MM, the centre if the arm reaches it
    there, else the reachable point nearest the centre; None if no point of the mouth is reachable at z.
    """
    c = np.array(b["xy"], float)
    r = max(0.0, float(b.get("radius_mm") or bucket.RADIUS_MM) - MOUTH_MARGIN_MM)
    best = None
    for dx in np.arange(-r, r + 0.1, 5.0):
        for dy in np.arange(-r, r + 0.1, 5.0):
            if math.hypot(dx, dy) > r:
                continue
            x, y = c[0] + dx, c[1] + dy
            if not ppa.reachable(x, y, z) or math.hypot(x, y) < 150.0:
                continue
            d = math.hypot(dx, dy)
            if best is None or d < best[0]:
                best = (d, float(x), float(y))
    return None if best is None else (best[1], best[2])


async def look_for_bucket(rig, k: dict, poses: list):
    """Carrying: look from each reachable pose (nearest first) until a frame shows the bucket; None if none does."""
    p = await rig.pose()
    at = (p.x, p.y)
    left = [q for q in poses if ppa.reachable(*q) and math.hypot(q[0], q[1]) >= 150.0]
    while left:
        x, y, z = min(left, key=lambda q: (q[2] > ppa.ALIGN_Z + 1, math.hypot(q[0] - at[0], q[1] - at[1])))
        left.remove((x, y, z))
        at = (x, y)
        await ppa.travel(rig, x, y, theta=0.0, z=z)
        q = await rig.pose()
        rgb, depth = await rig.frame()
        b = from_frame(rgb, depth, {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta}, k)
        if b is not None:
            print(f"  bucket: seen from ({x:.0f},{y:.0f},{z:.0f}) at ({b['xy'][0]:.0f},{b['xy'][1]:.0f})")
            return dict(b, source="search")
    return None


async def place_in_bucket(rig, height: float, grip: float, times: dict, b: dict) -> dict:
    """
    Carry the held object over the bucket `b` and let go. Returns {"xy" (flange xy of the release), "z", "off_mm" (from the
    bucket's centre)}; raises ppa.SquareNotFound (still holding) if the arm reaches no point of the mouth.
    Up first where it is, then across: the object then crosses the rim already above it. No hand check over the mouth: the
    bucket is in the depth sensor's blind range there and read as an intruder.
    """
    t = time.monotonic()
    times["at"] = "carry"
    z = release_z(b, height, grip)
    xy = release_xy(b, z)
    if xy is None:
        raise ppa.SquareNotFound(f"bucket at ({b['xy'][0]:.0f},{b['xy'][1]:.0f}): no point of its mouth within reach at z {z:.0f}")
    print(f"3/5 carry to the bucket ({b.get('source')}): centre ({b['xy'][0]:.0f},{b['xy'][1]:.0f}), "
          f"release at ({xy[0]:.0f},{xy[1]:.0f}) z {z:.0f}")
    p = await rig.pose()
    if p.z < z - 1.0:
        await rig.rise(z, settle=False)
    await ppa.travel(rig, xy[0], xy[1], theta=0.0, z=z, settle=False)
    times["carry"] = round(time.monotonic() - t, 1)
    times["at"] = "place"
    print("4/5 release over the bucket")
    await ppa.release_jaws(rig)
    ppa.save_state(False)
    times["place"] = round(time.monotonic() - t - times["carry"], 1)
    return {"xy": xy, "z": z, "off_mm": math.hypot(xy[0] - b["xy"][0], xy[1] - b["xy"][1])}
