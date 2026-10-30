"""
Put the held object into the orange bucket: where it is (found frame, live map, or a look around while carrying), then over
its mouth high enough that the object clears the rim, and let go.

WHY: the user replaced the blue square with a ~180 mm tall Halloween bucket and moves it around; nothing is lowered into it.
{VIAM CENTERAL 2026-10-30 "找到桶 → 抓起 → 飞到桶口上方 Z400 → 松手"}
{USER 2026-10-30 "MAKE SURE IT CAN WORK WHERE EVER THE BUCKET IS, THE BUCKET CAN MOVE"}
[CONFIDENCE: SINGLE-SRC 70% — offline only; the bucket's position comes from AGENT3's sorter.bucket on real frames]
Flow: executor.execute -> (after the pick) fresh_bucket() or locate() -> place_in_bucket(): over the mouth, a frame there
confirms it (or moves to where the bucket now is), then release_jaws. Never lets go without a bucket estimate under 5 s old.
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
FRESH_S = 5.0  # a bucket estimate older than this is looked for again: people move it
CONFIRM_MM = 30.0  # the frame over the mouth puts it farther than this from the target: move there
LOOK_Z = 580.0  # the whole 197 mm bucket fits one view from here
RELEASE_TRIES = 3  # moves toward a bucket that keeps moving before giving up (still holding)
HINT_SKIP_MM = 150.0  # a search pose this close to the one over the last-known spot shows the same table: not looked from again
_LAST: dict = {}  # the last bucket this process saw ({xy, seen_ts}): where the next search looks first, however old


class BucketOutOfReach(ppa.SquareNotFound):
    """No point of the bucket's mouth is within the arm's reach at the release height; still holding the object."""


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


def fresh_bucket(found: dict, live: dict, k: dict, now: float = None):
    """The bucket from the found frame or the live map if seen under FRESH_S ago, else None (look again: it may have moved)."""
    now = time.time() if now is None else now
    if found and now - float(found.get("ts") or 0.0) <= FRESH_S:
        b = from_frame(found.get("rgb"), found.get("depth"), found.get("pose"), k)
        if b is not None:
            return dict(b, source="frame", seen_ts=float(found["ts"]))
    m = (live or {}).get("bucket")
    if m and now - float(m.get("seen_ts") or 0.0) <= FRESH_S:
        return dict(m, source="map")
    return None


def remember(b) -> None:
    """Keep `b`'s centre as the last-known bucket spot (the first look of the next search)."""
    if b and b.get("xy") is not None:
        _LAST.update(xy=[float(b["xy"][0]), float(b["xy"][1])], seen_ts=float(b.get("seen_ts") or time.time()))


def last_known(live: dict = None):
    """
    Where the bucket was last seen, however long ago: the newer of the live map's entry and the last bucket this process
    saw; None if neither. Old is fine: it is only where the search looks first.
    {USER 2026-10-30 "IT IS ALWAYS SEARCHING A ENTIRE CIRCLE, IT SHOULD FIRST GO TO THE REGION PREVIOSU FIND THEN IF NOT FOUND"}
    """
    cands = [e for e in ((live or {}).get("bucket"), _LAST) if e and e.get("xy") is not None]
    if not cands:
        return None
    xy = max(cands, key=lambda e: float(e.get("seen_ts") or 0.0))["xy"]
    return float(xy[0]), float(xy[1])


def _over(xy):
    """(x, y, LOOK_Z) over `xy`, pulled toward HOME_XY in 25 mm steps until reachable; None if nothing on the way is."""
    hx, hy = ppa.HOME_XY
    x, y = xy
    for _ in range(40):
        if ppa.reachable(x, y, LOOK_Z):
            return x, y, LOOK_Z
        d = math.hypot(x - hx, y - hy)
        if d <= 25.0:
            break
        x, y = x + (hx - x) * 25.0 / d, y + (hy - y) * 25.0 / d
    return None


async def look(rig, k: dict):
    """The bucket in a frame from where the arm is, or None."""
    q = await rig.pose()
    rgb, depth = await rig.frame()
    b = from_frame(rgb, depth, {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta}, k)
    if b is None:
        return None
    b = dict(b, source=f"look ({q.x:.0f},{q.y:.0f},{q.z:.0f})", seen_ts=time.time())
    remember(b)
    return b


async def locate(rig, k: dict, poses: list, hint=None):
    """
    Look for the bucket while holding: from here, then from LOOK_Z over `hint` (where it was last seen, last_known()), then
    from LOOK_Z over here, then from each of `poses` ((x, y, z), nearest first); the first full view wins, else the best
    partial one; None if no view shows it. With a hint, the rise over here and the poses near the hint's are skipped.
    {VIAM CENTERAL 2026-10-30 "先在当前位置看一眼，看不到就升到 Z580 看全景"}
    {USER 2026-10-30 "IT SHOULD FIRST GO TO THE REGION PREVIOSU FIND THEN IF NOT FOUND"}
    [CONFIDENCE: SINGLE-SRC 75% — offline (tests/test_drop_bucket.py); the logs found it at the 3rd-5th pose every time]
    """
    best = await look(rig, k)
    if best is not None and not best["partial"]:
        return best
    p = await rig.pose()
    first = [(p.x, p.y, LOOK_Z)] if ppa.reachable(p.x, p.y, LOOK_Z) else []
    left = [q for q in poses if ppa.reachable(*q) and math.hypot(q[0], q[1]) >= 150.0]
    over = _over(hint) if hint is not None else None
    if over is not None:
        print(f"  bucket: looking first where it was last seen ({hint[0]:.0f},{hint[1]:.0f})")
        await ppa.travel(rig, over[0], over[1], theta=0.0, z=over[2])
        b = await look(rig, k)
        if b is not None and not b["partial"]:
            return b
        if b is not None and (best is None or b["confidence"] > best["confidence"]):
            best = b
        p = await rig.pose()
        first = []  # moved away from "here": its rise would look at the hint's table again
        left = [q for q in left if math.hypot(q[0] - over[0], q[1] - over[1]) > HINT_SKIP_MM]
    at = (p.x, p.y)
    while first or left:
        if first:
            x, y, z = first.pop()
            await rig.rise(z)  # straight up where it is: no sideways leg
        else:
            x, y, z = min(left, key=lambda q: math.hypot(q[0] - at[0], q[1] - at[1]))
            left.remove((x, y, z))
            await ppa.travel(rig, x, y, theta=0.0, z=z)
        at = (x, y)
        b = await look(rig, k)
        if b is not None and not b["partial"]:
            return b
        if b is not None and (best is None or b["confidence"] > best["confidence"]):
            best = b
    return best


def lowest_release_z(b: dict, height: float, grip: float) -> float:
    """The lowest flange z at which the held object's bottom still clears the rim by RIM_CLEAR_MM: for a far bucket."""
    tips_at_align = ppa.TABLE_DEPTH_MM - ppa.TIP_BELOW_CAM
    need = float(b.get("rim_height_mm") or bucket.RIM_HEIGHT_MM) + RIM_CLEAR_MM + (height - grip)
    return ppa.ALIGN_Z + need - tips_at_align


def release_z(b: dict, height: float, grip: float) -> float:
    """
    Flange z at which the held object's bottom is RIM_CLEAR_MM over the rim: at least ALIGN_Z. At ALIGN_Z the fingertips are
    TABLE_DEPTH_MM - TIP_BELOW_CAM (~241 mm) over the table, and the object hangs (height - grip) below them.
    """
    tips_at_align = ppa.TABLE_DEPTH_MM - ppa.TIP_BELOW_CAM
    need = float(b.get("rim_height_mm") or bucket.RIM_HEIGHT_MM) + RIM_CLEAR_MM + (height - grip)
    return max(ppa.ALIGN_Z, ppa.ALIGN_Z + need - tips_at_align)


def release_xy(b: dict, z: float, off=(0.0, 0.0)):
    """
    Flange xy at theta 0 for the release: the held object's centre (`off` from the jaws at theta 0) inside the mouth by
    MOUTH_MARGIN_MM, nearest the bucket's centre among the points the arm reaches at z, kept as far from the base as a grasp
    (MIN_GRASP_R_MM + REACH_MARGIN_MM: the people at the table); None if there is none.
    """
    from sorter.grasp import REACH_MARGIN_MM

    c = np.array(b["xy"], float)
    r = max(0.0, float(b.get("radius_mm") or bucket.RADIUS_MM) - MOUTH_MARGIN_MM)
    near = ppa.MIN_GRASP_R_MM + REACH_MARGIN_MM
    best = None
    # The reach model only orders the points (the planner decides): within it first, then nearest the base
    for dx in np.arange(-r, r + 0.1, 5.0):
        for dy in np.arange(-r, r + 0.1, 5.0):
            d = math.hypot(dx, dy)
            if d > r:
                continue
            x, y = c[0] + dx - off[0], c[1] + dy - off[1]
            if not ppa.allowed(x, y) or not near <= math.hypot(x, y) <= ppa.SAFE_REACH_MM:
                continue
            key = (not ppa.reachable(x, y, z), d if ppa.reachable(x, y, z) else math.hypot(x, y))
            if best is None or key < best[0]:
                best = (key, float(x), float(y))
    return None if best is None else (best[1], best[2])


def where_to_move(b: dict, z: float) -> str:
    """What to tell the person when the arm reaches no point of the mouth: which way to move the bucket and about how far."""
    from sorter.grasp import REACH_MARGIN_MM

    x, y = b["xy"]
    d = math.hypot(x, y)
    (z0, r0), (z1, r1) = ppa.REACH_R_MM
    far = r0 if z <= z0 else r1 if z >= z1 else r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    near = ppa.MIN_GRASP_R_MM + REACH_MARGIN_MM
    mouth = float(b.get("radius_mm") or bucket.RADIUS_MM) - MOUTH_MARGIN_MM
    if d - mouth > far:
        return f"move it about {d - mouth - far + 30:.0f} mm closer to the arm"
    if d + mouth < near:
        return f"move it about {near - d - mouth + 30:.0f} mm away from the arm"
    return "move it out of the arm's keep-out area"


async def carry_over(rig, b: dict, xy, z: float, z_low: float) -> float:
    """
    Up, then across to the release point at z, the object crossing the rim above it; where the planner refuses that, the
    same at z_low, and from a pose too far out to rise in, first in toward the base at this height (not past the bucket).
    Returns the z it got to; raises BucketOutOfReach (still holding) when neither works.
    {VIAM CENTERAL 2026-10-30 "搬运到桶也一样：桶远时，降低搬运高度，前提是高于桶沿 25 MM"}
    """
    for zz in ([z, z_low] if z_low < z - 1.0 else [z]):
        try:
            p = await rig.pose()
            if p.z < zz - 1.0:
                try:
                    await rig.rise(zz, settle=False)
                except ppa.EStop:
                    raise
                except ppa.MoveRefused:
                    q = await rig.pose()
                    r = math.hypot(q.x, q.y)
                    ix, iy = (q.x * (r - 120.0) / r, q.y * (r - 120.0) / r) if r > 400.0 else (q.x, q.y)
                    # The way in must not pass the bucket at this low height
                    t = np.clip(np.dot(np.subtract(b["xy"], (q.x, q.y)), np.subtract((ix, iy), (q.x, q.y)))
                                / max(1.0, math.dist((q.x, q.y), (ix, iy)) ** 2), 0.0, 1.0)
                    closest = np.add((q.x, q.y), t * np.subtract((ix, iy), (q.x, q.y)))
                    if math.dist(closest, b["xy"]) < float(b.get("radius_mm") or bucket.RADIUS_MM) + 60.0:
                        raise
                    await rig.goto(ix, iy, q.z, settle=False)
                    await rig.rise(zz, settle=False)
            await ppa.travel(rig, xy[0], xy[1], theta=0.0, z=zz)
            return zz
        except ppa.EStop:
            raise
        except ppa.MoveRefused as e:
            print(f"  bucket: carry at z {zz:.0f} refused ({e})")
    raise BucketOutOfReach(f"bucket at ({b['xy'][0]:.0f},{b['xy'][1]:.0f}): the arm cannot get over its mouth: {where_to_move(b, z)}")


async def place_in_bucket(rig, height: float, grip: float, times: dict, b: dict, k: dict, poses: list,
                          off=(0.0, 0.0)) -> dict:
    """
    Carry the held object over the bucket `b` and let go. Returns {"xy" (flange xy of the release), "z", "off_mm" (object
    centre from the bucket's centre, by the last look), "looks"}. Still holding, raises BucketOutOfReach (no point of the mouth
    within reach: says which way to move it) or ppa.SquareNotFound (the bucket is nowhere to be seen, or kept moving).
    Up first where it is, then across: the object then crosses the rim already above it. Over the mouth a frame confirms the
    bucket within CONFIRM_MM, else the arm follows it; a frame without it sends the arm up to LOOK_Z over here to look, and a
    look from there that agrees stands as the confirmation. {VIAM CENTERAL 2026-10-30 "松手前再确认一次 ... 不在桶外松手"}
    No hand check over the mouth: the bucket is in the depth sensor's blind range there and read as an intruder.
    """
    t = time.monotonic()
    times["at"] = "carry"
    looks = []
    for _ in range(RELEASE_TRIES):
        z = release_z(b, height, grip)
        xy = release_xy(b, z, off)
        if xy is None:
            raise BucketOutOfReach(f"bucket at ({b['xy'][0]:.0f},{b['xy'][1]:.0f}) is out of the arm's reach for a release: "
                                   f"{where_to_move(b, z)}")
        print(f"3/5 carry to the bucket ({b.get('source')}): centre ({b['xy'][0]:.0f},{b['xy'][1]:.0f}), "
              f"release at ({xy[0]:.0f},{xy[1]:.0f}) z {z:.0f}")
        z = await carry_over(rig, b, xy, z, lowest_release_z(b, height, grip))
        seen = await look(rig, k)
        if seen is None:
            # Over the mouth the bucket may fill the view: from higher up it is whole
            print("  bucket: not in the view over its mouth; looking from higher up")
            q = await rig.pose()
            if ppa.reachable(q.x, q.y, LOOK_Z):
                await rig.rise(LOOK_Z)
                seen = await look(rig, k)
            if seen is None:
                seen = await locate(rig, k, poses, hint=b["xy"])
            if seen is None:
                raise ppa.SquareNotFound("bucket: not seen again before the release; still holding, nothing let go")
            d = math.dist(seen["xy"], b["xy"])
            looks.append(round(d, 1))
            if d > CONFIRM_MM:
                print(f"  bucket moved {d:.0f} mm: following it")
                b = seen
                continue
            q = await rig.pose()
            if math.hypot(q.x - xy[0], q.y - xy[1]) <= 1.0:
                await rig.descend(z, settle=False)  # back down where it looked from: no sideways leg
            else:
                await ppa.travel(rig, xy[0], xy[1], theta=0.0, z=z, settle=False)
            break
        d = math.dist(seen["xy"], b["xy"])
        looks.append(round(d, 1))
        if d > CONFIRM_MM:
            print(f"  bucket {d:.0f} mm from where it was thought to be: following it")
            b = seen
            continue
        break
    else:
        raise ppa.SquareNotFound(f"bucket kept moving ({looks} mm between looks); still holding, nothing let go")
    times["carry"] = round(time.monotonic() - t, 1)
    times["at"] = "place"
    # Nothing on the jaws: no release and no delivery reported; the job's retry looks again
    # {VIAM CENTERAL 2026-10-30 "在桶口上方松手前，如果 ... 爪子是空的，就不松手，报 OBJECTLOST，走重试"}
    if getattr(rig, "grip_mode", "external") == "external" and hasattr(rig, "jaw_mm"):
        jaw = await rig.jaw_mm()
        if jaw is not None and jaw < ppa.EMPTY_JAW_MM:
            rig.picked_at = None
            raise ppa.ObjectLost(f"over the bucket the jaws are closed to {jaw:.1f} mm: nothing held, not let go")
    print(f"4/5 release over the bucket (confirmed within {looks[-1]:.0f} mm)")
    await ppa.release_jaws(rig)
    ppa.save_state(False)
    remember(b)  # released over it: the next object's search looks here first
    times["place"] = round(time.monotonic() - t - times["carry"], 1)
    centre = (xy[0] + off[0], xy[1] + off[1])
    return {"xy": xy, "z": z, "off_mm": math.dist(centre, b["xy"]), "looks": looks}
