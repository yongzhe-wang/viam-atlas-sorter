"""
Arm-driven survey of the table: frames for AGENT3's object scan, free drop spots, and where things are.

WHY: the user asked for the arm to scan the objects instead of a person holding them up; the same frames
also tell us where the table is free (the table is crowded, so a fixed reset rectangle no longer works)
and where the yellow block is for the trials. No gripper commands; objects are never touched.

Poses (AGENT3's label_scan.py spec): theta 0, z=400 only, an overlapping grid with ~90 mm steps so every
object is fully in view in >= 3 frames; label_scan maps pixels to table x/y assuming theta 0. Each pose:
settle (Rig.goto waits SETTLE_S), if a person is in view wait up to 10 s, save data/scan/<n>.jpg +
<n>_depth.npy (uint16 mm, aligned) + <n>.json ({"pose": {x, y, z, theta}}).
Then: data/scan/free_spots.json (flange x/y where an object can be put down) and data/scan/objects.json.
Ends parked at HOME (250, 0, 400). Usage: python scripts/survey_scan.py
"""
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import pick_place_auto as ppa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import perception  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "scan"
GRID_X = (220.0, 310.0, 400.0, 490.0)
GRID_Y = (-330.0, -240.0, -150.0, -60.0, 30.0, 120.0, 210.0, 300.0)
MERGE_MM = 60.0  # objects closer than this across frames are the same object
FREE_STEP_PX = 40  # candidate spacing for free spots in each frame
FREE_CLEAR_PX = 150.0  # ~60 mm at the table depth seen from z=400: object/paper clearance
REACH = (150.0, 600.0)  # flange distance from the base the arm has used safely
VETO_MM = 15.0  # a "not free" sample this close to a free one (another frame, same place) vetoes it


def pixel_to_world(rig, pose, u: float, v: float, depth: float) -> tuple:
    """
    Flange x/y that would put the jaws over colour pixel (u, v) on a plane `depth` mm from the camera.

    Valid at wrist theta 0: image +u is world -y and image +v is world -x (measured shift test), at
    fx/depth px per mm, around the jaw-axis pixel (pick_place_auto.axis_pixel).
    """
    au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, depth)
    k = rig.fx / depth
    return pose.x - (v - av) / k, pose.y - (u - au) / k


def free_mask(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Bool image: True where a FREE_CLEAR_PX square around the pixel holds no object, no unreadable depth,
    only table colour, and no tape/paper. Same criteria as pick_place_auto.spot_free, computed for every
    pixel at once (box windows on a 4x-downsampled grid) so a whole frame costs one pass.
    """
    # Same rules as ppa.spot_free: height above the fitted plane > SPOT_MIN_H_MM (thin things like a
    # screwdriver shaft are 8-9 mm), any unreadable depth, any non-table colour
    a_, b_, c_ = perception.fit_table(depth)
    h, w = depth.shape
    plane = a_ * np.arange(w)[None, :] + b_ * np.arange(h)[:, None] + c_
    up = (depth > 0) & (plane - depth > ppa.SPOT_MIN_H_MM)
    blind = depth == 0
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    paper = (hsv[..., 2] > 200) & (hsv[..., 1] < 40)
    odd = perception.non_table_colour(rgb, (depth > 0) & (np.abs(plane - depth) < 3))
    bad = up | blind | odd | paper | ppa.blue_mask(rgb)
    bad[ppa.FINGER_ROW:] = True  # fingers: never trust that band
    small = bad[::4, ::4].astype(float)
    win = int(2 * FREE_CLEAR_PX / 4) | 1
    # fraction of "bad" pixels in the window; 0.3% tolerance for speckle (a thin shaft is ~1-2%)
    frac = ndimage.uniform_filter(small, size=win, mode="constant", cval=1.0)
    ok = np.repeat(np.repeat(frac < 0.003, 4, axis=0), 4, axis=1)
    return ok[: rgb.shape[0], : rgb.shape[1]]


def person(seg) -> bool:
    """
    A person in view: skin-coloured blobs/holes. Static tall things (the ~110 mm transparent cups read
    "blob 112 mm tall, skin 0%") are not people and must not stall the survey (it waited 30 s per pose).
    """
    return any("skin 0%" not in i["reason"] for i in seg.intruders)


def frame_free_samples(rig, pose, rgb: np.ndarray, depth: np.ndarray, table: float) -> list:
    """
    (x, y, free?) samples on a pixel grid, only where the whole clearance window lies inside the frame.

    WHY: windows reaching past the border were counted as occupied (outside = "bad"), and every such edge
    sample vetoed free samples from neighbouring frames: the first survey found 0 free spots.
    """
    ok = free_mask(rgb, depth)
    m = int(FREE_CLEAR_PX) + 4
    out = []
    for v in range(m, ppa.FINGER_ROW - m, FREE_STEP_PX):
        for u in range(m, rgb.shape[1] - m, FREE_STEP_PX):
            wx, wy = pixel_to_world(rig, pose, u, v, table)
            out.append((wx, wy, bool(ok[v, u])))
    return out


def merge_free(samples: list) -> list:
    """
    Spots free in some frame, not vetoed by another frame at the same place (VETO_MM), in reach, >= 50 mm apart.

    The 60 mm clearance is already inside each sample; a 40 mm veto on top of it demanded ~100 mm and left
    1 spot on the crowded table. The veto only has to catch frames that disagree about the same place.
    """
    bad = np.array([(x, y) for x, y, f in samples if not f]) if samples else np.zeros((0, 2))
    spots = []
    for x, y, f in samples:
        if not f or not ppa.allowed(x, y) or not REACH[0] <= math.hypot(x, y) <= REACH[1]:
            continue
        if len(bad) and np.min(np.hypot(bad[:, 0] - x, bad[:, 1] - y)) < VETO_MM:
            continue
        if all(math.hypot(x - s[0], y - s[1]) > 50.0 for s in spots):
            spots.append((round(x), round(y)))
    return spots


def recompute_free() -> list:
    """Free spots from the saved frames in data/scan (no arm needed); rewrites free_spots.json."""
    from types import SimpleNamespace

    samples = []
    for meta_file in sorted(OUT.glob("[0-9][0-9][0-9].json")):
        meta = json.loads(meta_file.read_text())
        rgb = np.asarray(Image.open(meta_file.with_suffix(".jpg")).convert("RGB"))
        depth = np.load(str(meta_file.with_suffix("")) + "_depth.npy").astype(float)
        rig = SimpleNamespace(**meta["intrinsics"])
        table, _ = ppa.table_and_near(depth)
        samples += frame_free_samples(rig, SimpleNamespace(**meta["pose"]), rgb, depth, table)
    spots = merge_free(samples)
    (OUT / "free_spots.json").write_text(json.dumps(spots))
    print(f"free spots: {len(spots)} from {len(samples)} samples ({sum(f for *_, f in samples)} free)")
    return spots


async def capture(rig, n: int, kind: str, extra: dict = None) -> tuple:
    """If a person is in view wait up to 10 s, then save frame n; returns (rgb, depth, pose, seg)."""
    t0 = time.monotonic()
    while True:
        rgb, depth = await rig.frame()
        seg = perception.segment_full(rgb, depth)
        if not person(seg) or time.monotonic() - t0 > 10:
            break
        print(f"  person in view ({seg.intruders[0]['reason']}), waiting")
        await asyncio.sleep(1.0)
    p = await rig.pose()
    Image.fromarray(rgb).save(OUT / f"{n:03d}.jpg", quality=92)
    np.save(OUT / f"{n:03d}_depth.npy", np.clip(depth, 0, 65535).astype(np.uint16))
    meta = {"n": n, "kind": kind, "ts": time.time(), "pose": {"x": p.x, "y": p.y, "z": p.z, "theta": p.theta},
            "intrinsics": {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy},
            "objects_seen": len(seg.objects), "person_in_view": person(seg), **(extra or {})}
    (OUT / f"{n:03d}.json").write_text(json.dumps(meta, indent=1))
    return rgb, depth, p, seg


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Continue numbering: earlier frames may already be labelled by AGENT3 (label_scan) and must not change
    existing = [int(f.stem) for f in OUT.glob("[0-9][0-9][0-9].json")]
    first = max(existing) + 1 if existing else 0
    survey_id = time.strftime("%Y%m%d-%H%M%S")
    rig = ppa.Rig()
    await rig.start()
    t_start = time.monotonic()
    n, objects, free_samples = first, [], []
    try:
        await ppa.apply_workspace_limits(rig)
        grid = [(x, y) for x in GRID_X for y in GRID_Y if ppa.allowed(x, y) and REACH[0] <= math.hypot(x, y) <= REACH[1]]
        print(f"survey: {len(grid)} grid points, theta 0, z {ppa.ALIGN_Z:.0f}")
        for x, y in grid:
            await ppa.travel(rig, x, y, theta=0.0)
            rgb, depth, p, seg = await capture(rig, n, "grid", {"survey_id": survey_id})
            n += 1
            table, _ = ppa.table_and_near(depth)
            # Objects -> world positions (edge-cut ones too: their centre is approximate, merged below)
            for o in list(seg.objects) + [o for o, _ in seg.excluded]:
                wx, wy = pixel_to_world(rig, p, *o.centroid_px, o.top_depth_mm)
                objects.append({"x": wx, "y": wy, "colour": o.colour, "height_mm": round(o.height_mm, 1),
                                "edge": bool(o.touches_edge), "frame": n - 1})
            # Free-spot samples on a pixel grid
            free_samples += frame_free_samples(rig, p, rgb, depth, table)
        # Merge objects seen from several grid points (prefer fully visible sightings)
        merged = []
        for o in sorted(objects, key=lambda o: o["edge"]):
            if all(math.hypot(o["x"] - m["x"], o["y"] - m["y"]) > MERGE_MM for m in merged):
                merged.append(o)
        for m in merged:
            m["reach_mm"] = round(math.hypot(m["x"], m["y"]))
            m["reachable"] = ppa.allowed(m["x"], m["y"]) and REACH[0] <= m["reach_mm"] <= REACH[1]
        print(f"survey: {len(merged)} distinct objects")
        for m in merged:
            print(f"  {m['colour']:7s} at ({m['x']:.0f},{m['y']:.0f}) h={m['height_mm']} mm, {m['reach_mm']} mm from base"
                  f"{'' if m['reachable'] else '  <- NOT reachable'}")
        spots = merge_free(free_samples)
        (OUT / "free_spots.json").write_text(json.dumps(spots))
        (OUT / "objects.json").write_text(json.dumps(merged, indent=1))
        print(f"survey: {len(spots)} free drop spots -> {OUT / 'free_spots.json'}")
        yellow = [m for m in merged if m["colour"] == "yellow"]
        for m in yellow:
            print(f"  yellow object at ({m['x']:.0f},{m['y']:.0f}) height {m['height_mm']} mm")
    finally:
        try:
            await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
        finally:
            await rig.close()
        print(f"survey: frames {first:03d}-{n - 1:03d} in {OUT} ({time.monotonic() - t_start:.0f} s); parked at {ppa.HOME_XY}")


async def around(points: list, offset: float = 40.0) -> None:
    """
    Extra library frames: 4 views per point, flange at (x +- offset, y +- offset), theta 0, z ALIGN_Z.

    WHY: an object that the grid only saw in corner views (the green cube near the base) needs centred
    views for the object library. Indices continue after the newest frame in data/scan.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [int(f.stem) for f in OUT.glob("[0-9][0-9][0-9].json")]
    n = max(existing) + 1 if existing else 0
    rig = ppa.Rig()
    await rig.start()
    try:
        await ppa.apply_workspace_limits(rig)
        for cx, cy in points:
            for dx, dy in ((-offset, -offset), (-offset, offset), (offset, -offset), (offset, offset)):
                x, y = cx + dx, cy + dy
                if not ppa.allowed(x, y) or math.hypot(x, y) < 120.0:
                    print(f"  skip ({x:.0f},{y:.0f}): keep-out or too close to the base")
                    continue
                await ppa.travel(rig, x, y, theta=0.0)
                await capture(rig, n, "library", {"around": [cx, cy]})
                print(f"  frame {n:03d} at ({x:.0f},{y:.0f})")
                n += 1
    finally:
        try:
            await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
        finally:
            await rig.close()


if __name__ == "__main__":
    # --free-only: recompute free_spots.json from the saved frames without moving the arm
    if "--free-only" in sys.argv:
        recompute_free()
    elif "--around" in sys.argv:
        # --around X,Y [X,Y ...]: 4 library views around each flange point
        pts = [tuple(map(float, a.split(","))) for a in sys.argv[sys.argv.index("--around") + 1:]]
        asyncio.run(around(pts))
    else:
        asyncio.run(main())
