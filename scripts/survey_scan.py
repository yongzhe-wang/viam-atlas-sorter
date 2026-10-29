"""
Arm-driven survey of the table: frames for AGENT3's object scan, free drop spots, and where things are.

WHY: the user asked for the arm to scan the objects instead of a person holding them up; the same frames
also tell us where the table is free (the table is crowded, so a fixed reset rectangle no longer works)
and where the yellow block is for the trials. No gripper commands; objects are never touched.

Poses: grid GRID_X x GRID_Y at z=400 and z=470 (tool down, theta 0), then over every object seen at
z=400, wrist theta +-35 deg. Each pose: settle (Rig.goto waits SETTLE_S), wait for intruders to leave
(perception, max 30 s), save data/scan/<n>.jpg + <n>_depth.npy (uint16 mm, aligned) + <n>.json.
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
GRID_X = (250.0, 380.0, 500.0)
GRID_Y = (-300.0, -100.0, 100.0, 300.0)
HIGH_Z = 470.0
OBJ_THETAS = (35.0, -35.0)
MERGE_MM = 60.0  # objects closer than this across frames are the same object
FREE_STEP_PX = 40  # candidate spacing for free spots in each frame
FREE_CLEAR_PX = 150.0  # ~60 mm at the table depth seen from z=400: object/paper clearance
REACH = (150.0, 600.0)  # flange distance from the base the arm has used safely


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
    table, _ = ppa.table_and_near(depth)
    up = (depth > 0) & (depth < table - 10)
    blind = depth == 0
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    paper = (hsv[..., 2] > 200) & (hsv[..., 1] < 40)
    odd = perception.non_table_colour(rgb, (depth > 0) & (np.abs(depth - table) < 3))
    bad = up | blind | odd | paper | ppa.blue_mask(rgb)
    bad[ppa.FINGER_ROW:] = True  # fingers: never trust that band
    small = bad[::4, ::4].astype(float)
    win = int(2 * FREE_CLEAR_PX / 4) | 1
    # fraction of "bad" pixels in the window; 1% tolerance for speckle
    frac = ndimage.uniform_filter(small, size=win, mode="constant", cval=1.0)
    ok = np.repeat(np.repeat(frac < 0.01, 4, axis=0), 4, axis=1)
    return ok[: rgb.shape[0], : rgb.shape[1]]


async def capture(rig, n: int, kind: str, extra: dict = None) -> tuple:
    """Wait for intruders to leave (max 30 s), then save frame n; returns (rgb, depth, pose, seg)."""
    t0 = time.monotonic()
    while True:
        rgb, depth = await rig.frame()
        seg = perception.segment_full(rgb, depth)
        if not seg.intruders or time.monotonic() - t0 > 30:
            break
        print(f"  intruder in view ({seg.intruders[0]['reason']}), waiting")
        await asyncio.sleep(1.0)
    p = await rig.pose()
    Image.fromarray(rgb).save(OUT / f"{n:03d}.jpg", quality=92)
    np.save(OUT / f"{n:03d}_depth.npy", np.clip(depth, 0, 65535).astype(np.uint16))
    meta = {"n": n, "kind": kind, "ts": time.time(), "pose": {"x": p.x, "y": p.y, "z": p.z, "theta": p.theta},
            "intrinsics": {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy},
            "objects_seen": len(seg.objects), "intruder": bool(seg.intruders), **(extra or {})}
    (OUT / f"{n:03d}.json").write_text(json.dumps(meta, indent=1))
    return rgb, depth, p, seg


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rig = ppa.Rig()
    await rig.start()
    t_start = time.monotonic()
    n, objects, free_samples = 0, [], []
    try:
        await ppa.apply_workspace_limits(rig)
        grid = [(x, y) for x in GRID_X for y in GRID_Y if ppa.allowed(x, y) and REACH[0] <= math.hypot(x, y) <= REACH[1]]
        print(f"survey: {len(grid)} grid points x 2 heights")
        for x, y in grid:
            await ppa.travel(rig, x, y, theta=0.0)
            rgb, depth, p, seg = await capture(rig, n, "grid")
            n += 1
            table, _ = ppa.table_and_near(depth)
            # Objects -> world positions (edge-cut ones too: their centre is approximate, merged below)
            for o in list(seg.objects) + [o for o, _ in seg.excluded]:
                wx, wy = pixel_to_world(rig, p, *o.centroid_px, o.top_depth_mm)
                objects.append({"x": wx, "y": wy, "colour": o.colour, "height_mm": round(o.height_mm, 1),
                                "edge": bool(o.touches_edge), "frame": n - 1})
            # Free-spot samples on a pixel grid
            ok = free_mask(rgb, depth)
            for v in range(FREE_STEP_PX, ppa.FINGER_ROW, FREE_STEP_PX):
                for u in range(FREE_STEP_PX, rgb.shape[1] - FREE_STEP_PX, FREE_STEP_PX):
                    wx, wy = pixel_to_world(rig, p, u, v, table)
                    free_samples.append((wx, wy, bool(ok[v, u])))
            await rig.goto(x, y, HIGH_Z)
            await capture(rig, n, "grid_high")
            n += 1
        # Merge objects seen from several grid points (prefer fully visible sightings)
        merged = []
        for o in sorted(objects, key=lambda o: o["edge"]):
            if all(math.hypot(o["x"] - m["x"], o["y"] - m["y"]) > MERGE_MM for m in merged):
                merged.append(o)
        merged = [m for m in merged if ppa.allowed(m["x"], m["y"]) and REACH[0] <= math.hypot(m["x"], m["y"]) <= REACH[1]]
        print(f"survey: {len(merged)} distinct objects; wrist views over each")
        for i, m in enumerate(merged):
            for th in OBJ_THETAS:
                await ppa.travel(rig, m["x"], m["y"], theta=th)
                await capture(rig, n, "object", {"object": i, "object_xy": [m["x"], m["y"]], "colour": m["colour"]})
                n += 1
        # Free spots: free in some frame and not contradicted by any frame within 40 mm
        bad = np.array([(x, y) for x, y, f in free_samples if not f]) if free_samples else np.zeros((0, 2))
        spots = []
        for x, y, f in free_samples:
            if not f or not ppa.allowed(x, y) or not REACH[0] <= math.hypot(x, y) <= REACH[1]:
                continue
            if len(bad) and np.min(np.hypot(bad[:, 0] - x, bad[:, 1] - y)) < 40.0:
                continue
            if all(math.hypot(x - s[0], y - s[1]) > 50.0 for s in spots):
                spots.append((round(x), round(y)))
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
        print(f"survey: {n} frames in {OUT} ({time.monotonic() - t_start:.0f} s); parked at {ppa.HOME_XY}")


if __name__ == "__main__":
    asyncio.run(main())
