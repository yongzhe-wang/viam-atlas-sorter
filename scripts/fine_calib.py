"""
Fine calibration of where the jaws are, from a block the arm itself puts down: grasp it at a wrist angle, put it down at
the next spot at that angle, look from above. The jaws centre the block across their closing direction, so its centre then
lies on the real jaw line: the gap to the predicted jaw point, across the jaws, is the model's error there.

WHY: grasps landed a few mm off and the error grew with the wrist turned (the red lollipop pushed ~40 mm at -30 deg).
{USER 2026-10-30 "WE NEED SLIGHTLY A FEW MORE FINEGRAIN 校准"}
{VIAM CENTERAL 2026-10-30 "θ 取 0、±45、90，位置取工作区的 3 个点 ... 每次放下的位置就是下次抓的位置"}
[CONFIDENCE: INFERRED 70% — the fit is checked on synthetic samples; the run itself has not been on the arm]

What it fits, and why only that: every look is at theta 0, where the camera-to-jaw offset (perception.JAW_IN_CAM_MM) shifts
the measured block and the predicted jaw point alike, so the samples cannot tell it apart from the tool axis. They do tell
  axis_shift_mm  where the jaw axis really is in the camera, as a shift of pick_place_auto.axis_pixel: across the jaws
                 (image u) only. Along the fingers it turns with the wrist as the closing direction does, so no sample sees
                 it; it stays as it is (the fingers are 20 mm wide: a few mm there does not lose a grasp)
  tool_axis_mm   the wrist's rotation axis from the jaw axis, perception.TOOL_AXIS_MM (samples at +-45 and 90 deg)
Written to sorter/fine_calib.json (shipped with the module; loaded by pick_place_auto and perception at start; delete it to
go back). A fit is written only if it lowers the residual.

Run with the module idle (one arm driver), a blue or yellow block on the table in the observe view, the three spots clear:
  VIAM_API_KEY=... VIAM_API_KEY_ID=... VIAM_MACHINE_ADDRESS=... .venv/bin/python scripts/fine_calib.py run --colour blue
  .venv/bin/python scripts/fine_calib.py fit      # refit from data/fine_calib_samples.jsonl, no arm
Stops at the first thing it cannot do (block not found, a finger blocked, out of reach, an emergency stop).
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pick_place_auto as ppa  # noqa: E402

from sorter import perception  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "fine_calib_samples.jsonl"
OUT = ppa.FINE_CALIB_FILE
# Three spots well inside the observe and look poses' views, >= 225 mm from the base: (480, -60) was put down and not seen
# again there on the arm {VIAM CENTERAL 2026-10-30 "第三个点改到 OBSERVE 视野和 LOOK POSE 都稳妥的位置，比如 (380,−40)"}
SPOTS = [(330.0, -120.0), (420.0, 60.0), (380.0, -40.0)]
HIGH_Z = 580.0  # a block not found from ALIGN_Z is looked for from here, over the same spot
FIT_WINDOW_S = 7200.0  # samples this recent are fitted together: a rerun goes on from the last one
THETAS = [0.0, 45.0, -45.0, 90.0]
GRIP_MM = 15.0
NEAR_MM = 225.0


def intr(rig) -> dict:
    return {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy}


def find_block(rgb, depth, pose: dict, k: dict, colour: str, near_xy):
    """(table xy of its top centre, Obj) of the block of `colour` nearest near_xy, or None."""
    best = None
    for o in perception.segment_full(rgb, depth).objects:
        mm = o.top_depth_mm / k["fx"]
        if o.touches_edge or not 20.0 <= o.height_mm <= 80.0 or o.width_px * mm > 60.0 or o.length_px * mm > 70.0:
            continue
        if o.colour != colour and (getattr(o, "colours", None) or {}).get(colour, 0.0) < 0.3:
            continue
        u, v = o.top_centroid_px or o.centroid_px
        xy = perception.world_xy(pose, k, u, v, o.top_depth_mm)
        d = math.dist(xy, near_xy)
        if best is None or d < best[0]:
            best = (d, (float(xy[0]), float(xy[1])), o)
    return None if best is None or best[0] > 120.0 else best[1:]


def jaw_offset(theta: float, top: float, k: dict) -> tuple:
    """Jaw point minus flange xy at this wrist angle, by the current model (as executor.approach uses it)."""
    au, av = ppa.axis_pixel(k["fx"], k["fy"], k["cx"], k["cy"], top)
    return perception.world_xy({"x": 0.0, "y": 0.0, "z": ppa.ALIGN_Z, "theta": theta}, k, au, av, top)


async def look_at(rig, xy) -> tuple:
    """Over table point xy at ALIGN_Z, theta 0, a frame: (rgb, depth, pose)."""
    fx, fy = perception.look_flange_xy(tuple(xy))
    await ppa.travel(rig, fx, fy, theta=0.0)
    q = await rig.pose()
    rgb, depth = await rig.frame()
    return rgb, depth, {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta}


async def find_near(rig, colour: str, xy):
    """
    The block of `colour` near table point xy: looked at from ALIGN_Z, then from HIGH_Z over the same spot (more table in
    view). (table xy, Obj, frame pose) or None. {VIAM CENTERAL 2026-10-30 "找不到方块时，先升到 Z580 原地看一眼再判断"}
    """
    k = intr(rig)
    rgb, depth, pose = await look_at(rig, xy)
    hit = find_block(rgb, depth, pose, k, colour, xy)
    if hit is None and ppa.reachable(pose["x"], pose["y"], HIGH_Z):
        print(f"  no {colour} block near ({xy[0]:.0f},{xy[1]:.0f}) from z {pose['z']:.0f}; looking from z {HIGH_Z:.0f}")
        await rig.rise(HIGH_Z)
        q = await rig.pose()
        rgb, depth = await rig.frame()
        pose = {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta}
        hit = find_block(rgb, depth, pose, k, colour, xy)
    return None if hit is None else (hit[0], hit[1], pose)


async def grasp_z(rig, height: float) -> tuple:
    """(flange z for the fingertips GRIP_MM below the top, table depth) here; the floors as in ppa.pick."""
    z0, table, _ = await rig.measure()
    grip = min(GRIP_MM, height / 2.0)
    hard = ppa.MIN_GRASP_Z_THIN if height < ppa.THIN_MM else ppa.MIN_GRASP_Z
    return max(z0 - (table - height + grip - ppa.TIP_BELOW_CAM), rig.floor_z, hard), table, z0


async def sample(rig, colour: str, at_xy, to_xy, theta: float) -> dict:
    """Grasp the block at at_xy with the wrist at theta, put it down at to_xy at the same angle, measure it there."""
    k = intr(rig)
    for xy in (at_xy, to_xy):
        if math.hypot(*xy) < NEAR_MM:
            raise ppa.TooClose(ppa.too_close_msg(*xy))
    hit = await find_near(rig, colour, at_xy)
    if hit is None:
        raise RuntimeError(f"no {colour} block near ({at_xy[0]:.0f},{at_xy[1]:.0f})")
    c0, o, pose = hit
    if pose["z"] > ppa.ALIGN_Z + 1.0:
        await look_at(rig, c0)  # seen from high up: back down over it, where the grasp is planned from
    top, height = o.top_depth_mm - (pose["z"] - ppa.ALIGN_Z), o.height_mm
    jx, jy = jaw_offset(theta, top, k)
    await ppa.travel(rig, c0[0] - jx, c0[1] - jy, theta=theta)
    z, table, z0 = await grasp_z(rig, height)
    await rig.wait_clear(ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, top))
    why = ppa.finger_obstacle(rig.last_depth, rig.fx, rig.fy, rig.cx, rig.cy, z0 - z + ppa.TIP_BELOW_CAM, table)
    if why is not None:
        raise ppa.UnsafeGrasp(f"calibration grasp at {theta:+.0f} deg: {why}")
    await rig.open()
    await rig.descend(z, settle=False)
    await rig.grab()
    await rig.rise(ppa.ALIGN_Z, settle=False)
    tx, ty = to_xy[0] - jx, to_xy[1] - jy
    await ppa.travel(rig, tx, ty, theta=theta)
    z_put, _, _ = await grasp_z(rig, height)  # the table where it goes down: bottom on the table, no drop
    q = await rig.pose()
    release = {"x": q.x, "y": q.y, "z": z_put, "theta": q.theta}
    await rig.descend(z_put, settle=False)
    await rig.open()
    await rig.rise(ppa.ALIGN_Z)
    hit = await find_near(rig, colour, to_xy)
    if hit is None:
        raise RuntimeError(f"the block put down at ({to_xy[0]:.0f},{to_xy[1]:.0f}) is not seen there, from z 400 or 580")
    c1 = hit[0]
    s = {"ts": time.time(), "theta": theta, "release": release, "top_depth_mm": top, "height_mm": height,
         "before": list(c0), "after": list(c1), "intr": k}
    pred = predict(s, (0.0, 0.0), perception.TOOL_AXIS_MM)
    print(f"  theta {theta:+.0f}: put down at ({c1[0]:.1f},{c1[1]:.1f}), predicted jaw ({pred[0]:.1f},{pred[1]:.1f}), "
          f"across-jaw error {across_error(s, (0.0, 0.0), perception.TOOL_AXIS_MM):+.1f} mm")
    with SAMPLES.open("a") as f:
        f.write(json.dumps(s) + "\n")
    return {"xy": c1}


def predict(s: dict, shift_mm, tool_axis) -> tuple:
    """Jaw point at the sample's release pose, with the axis pixel shifted by shift_mm (camera mm) and this tool axis."""
    k, top = s["intr"], s["top_depth_mm"]
    au, av = ppa.axis_pixel(k["fx"], k["fy"], k["cx"], k["cy"], top, shift_mm=(0.0, 0.0))
    au, av = au + shift_mm[0] * k["fx"] / top, av + shift_mm[1] * k["fy"] / top
    saved = perception.TOOL_AXIS_MM
    perception.TOOL_AXIS_MM = tuple(tool_axis)
    try:
        r = s["release"]
        return perception.world_xy({"x": r["x"], "y": r["y"], "z": ppa.ALIGN_Z, "theta": r["theta"]}, k, au, av, top)
    finally:
        perception.TOOL_AXIS_MM = saved


def across(s: dict) -> np.ndarray:
    """Unit vector of the jaws' closing direction on the table at the sample's release pose."""
    k, top, r = s["intr"], s["top_depth_mm"], s["release"]
    au, av = ppa.axis_pixel(k["fx"], k["fy"], k["cx"], k["cy"], top, shift_mm=(0.0, 0.0))
    pose = {"x": r["x"], "y": r["y"], "z": ppa.ALIGN_Z, "theta": r["theta"]}
    d = np.subtract(perception.world_xy(pose, k, au + 20.0, av, top), perception.world_xy(pose, k, au - 20.0, av, top))
    return d / np.linalg.norm(d)


def across_error(s: dict, shift_mm, tool_axis) -> float:
    """mm the put-down block lies off the predicted jaw line, across the jaws (the only component they set)."""
    return float(np.dot(np.subtract(s["after"], predict(s, shift_mm, tool_axis)), across(s)))


def fit(samples: list) -> dict:
    """
    Least squares over (axis shift across the jaws, tool axis x, y), from the current model; the residuals before and after.
    Robust (soft_l1, 3 mm): a block that bounced or tipped on the put-down does not pull the fit.
    """
    from scipy.optimize import least_squares

    t0 = np.array(perception.TOOL_AXIS_MM, float)
    s0 = np.array(ppa.AXIS_SHIFT_MM, float)

    def params(p):
        return (p[0], s0[1]), p[1:]

    def res(p):
        shift, tool = params(p)
        return [across_error(s, shift, tool) for s in samples]
    p0 = np.array([s0[0], *t0])
    before = np.array(res(p0))
    sol = least_squares(res, p0, loss="soft_l1", f_scale=3.0)
    after = np.array(res(sol.x))
    shift, tool = params(sol.x)
    return {"axis_shift_mm": [round(float(v), 2) for v in shift], "tool_axis_mm": [round(float(v), 2) for v in tool],
            "rms_before_mm": round(float(np.sqrt(np.mean(before ** 2))), 2), "rms_after_mm": round(float(np.sqrt(np.mean(after ** 2))), 2),
            "residuals_before": [round(float(v), 1) for v in before], "residuals_after": [round(float(v), 1) for v in after],
            "thetas": [s["theta"] for s in samples], "n": len(samples), "ts": time.time()}


def fit_and_write(samples: list) -> dict:
    """Fit; write OUT only when it is better and has samples at more than one wrist angle."""
    if len(samples) < 4 or len({round(s["theta"]) for s in samples}) < 2:
        print(f"need >= 4 samples at >= 2 wrist angles, have {len(samples)}")
        return None
    out = fit(samples)
    print(json.dumps({k: v for k, v in out.items() if k != "ts"}, indent=1))
    if out["rms_after_mm"] < out["rms_before_mm"]:
        OUT.write_text(json.dumps(out, indent=1))
        print(f"written {OUT} (rms {out['rms_before_mm']} -> {out['rms_after_mm']} mm); commit it and redeploy the module")
    else:
        print("not written: the fit does not lower the residual")
    return out


def recent_samples() -> list:
    """The samples of this calibration session (FIT_WINDOW_S), oldest first."""
    if not SAMPLES.exists():
        return []
    rows = [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]
    return [r for r in rows if time.time() - float(r.get("ts", 0.0)) <= FIT_WINDOW_S]


async def run(colour: str, n: int) -> None:
    """
    n more samples, then a fit over this session's samples. Starts where the last sample put the block down (a rerun after
    a stop needs no one to move it), else from the observe view. {VIAM CENTERAL 2026-10-30 "从方块当前位置开始"}
    """
    rig = ppa.Rig()
    await rig.start()
    try:
        before = recent_samples()
        at, hit = None, None
        if before:
            last = tuple(before[-1]["after"])
            print(f"resuming: {len(before)} samples so far, the block was last put down at ({last[0]:.0f},{last[1]:.0f})")
            hit = await find_near(rig, colour, last)
        if hit is None:
            await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
            q = await rig.pose()
            rgb, depth = await rig.frame()
            found = find_block(rgb, depth, {"x": q.x, "y": q.y, "z": q.z, "theta": q.theta}, intr(rig), colour, (q.x + 150.0, q.y))
            hit = None if found is None else (*found, None)
        if hit is None:
            raise RuntimeError(f"no {colour} block where it was last put down, nor in the observe view")
        at = hit[0]
        k0 = len(before)
        for i in range(n):
            j = k0 + i
            to = SPOTS[j % len(SPOTS)]
            if math.dist(to, at) < 40.0:
                to = SPOTS[(j + 1) % len(SPOTS)]  # it already lies there: on to the next spot
            theta = THETAS[j % len(THETAS)]
            print(f"sample {j + 1}: theta {theta:+.0f}, ({at[0]:.0f},{at[1]:.0f}) -> ({to[0]:.0f},{to[1]:.0f})")
            at = (await sample(rig, colour, at, to, theta))["xy"]
        await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
    finally:
        fit_and_write(recent_samples())
        await rig.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=["run", "fit"])
    ap.add_argument("--colour", default="blue")
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a.colour, a.n))
    else:
        fit_and_write(recent_samples())


if __name__ == "__main__":
    main()
