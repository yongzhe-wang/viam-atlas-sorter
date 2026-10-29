"""
Repeated pick-and-place trials with self-reset, no person needed between runs.

WHY: one success says nothing about reliability. N runs give a success rate, per-phase times and the
failure modes, and every place also yields jaw-calibration samples (pick_place_auto.observe_release).

Each trial: pick the block wherever it is -> place it in the blue square -> go home and check with the
camera that the block is inside the hole. Between trials the block is picked out of the square and put
back at a random free spot inside RESET_X x RESET_Y, so the next trial starts somewhere new.

Logs one `attempts`-style document per trial (docs/EXEC_PLAN.md) to sorter.store.Store when that module
exists, else to data/trials.jsonl. Usage: python scripts/run_trials.py [N]   (default 10)
Run it with a hand near the e-stop.
"""
import asyncio
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import pick_place_auto as ppa

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = DATA / "trials.jsonl"
# Random reset area, flange x/y in mm: beside the tape square on the -y side, where the block started.
# A yellow cylinder now sits near (346, -138), so the area is shifted toward the base; place_at also
# refuses any spot with an object within ~90 mm.
RESET_X = (230.0, 330.0)
RESET_Y = (-200.0, -60.0)
# Top of the block is ~347 mm from the camera at ALIGN_Z; the jaw axis meets it near this pixel, which is
# where a block right under the arm appears. Seeds the Tracker.
SEED_DEPTH = 347.0
# Flange x/y where the square servo converged (run 2, 2026-10-29); from here the whole hole is in view
SQUARE_XY = (387.0, 56.0)


def in_square(rgb: np.ndarray, tracker=None):
    """
    True/False: is the block's centroid inside the tape hole? None if the hole or block is not visible.

    Unlike find_square, yellow is NOT treated as wall here, so a block inside the hole stays part of it.
    """
    blue = ndimage.binary_closing(ppa.blue_mask(rgb), iterations=3)
    lab, n = ndimage.label(~blue)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    best = None
    for i in range(1, n + 1):
        if i in border:
            continue
        ys, xs = np.nonzero(lab == i)
        if len(xs) > 2000 and (best is None or len(xs) > len(best[0])):
            best = (xs, ys)
    blk = tracker(rgb) if tracker else ppa.find_block(rgb)
    if best is None or blk is None:
        return None
    xs, ys = best
    return bool(xs.min() <= blk[0] <= xs.max() and ys.min() <= blk[1] <= ys.max())


def open_store():
    """sorter.store.Store if AGENT 2's module has landed, else None (JSONL fallback)."""
    sys.path.insert(0, str(ROOT))
    try:
        from sorter.store import Store
        return Store()
    except Exception as e:
        print(f"store: using {LOG} ({type(e).__name__}: {e})")
        return None


def log(store, doc: dict) -> None:
    if store is not None:
        try:
            store.log_attempt(doc)
            return
        except Exception as e:
            print(f"store.log_attempt failed, writing JSONL instead: {e}")
    DATA.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(doc) + "\n")


def seed_tracker(rig) -> "ppa.Tracker":
    """Tracker for the yellow object right under the arm (where every trial and reset starts)."""
    return ppa.Tracker(ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, SEED_DEPTH))


async def snapshot(rig, path: Path) -> np.ndarray:
    """Colour frame from the current pose, also saved as JPEG for the attempt record."""
    rgb, _ = await rig.frame()
    Image.fromarray(rgb).save(path, quality=85)
    return rgb


async def trial(rig, store, run_id: str, i: int) -> dict:
    """
    One measured pick-into-square. Never raises: failures are recorded with the phase they happened in.

    After a failure the arm is lifted and, if it still holds the block, the block is released in place so
    the reset step can find it.
    """
    DATA.mkdir(exist_ok=True)
    doc = {"ts": time.time(), "run_id": run_id, "mode": "trial", "n": i, "target_text": "yellow block",
           "label": "yellow block", "success": False, "fail_phase": None, "fail_reason": None}
    times = {}
    t0 = time.monotonic()
    try:
        # Starts wherever the previous reset left the arm: directly above the block
        doc["img_before"] = str(DATA / f"trial_{run_id}_{i:02d}_before.jpg")
        await snapshot(rig, Path(doc["img_before"]))
        tracker = seed_tracker(rig)
        height, grip = await ppa.pick(rig, times, tracker)
        doc["height_mm"], doc["grasp"] = round(height, 1), {"grip_depth": round(grip, 1)}
        await ppa.place_in_square(rig, height, grip, times)
        # Check from right above the square, where the whole hole is in view
        times["at"] = "check"
        doc["img_after"] = str(DATA / f"trial_{run_id}_{i:02d}_after.jpg")
        ok = in_square(await snapshot(rig, Path(doc["img_after"])), seed_tracker(rig))
        doc["success"] = ok is True
        if not doc["success"]:
            doc["fail_phase"], doc["fail_reason"] = "check", "block not seen inside the hole" if ok is False else "block or hole not visible"
    except Exception as e:
        doc["fail_phase"], doc["fail_reason"] = times.get("at", "start"), f"{type(e).__name__}: {e}"
        try:
            await rig.lift()
            if await rig.holding():
                # Leave the block on the table so the next reset can pick it up from there
                await ppa.release_here(rig, doc.get("height_mm", 31.0), (doc.get("grasp") or {}).get("grip_depth", 15.0))
        except Exception as e2:
            doc["fail_reason"] += f" | recovery failed: {e2}"
    times.pop("at", None)
    doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
    log(store, doc)
    return doc


async def reset(rig) -> float:
    """
    Move the block out of the square to a random free spot. Returns seconds taken; raises if impossible.

    Called with the arm above the square (where the trial ended), so the block is in view.
    """
    t0 = time.monotonic()
    height, grip = await ppa.pick(rig, None, seed_tracker(rig))
    for _ in range(6):
        x, y = random.uniform(*RESET_X), random.uniform(*RESET_Y)
        print(f"reset: trying ({x:.0f},{y:.0f})")
        if await ppa.place_at(rig, x, y, height, grip):
            return round(time.monotonic() - t0, 1)
    raise RuntimeError("no free reset spot found; block still held")


async def main(n: int) -> None:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    store = open_store()
    rig = ppa.Rig()
    await rig.connect()
    results = []
    try:
        await ppa.travel(rig, *SQUARE_XY, theta=0.0)
        # Start from outside the square so trial 1 is a real pick-into-square
        if in_square((await rig.frame())[0], seed_tracker(rig)):
            print("block starts in the square: resetting first")
            await reset(rig)
        for i in range(1, n + 1):
            print(f"=== trial {i}/{n} ===")
            doc = await trial(rig, store, run_id, i)
            results.append(doc)
            print(f"trial {i}: success={doc['success']} {doc['duration_s']}s phases={doc['phase_times']} "
                  f"fail={doc['fail_phase']} {doc['fail_reason'] or ''}")
            if i < n:
                try:
                    print(f"reset took {await reset(rig)}s")
                except Exception as e:
                    print(f"reset failed ({type(e).__name__}: {e}); stopping")
                    await rig.lift()
                    break
        await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
    finally:
        await rig.close()
        ok = [d for d in results if d["success"]]
        if results:
            print(f"SUMMARY run_id={run_id}: {len(ok)}/{len(results)} succeeded; mean time per run "
                  f"{np.mean([d['duration_s'] for d in results]):.0f}s (successful only: "
                  f"{np.mean([d['duration_s'] for d in ok]) if ok else float('nan'):.0f}s)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
