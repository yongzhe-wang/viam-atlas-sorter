"""
Repeated pick-and-place trials with self-reset, no person needed between runs.

WHY: one success says nothing about reliability. N runs give a success rate, per-phase times and the
failure modes, and every place also yields jaw-calibration samples (pick_place_auto.observe_release).

Each trial: pick the block wherever it is -> place it in the blue square -> go home and check with the
camera that the block is inside the hole. Before the first trial and after every trial, sorter.executor.clear_square() empties the square
(block -> random free spot), so trials loop unattended and each starts somewhere new.

Logs one `attempts`-style document per trial (docs/EXEC_PLAN.md) to sorter.store.Store when that module
exists, else to data/trials.jsonl. Usage: python scripts/run_trials.py [N]   (default 10)
Run it with a hand near the e-stop.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import pick_place_auto as ppa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import executor  # noqa: E402  square clearing (reset) shared with the dashboard's Clear button

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = DATA / "trials.jsonl"
# Top of the block is ~347 mm from the camera at ALIGN_Z; the jaw axis meets it near this pixel, which is
# where a block right under the arm appears. Seeds the Tracker.
SEED_DEPTH = 347.0


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


def survey_block():
    """
    Flange x/y of the yellow block from data/scan/objects.json (scripts/survey_scan.py), or None.

    The yellow cylinder lying on the table is ~45 mm tall, the block ~32 mm, so the shortest yellow wins.
    """
    try:
        objs = json.loads((DATA / "scan" / "objects.json").read_text())
    except (OSError, ValueError):
        return None
    yellow = [o for o in objs if o.get("colour") == "yellow" and 15 <= o.get("height_mm", 0) <= 40]
    if not yellow:
        return None
    best = min(yellow, key=lambda o: abs(o["height_mm"] - 32.0))
    print(f"survey: yellow block at ({best['x']:.0f},{best['y']:.0f}), {best['height_mm']} mm tall")
    return best["x"], best["y"]


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


async def trial(rig, store, run_id: str, i: int, square_xy=None) -> dict:
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
        await ppa.place_in_square(rig, height, grip, times, square_xy=square_xy)
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


async def main(n: int) -> None:
    # RUN_ID is kept across supervisor restarts so the count continues instead of starting over
    run_id = os.environ.get("RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    store = open_store()
    done = [d for d in (store.list_attempts(limit=0) if store else []) if d.get("run_id") == run_id and d.get("mode") == "trial"]
    if done:
        print(f"resuming run {run_id}: {len(done)} trial(s) already logged")
    rig = ppa.Rig()
    await rig.start()
    results = []
    try:
        # Obstacles (walls) shrink the search grid and reset area; ranges are printed for the log
        await executor.apply_limits(rig)
        # A previous process may have died holding the block: put it in the square first
        await executor.resume(rig, store)
        # Where the block is if the square starts empty (env BLOCK_XY="x,y", flange mm); else from the resets
        block_xy = tuple(map(float, os.environ["BLOCK_XY"].split(","))) if os.environ.get("BLOCK_XY") else survey_block()
        # Empty the square first, so trial 1 is a real pick-into-square
        cleared = await executor.clear_square(rig, store)
        print(f"reset: {cleared['status']} removed={cleared['removed']} wipes={cleared['wipes']} {cleared['duration_s']}s")
        for i in range(len(done) + 1, n + 1):
            if not cleared["success"]:
                print(f"square not empty ({cleared['fail_reason']}); parked, not starting trial {i}")
                break
            print(f"=== trial {i}/{n} ===")
            if cleared.get("dropped_at"):
                block_xy = cleared["dropped_at"][-1]
            # The reset leaves the arm over the (now empty) square; the block is where it was last dropped
            if block_xy is not None:
                await ppa.travel(rig, *block_xy, theta=0.0)
            doc = await trial(rig, store, run_id, i, cleared.get("square_xy"))
            results.append(doc)
            print(f"trial {i}: success={doc['success']} {doc['duration_s']}s phases={doc['phase_times']} "
                  f"fail={doc['fail_phase']} {doc['fail_reason'] or ''}")
            # Clear the square after every trial (also after a failed one) so the next starts the same way
            cleared = await executor.clear_square(rig, store)
            print(f"reset: {cleared['status']} removed={cleared['removed']} wipes={cleared['wipes']} {cleared['duration_s']}s")
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
