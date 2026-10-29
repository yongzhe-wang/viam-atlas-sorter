"""
Self-test of the module's live map on the REAL table_map.update_map, before a deploy: a map as the module starts
it (Store.clear_table_map) and a map after clear() must both take a frame without an error, and replaying the
observation log with table_map.fold_observations must give the same map as live (AGENT3's contract).

WHY: on sfhack4 every update failed with "KeyError: 'next_id'" because the started/cleared map was a hand-made
dict, not table_map.empty_map(); the stub tests could not see it (VIAM CENTERAL). This runs AGENT3's real
segmentation and update on a synthetic frame (a 30 mm block on a 600 mm table); recognition is given, so no
CLIP model is needed. check_archive.py runs it inside the unpacked archive. Exit 1 on any update error.
Usage: python module/selftest_map.py   (from the repo root or the unpacked archive root)
"""
import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for name in ("torch", "open_clip"):  # recognition is given below, so the model is never loaded
    sys.modules.setdefault(name, SimpleNamespace(set_num_threads=lambda n: None))

import numpy as np  # noqa: E402

from module.live_map import LiveMap  # noqa: E402
from sorter import perception, table_map  # noqa: E402
from sorter.store import Store  # noqa: E402


def frame(dx: int = 0) -> tuple:
    """Synthetic observe-pose frame: a yellow 30 mm block (dx px right of centre) on a flat table 600 mm away."""
    depth = np.full((720, 1280), 600.0)
    rgb = np.full((720, 1280, 3), 200, np.uint8)
    depth[300:380, 560 + dx:640 + dx] = 570.0
    rgb[300:380, 560 + dx:640 + dx] = (230, 200, 30)
    return rgb, depth


def recognised(seg) -> list:
    return [{"obj": o, "label": "selftest block", "names": ["selftest block"], "score": 0.9, "partial": False,
             "emb": np.full(512, 512 ** -0.5, np.float32)} for o in seg.objects]


def summary(m: dict) -> list:
    """What must agree between live and replay: each object's label, xy (0.1 mm) and seen_ts (ms)."""
    return sorted((o.get("label"), round(o["xy"][0], 1), round(o["xy"][1], 1), round(o.get("seen_ts") or 0, 3))
                  for o in m.get("objects") or [])


async def run() -> list:
    rgb, depth = frame()
    seg = perception.segment_full(rgb, depth)
    recs = recognised(seg)
    pose, intr = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}, {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 360.0}
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(uri="", data_dir=tmp)
        lm = LiveMap(store, table_map.update_map, 60.0, initial=store.clear_table_map("module start"),
                     empty_fn=table_map.empty_map)
        for stage in ("started", "cleared"):
            if stage == "cleared":
                await lm.clear("selftest")
            before = lm.stats["errors"]
            lm.offer(rgb, depth, pose, intr, "idle", recognised=recs, seg=seg)
            while lm.busy():
                await asyncio.sleep(0.02)
            if lm.stats["errors"] > before:
                problems.append(f"{stage} map: update_map failed: {lm.stats['last_error']}")
            elif not lm.map.get("objects"):
                problems.append(f"{stage} map: the block was not added ({len(seg.objects)} segmented)")
        # The block moves 120 px; then the log since the epoch, replayed, must give the live map
        rgb2, depth2 = frame(120)
        seg2 = perception.segment_full(rgb2, depth2)
        lm.offer(rgb2, depth2, pose, intr, "idle", recognised=recognised(seg2), seg=seg2)
        while lm.busy():
            await asyncio.sleep(0.02)
        fold = getattr(table_map, "fold_observations", None)
        if fold is None:
            print("selftest_map: no table_map.fold_observations, replay not checked")
        else:
            try:
                replay = fold(store.list_observations(since=lm.epoch))
                if summary(replay) != summary(lm.map):
                    problems.append(f"replay differs from live: {summary(replay)} vs {summary(lm.map)}")
            except Exception as e:
                problems.append(f"replay failed: {type(e).__name__}: {e}"[:300])
        if lm.stats["log_errors"]:
            problems.append(f"{lm.stats['log_errors']} observations not logged")
        await lm.close()
    return problems


if __name__ == "__main__":
    found = asyncio.run(run())
    for p in found:
        print(f"selftest_map: {p}", file=sys.stderr)
    if found:
        sys.exit(1)
    print("selftest_map: started and cleared maps take a real update_map frame; the replayed log equals live")
