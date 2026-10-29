"""
Self-test of the module's live map on the REAL table_map.update_map, before a deploy: a map as the module starts
it (Store.clear_table_map) and a map after clear() must both take a frame without an error.

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


def frame() -> tuple:
    """Synthetic observe-pose frame: a yellow 30 mm block in the middle of a flat table 600 mm below the camera."""
    depth = np.full((720, 1280), 600.0)
    rgb = np.full((720, 1280, 3), 200, np.uint8)
    depth[300:380, 560:640] = 570.0
    rgb[300:380, 560:640] = (230, 200, 30)
    return rgb, depth


async def run() -> list:
    rgb, depth = frame()
    seg = perception.segment_full(rgb, depth)
    recs = [{"obj": o, "label": "selftest block", "names": ["selftest block"], "score": 0.9, "partial": False,
             "emb": np.zeros(512, np.float32)} for o in seg.objects]
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
        await lm.close()
    return problems


if __name__ == "__main__":
    found = asyncio.run(run())
    for p in found:
        print(f"selftest_map: {p}", file=sys.stderr)
    if found:
        sys.exit(1)
    print("selftest_map: started and cleared maps take a real update_map frame")
