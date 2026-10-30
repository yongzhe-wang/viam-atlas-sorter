"""
sorter.drop_bucket.locate with a fake arm: the bucket's last-known spot is looked at first, the circle only if it is gone.

{USER 2026-10-30 "IT IS ALWAYS SEARCHING A ENTIRE CIRCLE, IT SHOULD FIRST GO TO THE REGION PREVIOSU FIND THEN IF NOT FOUND"}
No frames, no arm. Run: python tests/test_drop_bucket.py
"""
import asyncio
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import drop_bucket as D  # noqa: E402

ppa = D.ppa
POSES = [(*ppa.HOME_XY, D.LOOK_Z)] + [(x, y, 580.0) for x in (170.0, 400.0) for y in (-180.0, 180.0)]


class Rig:
    """Where the flange is; a look sees the bucket when its centre is within 250 mm of the flange (a z 580 view)."""

    def __init__(self, at, bucket_xy):
        self.at, self.bucket, self.moves = list(at), bucket_xy, []

    async def pose(self):
        return SimpleNamespace(x=self.at[0], y=self.at[1], z=self.at[2], theta=0.0)

    async def frame(self):
        return "rgb", "depth"

    async def rise(self, z):
        self.at[2] = z
        self.moves.append(tuple(self.at))


async def _travel(rig, x, y, theta=0.0, z=0.0, **_):
    rig.at = [x, y, z]
    rig.moves.append((x, y, z))


def _from_frame(rgb, depth, pose, k):
    rig = _from_frame.rig
    if rig.bucket is None or pose["z"] < D.LOOK_Z or math.dist((pose["x"], pose["y"]), rig.bucket) > 250.0:
        return None
    return {"xy": list(rig.bucket), "radius_mm": 98.0, "partial": False, "confidence": 1.0}


def run(rig, hint):
    _from_frame.rig = rig
    ppa.travel, D.from_frame = _travel, _from_frame
    return asyncio.run(D.locate(rig, {}, POSES, hint=hint))


def test_last_known_spot_first():
    rig = Rig((450.0, -150.0, 400.0), (315.0, 300.0))  # just picked on the far side; the bucket where it was
    b = run(rig, (315.0, 300.0))
    assert b is not None and b["xy"] == [315.0, 300.0], b
    assert len(rig.moves) == 1 and math.dist(rig.moves[0][:2], (315.0, 300.0)) < 60.0, rig.moves  # one move, no circle


def test_moved_bucket_still_found_by_the_circle():
    rig = Rig((450.0, -150.0, 400.0), (200.0, -250.0))  # it was moved since
    b = run(rig, (315.0, 300.0))
    assert b is not None and b["xy"] == [200.0, -250.0], b
    assert math.dist(rig.moves[0][:2], (315.0, 300.0)) < 60.0, rig.moves  # the old spot first, then the circle
    assert all(math.dist(m[:2], rig.moves[0][:2]) > D.HINT_SKIP_MM for m in rig.moves[1:]), rig.moves


def test_no_hint_unchanged():
    rig = Rig((450.0, -150.0, 400.0), (315.0, 300.0))
    b = run(rig, None)
    # as before: the circle of sweep poses, nearest first (4 moves here, what the user saw); the hint saves 3 of them
    assert b is not None and rig.moves[0] == (400.0, -180.0, 580.0) and len(rig.moves) == 4, rig.moves


def test_last_known_newest_of_map_and_memory():
    D._LAST.clear()
    assert D.last_known({}) is None
    now = time.time()
    D.remember({"xy": [100.0, 200.0], "seen_ts": now - 100})
    assert D.last_known({"bucket": {"xy": [300.0, 300.0], "seen_ts": now - 10}}) == (300.0, 300.0)
    assert D.last_known({"bucket": {"xy": [300.0, 300.0], "seen_ts": now - 500}}) == (100.0, 200.0)
    assert D.last_known(None) == (100.0, 200.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
