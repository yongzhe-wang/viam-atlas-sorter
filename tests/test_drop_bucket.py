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

import numpy as np

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


def test_partial_view_stops_the_search():
    # {USER 2026-10-30 "IT ALREADY FIND IT RIGHT NOW BTU STILL TAKES THE ROUDN TRIPE"}
    rig = Rig((300.0, 200.0, 580.0), (315.0, 300.0))
    _from_frame.rig = rig
    ppa.travel = _travel
    D.from_frame = lambda *a: {"xy": [315.0, 300.0], "radius_mm": 98.0, "partial": True, "confidence": 0.05}
    b = asyncio.run(D.locate(rig, {}, POSES, hint=(315.0, 300.0)))
    assert b is not None and b["partial"] and rig.moves == [], rig.moves


def test_seen_within_a_minute_is_carried_to_at_once():
    D._LAST.clear()
    D.from_frame = lambda *a: None
    now = 1000.0
    assert D.fresh_bucket({}, {"bucket": {"xy": [1.0, 2.0], "seen_ts": now - 30}}, {}, now)["xy"] == [1.0, 2.0]
    assert D.fresh_bucket({}, {"bucket": {"xy": [1.0, 2.0], "seen_ts": now - 90}}, {}, now) is None
    D.remember({"xy": [5.0, 6.0], "seen_ts": now - 10, "radius_mm": 98.0})
    got = D.fresh_bucket({}, {"bucket": {"xy": [1.0, 2.0], "seen_ts": now - 30}}, {}, now)
    assert got["xy"] == [5.0, 6.0] and got["radius_mm"] == 98.0, got  # the newest
    D._LAST.clear()


class MouthRig(Rig):
    """Frames with a set share of orange pixels; the jaws let go by release_jaws."""

    def __init__(self, at, share):
        super().__init__(at, None)
        self.share = share

    async def frame(self):
        rgb = np.zeros((100, 100, 3), np.uint8)
        rgb[:] = (190, 150, 110)
        rgb[:int(100 * self.share)] = (240, 110, 30)
        return rgb, np.zeros((100, 100))


def _place(rig, seen_from_look):
    released = []

    async def release(_):
        released.append(True)

    ppa.travel, ppa.release_jaws, D.from_frame = _travel, release, lambda *a: seen_from_look
    b = {"xy": [315.0, 300.0], "radius_mm": 98.0, "rim_height_mm": 175.0, "source": "map"}
    times = {}
    done = asyncio.run(D.place_in_bucket(rig, 30.0, 10.0, times, b, {}, POSES))
    return done, released


def test_orange_over_the_mouth_needs_no_trip_up():
    # {LOG "BUCKET: NOT IN THE VIEW OVER ITS MOUTH; LOOKING FROM HIGHER UP"}: 6 carries of 7
    rig = MouthRig((450.0, 100.0, 400.0), 0.5)
    done, released = _place(rig, None)
    assert released and done["looks"] == [0.0], done
    assert all(m[2] <= 400.0 + 1 for m in rig.moves), rig.moves  # never rose to LOOK_Z


def test_no_orange_over_the_mouth_is_never_let_go():
    # Moved away: the table below, no orange anywhere: not released, the job puts it back
    rig = MouthRig((450.0, 100.0, 400.0), 0.0)
    try:
        _place(rig, None)
    except ppa.SquareNotFound as e:
        assert "not seen again" in str(e), e
    else:
        raise AssertionError("released with no bucket in view")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
