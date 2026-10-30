"""
SORTER_SMOOTH (sorter/smooth.py) on the real Rig.goto / Rig.frame / travel with a fake arm and camera, no robot:
off changes nothing; on, an air move to a pose stood at before is one quintic joint path, descents never are; a picture is
taken as soon as one captured after the move arrives, at most SETTLE_S, the old wait without a timestamp; the wait is counted.
{VIAM CENTERAL 2026-10-30 "放在 FLAG SORTER_SMOOTH 后面，默认关"}
"""
import asyncio
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pick_place_auto as ppa  # noqa: E402

from sorter import smooth  # noqa: E402

buf = io.BytesIO()
Image.fromarray(np.zeros((72, 128, 3), np.uint8)).save(buf, "JPEG")
JPEG = buf.getvalue()


class FakeArm:
    """Joints are a function of the pose (any fixed map will do: the memory only replays what it read)."""

    def __init__(self):
        self.p = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}
        self.moves, self.paths = [], []

    def joints(self):
        p = self.p
        return [p["x"] / 10, p["y"] / 10, p["z"] / 10, p["theta"], 0.0, 0.0]

    async def get_end_position(self):
        return SimpleNamespace(**self.p)

    async def move_to_position(self, pose):
        self.moves.append((round(pose.x), round(pose.y), round(pose.z)))
        self.p = {"x": pose.x, "y": pose.y, "z": pose.z, "theta": pose.theta}

    async def get_joint_positions(self):
        return SimpleNamespace(values=self.joints())

    async def move_through_joint_positions(self, path, options=None):
        self.paths.append((len(path), options.max_vel_degs_per_sec, options.max_acc_degs_per_sec2, list(path[-1].values)))
        j = path[-1].values
        self.p = {"x": j[0] * 10, "y": j[1] * 10, "z": j[2] * 10, "theta": j[3]}


class FakeCam:
    """Each picture is `lag` s behind now (what the camera stamps); stamp=False: no captured_at."""

    def __init__(self):
        self.lag, self.stamp, self.calls = 0.0, True, 0

    async def get_images(self, **kw):
        self.calls += 1
        await asyncio.sleep(0.03)
        t = time.time() - self.lag
        meta = SimpleNamespace(captured_at=SimpleNamespace(seconds=int(t), nanos=int((t % 1) * 1e9)) if self.stamp else
                               SimpleNamespace(seconds=0, nanos=0))
        return [SimpleNamespace(mime_type="image/jpeg", data=JPEG)], meta


def rig():
    r = ppa.Rig()
    r.arm, r.cam = FakeArm(), FakeCam()
    r.floor_z = 180.0
    return r


async def timed(coro):
    t0 = time.monotonic()
    await coro
    return time.monotonic() - t0


async def main():
    os.environ.pop("SORTER_SMOOTH", None)
    r = rig()
    w0 = dict(smooth.STATS)
    dt = await timed(r.goto(300.0, 0.0, 400.0))
    await r.frame()
    assert dt >= ppa.SETTLE_S and not r.arm.paths and r.cam.calls == 1 and smooth.MEMORY.poses == []
    assert smooth.since(w0)["image_wait_s"] == ppa.SETTLE_S and smooth.since(w0)["settles"] == 1
    print(f"1. off: a move then SETTLE_S {ppa.SETTLE_S} s asleep, counted as waiting; nothing remembered, no joint path")

    os.environ["SORTER_SMOOTH"] = "1"
    r = rig()
    await r.goto(250.0, 0.0, 400.0, settle=False)  # the observe pose, stood at once
    await ppa.travel(r, 400.0, 150.0, settle=False)  # first visit: the planner's move, remembered
    assert r.arm.moves[-1] == (400, 150, 400) and not r.arm.paths and len(smooth.MEMORY.poses) >= 2
    await ppa.travel(r, 250.0, 0.0, settle=False)  # back to observe: stood there before -> one joint path
    n, v, a, last = r.arm.paths[-1]
    assert (v, a) == (smooth.V_DEG_S, smooth.A_DEG_S2) and last == [25.0, 0.0, 40.0, 0.0, 0.0, 0.0] and 5 <= n <= 40
    assert r.arm.moves[-1] == (400, 150, 400) and r.at["x"] == 250.0
    print(f"2. on: first visit as before, the way back to a pose stood at is one joint path ({n} points, v {v}, a {a})")

    paths = len(r.arm.paths)
    await r.goto(250.0, 0.0, 250.0, settle=False)  # a descent: never a joint path
    await r.goto(250.0, 0.0, 400.0, settle=False)  # from below AIR_Z: never either
    assert len(r.arm.paths) == paths and r.arm.moves[-2:] == [(250, 0, 250), (250, 0, 400)]
    print("3. below z 280 (descend, rise from low) the planner's move, as before")

    q = smooth.minjerk([0.0], [90.0], 10)
    assert abs(q[-1][0] - 90.0) < 1e-9 and q[0][0] < 1.0 and all(b[0] >= a[0] for a, b in zip(q, q[1:]))
    assert abs(smooth.quintic_time(90.0) - max(1.875 * 90 / smooth.V_DEG_S, (5.77 * 90 / smooth.A_DEG_S2) ** 0.5)) < 1e-9
    print(f"4. quintic: 90 deg in {smooth.quintic_time(90.0):.2f} s, minimum-jerk samples monotonic")

    # The picture after a move: the camera's stamp decides, never longer than SETTLE_S, no stamp -> the old wait
    w0, r.cam.calls = dict(smooth.STATS), 0
    await r.goto(300.0, 0.0, 400.0)
    dt = await timed(r.frame())
    assert dt < 0.2 and r.cam.calls == 1 and smooth.since(w0)["frames_dropped"] == 0, (dt, r.cam.calls)
    r.cam.lag = 0.25  # the camera hands out pictures 0.25 s old: the first ones after the move are from before it
    await r.goto(320.0, 0.0, 400.0)
    dt = await timed(r.frame())
    assert 0.2 < dt < ppa.SETTLE_S and smooth.since(w0)["frames_dropped"] >= 2, (dt, smooth.STATS)
    r.cam.lag = 5.0  # a camera always behind: SETTLE_S at most, as before
    await r.goto(330.0, 0.0, 400.0)
    dt = await timed(r.frame())
    assert ppa.SETTLE_S <= dt < ppa.SETTLE_S + 0.1, dt
    r.cam.lag, r.cam.stamp = 0.0, False  # no stamp: the rest of SETTLE_S, then a new picture
    await r.goto(340.0, 0.0, 400.0)
    dt = await timed(r.frame())
    assert ppa.SETTLE_S <= dt < ppa.SETTLE_S + 0.15, dt
    calls = r.cam.calls
    await r.frame()  # no move since: one picture at once
    assert r.cam.calls == calls + 1
    print(f"5. on: picture as soon as one is captured after the move ({smooth.since(w0)['frames_dropped']} stale dropped), "
          f"SETTLE_S at most; no stamp -> the old wait; waits counted: {smooth.since(w0)['image_wait_s']} s")

    ppa.ESTOP["on"] = True
    try:
        await ppa.travel(r, 250.0, 0.0)
        raise AssertionError("moved under an E-stop")
    except ppa.EStop:
        pass
    finally:
        ppa.ESTOP["on"] = False
    print("6. E-stop: the joint path is refused like any move")
    os.environ.pop("SORTER_SMOOTH", None)


asyncio.run(main())
print("test_smooth OK")
