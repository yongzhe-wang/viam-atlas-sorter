"""
SORTER_SMOOTH (default off): air moves as one smooth joint path, and no fixed wait for the camera after a move.

WHY: the user: "WHY IS THE OPTIMIZATION FOR THE SMOOTH TRAJ NOT DONE", "YOU CAN CONTROL THE TIMING TO MAKE EVERYTHING
SMOOTH, SO THE CONNECTION BETWEEN VIEW AND ACTION IS SMOOTH" {VIAM CENTERAL 2026-10-30 "14:20 前 PUSH，放在 FLAG
SORTER_SMOOTH 后面，默认关"}. Hooks: pick_place_auto Rig.goto (move, arrival, settle), Rig.frame (fresh picture), travel.

1. An air move (both ends at z >= AIR_Z) to a pose the arm has stood at before goes as one move_through_joint_positions
   path: minimum-jerk samples, quintic timing T = max(1.875·|Δq|/v, sqrt(5.77·|Δq|/a)), v 150 deg/s, a 220 deg/s²
   (the ufactory driver takes MoveOptions in degrees; traj_test.py on sfhack4: 15 points, 0.63-0.72 s). No IK is written
   here: a pose's joints are read once the arm stands there (every air move ends with that read), so the observe pose and
   the bucket's mouth, visited by every job, go smoothly from the second visit on; a new pose moves as before. Descents
   and grasps never come here (z < AIR_Z). NOT KNOWN: how far a joint-space path strays from the straight line between
   two tool-down poses; both ends are high and the joints may change at most MAX_DQ_DEG.
2. Settling: instead of sleeping SETTLE_S after a move, a picture is used as soon as the camera says it was captured after
   the move ended (ResponseMetadata.captured_at); older ones are dropped and asked again, never more than SETTLE_S in all.
   No timestamp, or one off this machine's clock: the rest of SETTLE_S is slept, as before.
   NOT KNOWN: whether the RealSense module stamps the capture or the answer; frames_dropped staying 0 on the arm would
   say it stamps the answer, and then this is no better than no wait (the stale frames the lag test saw come back).
3. STATS: seconds the arm stood waiting for a picture (both modes, so they compare), dropped frames, smooth moves.
[CONFIDENCE: UNVERIFIED 60% — offline with a fake arm and camera only]
"""
import math
import os
import time

AIR_Z = 280.0  # only moves with both ends at or above this height (travel, carry over the bucket, back to observe)
V_DEG_S, A_DEG_S2 = 150.0, 220.0  # VIAM CENTERAL / AGENT1's spec
MAX_DQ_DEG = 120.0  # a larger joint change goes the planner's way
SAME_MM, SAME_DEG = 2.0, 0.5  # a pose this close to one stood at before has its joints
SAMPLE_S = 0.05  # one path point per 50 ms of motion
POINTS = (5, 40)
CLOCK_SKEW_S = 0.5  # a capture time further ahead of this clock is not believed

STATS = {"image_wait_s": 0.0, "frames_dropped": 0, "smooth_moves": 0, "settles": 0}
LAST_LAG = []  # capture time - move end, s, of the last accepted fresh frames (what the camera's stamp says)


def enabled() -> bool:
    return os.environ.get("SORTER_SMOOTH", "0") == "1"


def since(before: dict) -> dict:
    """This job's share of STATS: what the module adds to the job's summary."""
    return {"image_wait_s": round(STATS["image_wait_s"] - before["image_wait_s"], 2),
            "frames_dropped": STATS["frames_dropped"] - before["frames_dropped"],
            "smooth_moves": STATS["smooth_moves"] - before["smooth_moves"], "settles": STATS["settles"] - before["settles"]}


def quintic_time(dq_deg: float, v: float = V_DEG_S, a: float = A_DEG_S2) -> float:
    """Duration of a rest-to-rest quintic over the largest joint change: within both the speed and the acceleration."""
    return max(1.875 * dq_deg / v, math.sqrt(5.77 * dq_deg / a))


def minjerk(q0: list, q1: list, n: int) -> list:
    """n points from q0 (excluded) to q1 (included) on the minimum-jerk profile 10s³ - 15s⁴ + 6s⁵."""
    out = []
    for i in range(1, n + 1):
        s = i / n
        f = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        out.append([a + (b - a) * f for a, b in zip(q0, q1)])
    return out


class JointMemory:
    """Joints the arm had at air poses it stood at: (x, y, z, theta) -> joint angles, deg."""

    def __init__(self, size: int = 50):
        self.size, self.poses = size, []

    def _near(self, a: tuple, b: tuple) -> bool:
        return math.dist(a[:3], b[:3]) <= SAME_MM and abs(a[3] - b[3]) <= SAME_DEG

    def remember(self, pose: tuple, joints: list) -> None:
        self.poses = [e for e in self.poses if not self._near(e[0], pose)][-(self.size - 1):] + [(pose, list(joints))]

    def lookup(self, pose: tuple):
        return next((j for p, j in reversed(self.poses) if self._near(p, pose)), None)


MEMORY = JointMemory()


async def joint_move(rig, at, x: float, y: float, z: float) -> bool:
    """
    Rig.goto's move as one smooth joint path, if this is an air move to a pose stood at before; False: move as before.
    at: the pose goto just read. E-stop and a dropped link act as on any move (Rig.call "move").
    """
    if not enabled() or z < AIR_Z or at.z < AIR_Z:
        return False
    target = MEMORY.lookup((x, y, z, rig.theta))
    if target is None:
        return False
    from viam.proto.component.arm import JointPositions, MoveOptions

    q0 = list((await rig.call("get joints", lambda: rig.arm.get_joint_positions())).values)
    dq = max(abs(b - a) for a, b in zip(q0, target))
    if dq > MAX_DQ_DEG:
        return False
    if dq < 0.05:
        return True  # already there
    n = min(max(round(quintic_time(dq) / SAMPLE_S), POINTS[0]), POINTS[1])
    path = [JointPositions(values=q) for q in minjerk(q0, target, n)]
    opts = MoveOptions(max_vel_degs_per_sec=V_DEG_S, max_acc_degs_per_sec2=A_DEG_S2)
    await rig.call("move", lambda: rig.arm.move_through_joint_positions(path, options=opts))
    STATS["smooth_moves"] += 1
    return True


async def arrived(rig, x: float, y: float, z: float) -> None:
    """After an air move: remember the joints of this pose, so the next move here can be a joint path."""
    if enabled() and z >= AIR_Z:
        jp = await rig.call("get joints", lambda: rig.arm.get_joint_positions())
        MEMORY.remember((x, y, z, rig.theta), list(jp.values))


async def settle(rig, settle_s: float) -> None:
    """After a move a picture follows: off, sleep settle_s as before; on, note when the move ended (fresh_images waits)."""
    import asyncio

    STATS["settles"] += 1
    if enabled():
        rig.moved_at = time.time()
        return
    await asyncio.sleep(settle_s)
    STATS["image_wait_s"] += settle_s


def captured_at(res):
    """The camera's capture time of a get_images() answer, epoch s, or None (no metadata, not set, or off this clock)."""
    meta = res[1] if isinstance(res, tuple) and len(res) > 1 else None
    ts = getattr(meta, "captured_at", None)
    if ts is None or not getattr(ts, "seconds", 0):
        return None
    t = ts.seconds + ts.nanos / 1e9
    return t if t <= time.time() + CLOCK_SKEW_S else None


async def fresh_images(rig, fetch, settle_s: float):
    """
    get_images() (fetch) for Rig.frame: a picture captured after the last move ended, waiting at most settle_s since then.
    Off, or no move since the last picture: one fetch, as before.
    """
    import asyncio

    moved = getattr(rig, "moved_at", None)
    if not enabled() or moved is None:
        return await fetch()
    rig.moved_at = None
    t0 = time.monotonic()
    while True:
        res = await fetch()
        cap = captured_at(res)
        if cap is None:  # the camera does not say: the old rule, the rest of settle_s since the move ended
            rest = settle_s - (time.time() - moved)
            if rest > 0:
                await asyncio.sleep(rest)
                res = await fetch()
            break
        if cap > moved:
            LAST_LAG[:] = (LAST_LAG + [round(cap - moved, 3)])[-20:]
            break
        if time.time() - moved >= settle_s:
            break  # as long as before at most
        STATS["frames_dropped"] += 1
    STATS["image_wait_s"] += time.monotonic() - t0
    return res
