"""
{"dashboard": {"since": ts}}: everything the dashboard shows, in one DoCommand it sends at most twice a second.

WHY: the dashboard asked three commands on their own clocks (status every 1.5 s, seen_image every 1 s, table_map every
2 s: ~2.2 calls a second), and its picture was the last idle look: none while a job ran, and on screen 1.8 s old in the
median. The user wants it "all dynamic" and optimized {VIAM CENTERAL 2026-10-30 "DASHBOARD 对 MODULE 的请求每秒 ≤ 2 次，
超时 3 S"} {VIAM CENTERAL 2026-10-30 "相机：持续刷新…任务进行中也不能停"}.

The picture is the newest of: the idle look, every executor frame of a job (Rig.on_map_frame), and while idle a
colour-only frame grabbed for this call (not while a look runs, nor when perception is slow, which means a loaded robot).
Boxes: the idle look's recognition (a grab reuses it while the arm has not moved); an executor frame gets the live map's
last observation when that was taken from the same pose. The JPEG is encoded once per picture, and only sent to a caller
whose `since` is older.
"""
import asyncio
import base64
import io
import time
from typing import Optional

JPEG_WIDTH = 640
JPEG_QUALITY = 70
GRAB_EVERY_S = 0.4  # idle: a new colour frame for a call when the newest picture is older than this
GRAB_TIMEOUT_S = 1.0
ARM_TIMEOUT_S = 0.5
SAME_POSE_MM, SAME_POSE_DEG = 3.0, 1.0  # a frame from within this of another one's pose shows the same view
BOXES_FRESH_S = 5.0  # boxes from another frame of the same view are kept this long


def same_pose(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    return (max(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in ("x", "y", "z")) <= SAME_POSE_MM
            and abs(float(a.get("theta", 0.0)) - float(b.get("theta", 0.0))) <= SAME_POSE_DEG)


def boxes_from_seen(objects: list, intruders: list) -> list:
    """The idle look's summaries (module _perceive) as boxes in full-resolution pixels."""
    return ([{"bbox": o["bbox"], "label": o.get("label"), "names": [o["label"], *(o.get("aliases") or [])] if o.get("label") else [],
              "score": o.get("confidence"), "kind": "object"} for o in objects]
            + [{"bbox": i["bbox"], "kind": "intruder"} for i in intruders])


def boxes_from_observation(obs: dict) -> list:
    """A live-map observation record (table_map.observe_frame) as boxes; edge-cut objects are "partial"."""
    return ([{"bbox": d["bbox"], "label": d.get("label"), "names": d.get("names") or [], "score": d.get("score"),
              "kind": "partial" if d.get("partial") else "object"} for d in obs.get("detections") or []]
            + [{"bbox": list(b), "kind": "intruder"} for b in obs.get("intruders") or []])


class DashboardView:
    """The newest picture of the table, whoever took it, and its encoded form for the dashboard."""

    def __init__(self):
        self.frame = None  # {"ts", "rgb", "pose", "source", "boxes"}: the newest picture
        self._idle = None  # (ts, pose, boxes) of the last idle look: a grab from the same pose shows the same objects
        self._sent = (None, None)  # (frame ts, encoded answer): one encoding per picture, for every caller

    def put(self, ts: float, rgb, pose: dict, source: str, boxes: Optional[list] = None) -> None:
        """A new picture. Called inside the arm's control loop too, so it only keeps references."""
        if rgb is None or (self.frame is not None and ts < self.frame["ts"]):
            return
        if source == "idle" and boxes is not None:
            self._idle = (ts, dict(pose or {}), boxes)
        self.frame = {"ts": float(ts), "rgb": rgb, "pose": dict(pose or {}), "source": source, "boxes": boxes}

    def age(self) -> float:
        return time.time() - self.frame["ts"] if self.frame else 1e9

    def idle_boxes(self, pose: dict) -> Optional[list]:
        """The last idle look's boxes for a grab from `pose`, while they are fresh and the arm has not moved."""
        if self._idle and time.time() - self._idle[0] <= BOXES_FRESH_S and same_pose(self._idle[1], pose):
            return self._idle[2]
        return None

    def encoded(self, since: float, obs: Optional[dict], footprint) -> Optional[dict]:
        """
        The picture for a caller that already has `since`: {"ts"} when nothing is newer, else the JPEG (base64, at most
        JPEG_WIDTH wide), its boxes scaled to it and `view`, the table area it shows (footprint(pose, w, h) or None).
        """
        f = self.frame
        if f is None:
            return None
        if f["ts"] <= since:
            return {"ts": f["ts"]}
        if self._sent[0] == f["ts"]:
            return self._sent[1]
        from PIL import Image

        im = Image.fromarray(f["rgb"])
        h, w = f["rgb"].shape[:2]
        scale = min(1.0, JPEG_WIDTH / w)
        if scale < 1.0:
            im = im.resize((JPEG_WIDTH, round(h * scale)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        boxes = f["boxes"]
        if boxes is None and obs and same_pose(obs.get("pose"), f["pose"]) and abs(float(obs.get("ts", 0.0)) - f["ts"]) <= BOXES_FRESH_S:
            boxes = boxes_from_observation(obs)
        out = {"ts": f["ts"], "jpeg_b64": base64.b64encode(buf.getvalue()).decode(), "w": im.width, "h": im.height,
               "source": f["source"], "pose": f["pose"],
               "boxes": [{**b, "bbox": [round(float(c) * scale, 1) for c in b["bbox"]]} for b in boxes or []],
               "view": footprint(f["pose"], w, h)}
        self._sent = (f["ts"], out)
        return out


async def answer(svc, rt, arg, grab: bool) -> dict:
    """
    The {"dashboard"} answer for the module `svc`: status, live map, the arm's pose now, E-stop and phase, and the
    picture if newer than arg["since"]. rt: the arm/perception stack, None while warming up (nothing is imported then).
    grab: a colour frame may be taken for this call (the service decides: idle, no look running, perception keeping up).
    """
    since = float(arg.get("since") or 0.0) if isinstance(arg, dict) else 0.0
    arm = None
    try:
        p = await asyncio.wait_for(svc.arm.get_end_position(), ARM_TIMEOUT_S)
        arm = {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1), "theta": round(p.theta, 1)}
    except Exception:  # the arm busy answering the executor or the link slow: the dashboard keeps the last pose
        pass
    if grab and rt is not None and svc.view.age() >= GRAB_EVERY_S:
        try:
            res = await asyncio.wait_for(svc.camera.get_images(filter_source_names=["color"]), GRAB_TIMEOUT_S)
            rgb, _ = await asyncio.to_thread(rt.ppa.decode, res[0] if isinstance(res, tuple) else res)
            pose = arm or (svc.view.frame or {}).get("pose") or {}
            svc.view.put(time.time(), rgb, pose, "idle", svc.view.idle_boxes(pose))
        except Exception:  # no new picture this time; the newest one stays
            pass
    m = svc.live.light() if svc.live is not None else {}
    svc._mark_superseded(m)
    busy = svc._busy()
    out = {"status": svc._status(), "arm": arm,
           "map": {"map": m or {}, "now": time.time(),
                   "sweep": dict(svc.job_info.get("sweep") or {}) if busy else {},
                   "search": dict(svc.job_info.get("search") or {}) if busy else {},
                   "live": svc.live.summary() if svc.live is not None else {"live": False}}}
    if rt is not None:
        out["estop"] = bool(rt.ppa.ESTOP["on"])
        if busy and svc.rig is not None and hasattr(rt.executor, "phase_now"):
            out["phase"] = rt.executor.phase_now(svc.rig)
    intr = svc._intr

    def footprint(pose: dict, w: int, h: int) -> Optional[list]:
        """Where the picture's corners meet the table (mm), to draw what the camera covers on the map."""
        if rt is None or intr is None or not pose:
            return None
        depth = rt.ppa.TABLE_DEPTH_MM + float(pose.get("z", rt.ppa.ALIGN_Z)) - rt.ppa.ALIGN_Z
        return [[round(float(c), 1) for c in rt.perception.world_xy(pose, intr, u, v, depth)]
                for u, v in ((0, 0), (w, 0), (w, h), (0, h))]

    obs = ((svc.live.map or {}).get("last_update") or {}).get("observation") if svc.live is not None else None
    frame = await asyncio.to_thread(svc.view.encoded, since, obs, footprint)
    if frame is not None:
        out["frame"] = frame
    return out
