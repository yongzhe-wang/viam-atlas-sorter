"""
Record robot episodes for future VLA (vision-language-action) training: fixed-rate steps of wrist camera,
arm state and commanded action, one folder per command. rlds/sorter_vla/ turns these folders into RLDS.

WHY: the arm code moves in discrete jumps (goto, settle 1 s, look, goto); a policy needs a steady stream where
every step has an image, a state and an action. So the recorder never drives anything: it samples the camera
and the arm on its own timer, and learns the commanded targets by wrapping the three choke points every motion
already goes through (Rig.goto, Rig.open, Rig.grab). The control code is unchanged.

Raw episode folder (format version RAW_VERSION):
  <out>/<session>/<episode_id>/
    meta.json      instruction, kind, status/success/fail_reason, timing, camera intrinsics, robot, settings
    steps.jsonl    one line per step: i, t, t_state, t_image, ee {x,y,z mm, o_x,o_y,o_z, theta deg},
                   joints [deg], gripper_cmd (0 open / 1 close), target (last commanded goto or null),
                   image (rgb/NNNNNN.jpg), depth (depth/NNNNNN.png, 16-bit mm) or null
    rgb/  depth/

Timing: every step stores when its state and its image were read. The wrist camera is known to return frames up
to ~0.5 s old right after a move (README "Lessons"), so keep both clocks; the converter can shift images later.
Read failures (link drop, reconnect) skip a step and are counted in meta.json ("gaps"); they never stop control.

Usage (scripts/collect_vla.py does this around executor.execute):
    rec = VLARecorder(rig, "dataset_raw", hz=10)
    rec.attach()
    await rec.start_episode("put the green cube in the square", kind="grasp")
    doc = await executor.execute(rig, "green cube", store)
    await rec.end_episode(doc)
    rec.detach()
"""
import asyncio
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

RAW_VERSION = 1
READ_TIMEOUT_S = 2.0  # one state/image read; slower than this is a gap, not a stall
DEFAULT_ROBOT = "UFactory xArm6 + xArm gripper G2, wrist RealSense D435 (Viam machine sfhack4)"


def _pose_dict(p) -> dict:
    return {"x": float(p.x), "y": float(p.y), "z": float(p.z), "o_x": float(p.o_x), "o_y": float(p.o_y),
            "o_z": float(p.o_z), "theta": float(p.theta)}


def _default_decode(images):
    """The arm library's own decoder (colour JPEG + Viam DEPTHMAP), imported only when recording for real."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import pick_place_auto as ppa

    return ppa.decode(images)


class VLARecorder:
    def __init__(self, rig, out_dir, hz: float = 10.0, image_wh: tuple = (640, 360), save_depth: bool = True,
                 session: str = None, robot: str = DEFAULT_ROBOT, source: str = "scripted", decode=None):
        self.rig = rig
        self.out = Path(out_dir)
        self.hz = float(hz)
        self.image_wh = tuple(image_wh)
        self.save_depth = save_depth
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.robot = robot
        self.source = source
        self.decode = decode or _default_decode
        # Commanded state, updated by the wrapped choke points
        self.gripper_cmd = 0
        self.target = None  # {"x","y","z" mm, "theta" deg, "t"}: the goto in progress or last sent
        self._orig = {}
        self._task = None
        self._writer = ThreadPoolExecutor(max_workers=1)  # one thread: steps are written in order
        self._ep = None

    # ---- hooks ------------------------------------------------------------------------------------------------

    def attach(self) -> None:
        """Wrap rig.goto / rig.open / rig.grab on this instance so every command is seen. Idempotent."""
        if self._orig:
            return
        rig = self.rig
        self._orig = {"goto": rig.goto, "open": rig.open, "grab": rig.grab}

        async def goto(x, y, z, theta=None, settle=True, **kw):
            th = theta if theta is not None else getattr(rig, "theta", 0.0)
            self.target = {"x": float(x), "y": float(y), "z": float(z), "theta": float(th), "t": time.time()}
            return await self._orig["goto"](x, y, z, theta=theta, settle=settle, **kw)

        async def open_():
            self.gripper_cmd = 0
            return await self._orig["open"]()

        async def grab():
            self.gripper_cmd = 1
            return await self._orig["grab"]()

        rig.goto, rig.open, rig.grab = goto, open_, grab

    def detach(self) -> None:
        for name, fn in self._orig.items():
            setattr(self.rig, name, fn)
        self._orig = {}

    # ---- episodes ---------------------------------------------------------------------------------------------

    async def start_episode(self, instruction: str, kind: str = "grasp", **meta) -> Path:
        """Open a new episode folder and start sampling. Ends any episode still running (as 'interrupted')."""
        if self._ep is not None:
            await self.end_episode({"status": "interrupted", "success": False})
        eid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        path = self.out / self.session / eid
        (path / "rgb").mkdir(parents=True, exist_ok=True)
        if self.save_depth:
            (path / "depth").mkdir(exist_ok=True)
        rig = self.rig
        intr = {k: float(getattr(rig, k)) for k in ("fx", "fy", "cx", "cy") if hasattr(rig, k)}
        self._ep = {"path": path, "steps": 0, "gaps": 0, "lines": open(path / "steps.jsonl", "w"),
                    "meta": {"raw_version": RAW_VERSION, "episode_id": eid, "session": self.session,
                             "language_instruction": instruction, "kind": kind, "source": self.source,
                             "robot": self.robot, "hz": self.hz, "image_wh": list(self.image_wh),
                             "depth": self.save_depth, "camera_intrinsics_full_res": intr,
                             "start_ts": time.time(), **meta}}
        self._task = asyncio.get_event_loop().create_task(self._sample_loop())
        return path

    async def end_episode(self, result: dict = None) -> Path:
        """Stop sampling, flush the writer, write meta.json with the outcome. Returns the episode folder."""
        ep = self._ep
        if ep is None:
            return None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await asyncio.get_event_loop().run_in_executor(self._writer, lambda: None)  # drain queued writes
        ep["lines"].close()
        r = result or {}
        meta = ep["meta"]
        meta.update(end_ts=time.time(), n_steps=ep["steps"], gaps=ep["gaps"],
                    status=r.get("status"), success=bool(r.get("success")), fail_phase=r.get("fail_phase"),
                    fail_reason=r.get("fail_reason"), attempt_id=r.get("_id"), label=r.get("label"),
                    duration_s=round(time.time() - meta["start_ts"], 2))
        (ep["path"] / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        self._ep = None
        return ep["path"]

    def close(self) -> None:
        self._writer.shutdown(wait=True)

    # ---- sampling ---------------------------------------------------------------------------------------------

    async def _read(self):
        """(pose, joints, images, t_state, t_image) read in parallel; raises on any failure or timeout."""
        rig = self.rig

        async def state():
            p, j = await asyncio.gather(rig.arm.get_end_position(), rig.arm.get_joint_positions())
            return p, j, time.time()

        async def image():
            res = await rig.cam.get_images()
            return res, time.time()

        (p, j, ts), (imgs, ti) = await asyncio.wait_for(asyncio.gather(state(), image()), READ_TIMEOUT_S)
        return p, j, imgs, ts, ti

    async def _sample_loop(self) -> None:
        period = 1.0 / self.hz
        next_t = time.monotonic()
        while True:
            next_t += period
            ep = self._ep
            if getattr(self.rig, "status", None) == "reconnecting":
                ep["gaps"] += 1  # the control code owns the link while it reconnects
            else:
                t = time.time()
                try:
                    p, j, imgs, ts, ti = await self._read()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    ep["gaps"] += 1
                else:
                    i = ep["steps"]
                    ep["steps"] += 1
                    step = {"i": i, "t": t, "t_state": ts, "t_image": ti, "ee": _pose_dict(p),
                            "joints": [float(v) for v in getattr(j, "values", j)], "gripper_cmd": self.gripper_cmd,
                            "target": dict(self.target) if self.target else None}
                    self._writer.submit(self._write_step, ep, step, imgs)
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind (slow camera): run at the speed the robot allows

    def _write_step(self, ep: dict, step: dict, imgs) -> None:
        """Writer thread: decode, resize, save images, append the step line. A bad frame is a gap."""
        i = step["i"]
        try:
            rgb, depth = self.decode(imgs)
            Image.fromarray(rgb).resize(self.image_wh, Image.BILINEAR).save(ep["path"] / "rgb" / f"{i:06d}.jpg",
                                                                              quality=90)
            step["image"] = f"rgb/{i:06d}.jpg"
            step["depth"] = None
            if self.save_depth and depth is not None:
                d = np.clip(depth, 0, 65535).astype(np.uint16)
                Image.fromarray(d).resize(self.image_wh, Image.NEAREST).save(ep["path"] / "depth" / f"{i:06d}.png")
                step["depth"] = f"depth/{i:06d}.png"
            ep["lines"].write(json.dumps(step) + "\n")
        except Exception as e:
            ep["gaps"] += 1
            print(f"  vla recorder: step {i} dropped ({type(e).__name__}: {e})")
