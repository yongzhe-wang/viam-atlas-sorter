"""
Viam generic service `sorter-team:sorter:grasp-by-description`: describe an object, it goes in the blue square.

WHY: other robot builders should get our whole pipeline (perception + CLIP choice + depth-guided pick + reset
+ Atlas logging) by adding one service to their machine config, without our scripts or a laptop. Running
inside viam-server also removes the laptop <-> robot WebRTC link that dropped repeatedly during our trials:
the arm, gripper and camera are reached through viam-server's local connections.

Always ready: CLIP and the object library load in the background at startup ("warm" in status). While idle
with the arm at the observe pose, the service looks at the table about once a second and caches what it
recognises ("seen"). A grasp then decides from that cached frame in milliseconds ("decide_ms") and either
starts moving at once or answers at once that the object is not on the table.

DoCommand:
  {"grasp": "green square"}          decide + start in the background -> {"ok": true, "accepted": true, "job": id,
                                     "decide_ms": ..., "target": {...}} or {"ok": false, "reason": "no ... on the table"}
  {"grasp": "...", "wait": true}     same, but block until done -> the attempt summary
  {"reset": true}                    clear the square (pick-out, wipe fallback, verify empty)
  {"status": true}                   busy / running / last / warm / seen / library / store mode
  {"seen_image": true}               last idle frame as base64 JPEG (<= 640 px) with the seen boxes scaled to it
One job at a time; a second request while busy returns {"accepted": false, "error": "busy"}.

Config attributes (all optional except where the defaults don't match your machine):
  arm, gripper, camera            resource names, default "arm", "gripper", "cam"; declared as dependencies
  square_colour                   colour of the tape square to drop into; only "blue" is supported today
  mongodb_uri                     Atlas connection string; else env MONGODB_URI; else local JSONL files
  consume_dashboard_commands      true -> also run commands queued from the dashboard (replaces sorter/worker.py)
  home_xy [x, y]                  observe pose of the flange in mm (default from scripts/pick_place_auto.py)
  jaw_tip_px [u, v], tip_below_cam_mm   jaw-axis calibration for your camera mount (see module/README.md)

Arm logic is AGENT1's (scripts/pick_place_auto.py Rig, sorter/executor.py) and perception/recognition is
AGENT3's (sorter/perception.py, sorter/clip_embed.py); this file only adapts them.
"""
import asyncio
import base64
import hashlib
import inspect
import io
import os
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import ValueTypes, struct_to_dict

NAMESPACE = "sorter-team"  # must match "module_id" in meta.json; change both to your registry namespace
LOGGER = getLogger(__name__)
SUPPORTED_SQUARES = ("blue",)  # executor/pick_place_auto find the drop square by blue tape
HEARTBEAT_S = 5.0  # dashboard marks a worker offline after 15 s (agreed with AGENT1)
# Object library shipped in the archive by module/build.sh (views.jsonl + img/); read-only, versioned
BUNDLE_DIR = Path(__file__).resolve().parent.parent / "library"
SUMMARY_KEYS = ("_id", "status", "label", "score", "height_mm", "duration_s", "fail_phase", "fail_reason")
WATCH_PERIOD_S = 1.0  # idle perception: one frame about every second
SEEN_FRESH_S = 1.5  # a grasp decides from the cached frame only when it is at most this old (VIAM CENTERAL)
OBSERVE_TOL_MM = 15.0  # the arm counts as "at the observe pose" within this of HOME_XY / ALIGN_Z
SEEN_IMAGE_WIDTH = 640


def seed_library(bundle: Path, data: Path) -> str:
    """
    Copy the bundled object library into the writable data dir if it is missing or the bundle changed.

    WHY: without Atlas the module's Store on the robot would have no views, so "green square" could not tell
    the cube from the green cylinder. The writable dir ($VIAM_MODULE_DATA/data) survives redeploys and holds
    the module's attempts/commands/status; only views.jsonl and its images are replaced, and only when the
    bundle's content hash differs from the one seeded last time. Image paths ("img/<id>.jpg") stay valid.
    """
    src = bundle / "views.jsonl"
    if not src.exists():
        return "no bundled library"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    marker, dst = data / ".library_hash", data / "views.jsonl"
    if dst.exists() and marker.exists() and marker.read_text().strip() == digest:
        return f"library {digest} already seeded"
    (data / "img").mkdir(parents=True, exist_ok=True)
    for img in (bundle / "img").glob("*.jpg"):
        shutil.copy2(img, data / "img" / img.name)
    tmp = data / "views.jsonl.seeding"
    shutil.copyfile(src, tmp)
    tmp.replace(dst)  # atomic: a reader never sees half a library
    marker.write_text(digest)
    return f"seeded library {digest}"


def _runtime():
    """
    The arm/perception stack, imported on first use so viam-server can start the module and validate config
    quickly (warm-up then loads CLIP in the background). One place, so tests can swap it for stubs.
    """
    from sorter import clip_embed, executor, perception, worker

    return SimpleNamespace(ppa=executor.ppa, executor=executor, clean=worker.clean, perception=perception,
                           clip_embed=clip_embed)


def _check_xy(attrs: dict, key: str) -> None:
    v = attrs.get(key)
    if v is not None and not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
        raise ValueError(f"'{key}' must be a list of two numbers, got {v!r}")


def _box(b) -> list:
    return [round(float(x), 1) for x in b]


class GraspByDescription(Generic, EasyResource):
    """Generic service: grasp the object matching a text description and drop it in the blue square."""

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, "sorter"), "grasp-by-description")

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        """Reject bad attributes before anything moves; arm, gripper and camera are required dependencies."""
        attrs = struct_to_dict(config.attributes)
        names = []
        for key, default in (("arm", "arm"), ("gripper", "gripper"), ("camera", "cam")):
            v = attrs.get(key, default)
            if not isinstance(v, str) or not v:
                raise ValueError(f"'{key}' must be the name of your {key} resource")
            names.append(v)
        square = attrs.get("square_colour", "blue")
        if square not in SUPPORTED_SQUARES:
            raise ValueError(f"'square_colour' {square!r} not supported; supported: {', '.join(SUPPORTED_SQUARES)}")
        for key in ("home_xy", "jaw_tip_px"):
            _check_xy(attrs, key)
        t = attrs.get("tip_below_cam_mm")
        if t is not None and not (isinstance(t, (int, float)) and 50 <= t <= 400):
            raise ValueError("'tip_below_cam_mm' must be a number of mm between 50 and 400")
        if not isinstance(attrs.get("consume_dashboard_commands", False), bool):
            raise ValueError("'consume_dashboard_commands' must be true or false")
        uri = attrs.get("mongodb_uri")
        if uri is not None and not (isinstance(uri, str) and uri.startswith(("mongodb://", "mongodb+srv://"))):
            raise ValueError("'mongodb_uri' must be a mongodb:// or mongodb+srv:// connection string")
        return names, []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """
        Build the service from config + dependency handles. viam-server re-creates it on every config change,
        so there is no separate reconfigure step. Starts warm-up, idle perception and heartbeat loops.
        """
        from sorter.store import Store

        self = cls(config.name)
        attrs = struct_to_dict(config.attributes)
        self.attrs = attrs
        self.arm = dependencies[Arm.get_resource_name(attrs.get("arm", "arm"))]
        self.gripper = dependencies[Gripper.get_resource_name(attrs.get("gripper", "gripper"))]
        self.camera = dependencies[Camera.get_resource_name(attrs.get("camera", "cam"))]
        # Writable data under $VIAM_MODULE_DATA (kept across versions); with Atlas the library lives there instead
        uri = attrs.get("mongodb_uri") or os.environ.get("MONGODB_URI") or ""
        module_data = os.environ.get("VIAM_MODULE_DATA")
        data_dir = Path(module_data) / "data" if module_data else None
        if not uri and data_dir is not None:
            LOGGER.info(f"{config.name}: {seed_library(BUNDLE_DIR, data_dir)} into {data_dir}")
        self.store = Store(uri=uri, data_dir=str(data_dir) if data_dir else None)
        self.library = self.store.library_stats()  # fixed for this instance: the library only changes on redeploy
        self.worker_id = f"module:{config.name}"
        self.rig = None
        self.job: Optional[asyncio.Task] = None
        self.job_info: Dict[str, Any] = {}
        self.last: Dict[str, Any] = {}
        self.warm, self.warm_s, self.warm_error = False, None, None
        self.seen: Dict[str, Any] = {}  # summary of the latest idle look, for status / dashboard
        self._seen_raw = None  # {"ts", "objs", "rgb", "depth", "pose"} behind self.seen, for a fast grasp
        self._jpeg_cache = (None, None, 1.0)  # (frame ts, base64 jpeg, scale) for seen_image
        self._no_target_warned = False
        loop = asyncio.get_event_loop()
        self.tasks = [loop.create_task(self._heartbeat_loop()), loop.create_task(self._warmup()),
                      loop.create_task(self._watch_loop())]
        if attrs.get("consume_dashboard_commands", False):
            self.tasks.append(loop.create_task(self._command_loop()))
        LOGGER.info(f"{config.name}: store {'Atlas' if self.store.online else 'local JSONL'}, "
                    f"library {self.library['labels']} labels / {self.library['views']} views, "
                    f"dashboard commands {'on' if attrs.get('consume_dashboard_commands') else 'off'}")
        return self

    # ---- arm adapter -------------------------------------------------------------------------------

    async def _get_rig(self):
        """
        AGENT1's Rig, bound to this module's dependency handles instead of its own RobotClient + env vars.

        Overriding connect() keeps every motion limit, retry and settle wait of Rig unchanged.
        Calibration attributes override the sfhack4 defaults in pick_place_auto / executor.
        """
        if self.rig is not None:
            return self.rig
        rt = _runtime()
        ppa, executor = rt.ppa, rt.executor
        if self.attrs.get("home_xy"):
            ppa.HOME_XY = tuple(self.attrs["home_xy"])
            executor.OBSERVE_XY = [ppa.HOME_XY] + list(ppa.SEARCH_XY)
        if self.attrs.get("jaw_tip_px"):
            ppa.JAW_TIP_PX = tuple(self.attrs["jaw_tip_px"])
        if self.attrs.get("tip_below_cam_mm"):
            ppa.TIP_BELOW_CAM = float(self.attrs["tip_below_cam_mm"])
        handles = (self.arm, self.camera, self.gripper)

        class ModuleRig(ppa.Rig):
            async def connect(self) -> None:
                self.arm, self.cam, self.gripper = handles
                if not hasattr(self, "fx"):
                    intr = (await self.cam.get_properties()).intrinsic_parameters
                    self.fx, self.fy, self.cx, self.cy = intr.focal_x_px, intr.focal_y_px, intr.center_x_px, intr.center_y_px
                    await self.detect_depth_shift()

            async def close(self) -> None:
                return None  # viam-server owns these connections

        rig = ModuleRig()
        await rig.start()
        self.rig = rig
        return rig

    # ---- always ready: warm-up and idle perception ---------------------------------------------------

    def _busy(self) -> bool:
        return self.job is not None and not self.job.done()

    async def _warmup(self) -> None:
        """Load CLIP and the view library right after start, so the first grasp pays for neither."""
        t0 = time.monotonic()
        try:
            rt = await asyncio.to_thread(_runtime)
            await asyncio.to_thread(rt.clip_embed.embed_text, "warm up")  # loads the CLIP weights
            # Parses (and, offline, caches) the library; a unit vector because cosine search rejects zero vectors
            await asyncio.to_thread(self.store.search_views, [1.0] + [0.0] * 511, 1)
            self.warm, self.warm_s = True, round(time.monotonic() - t0, 1)
            LOGGER.info(f"{self.name}: warm in {self.warm_s} s")
        except Exception as e:
            self.warm_error = f"{type(e).__name__}: {e}"[:300]
            LOGGER.error(f"{self.name}: warm-up failed: {self.warm_error}")

    def _perceive(self, rt, rgb, depth) -> tuple:
        """
        Targets in one frame plus what the library recognises each as: (objs, [summaries], [intruders]).

        Same filter as executor.find_target: nothing cut by the image border, nothing already in the square.
        """
        seg = rt.perception.segment_full(rgb, depth)
        hole = rt.executor.hole_mask(rgb)
        objs = [o for o in seg.objects
                if not o.touches_edge and (hole is None or (o.mask & hole).sum() < 0.5 * o.mask.sum())]
        embs = rt.clip_embed.embed_images([o.crop for o in objs])
        objects = []
        for o, emb in zip(objs, embs):
            lab, names, conf = rt.clip_embed.recognise(emb, self.store, colour=o.colour, height_mm=o.height_mm)
            objects.append({"id": int(o.id), "label": lab, "aliases": [n for n in names if n != lab],
                            "confidence": round(float(conf), 3), "colour": o.colour,
                            "height_mm": round(float(o.height_mm), 1), "bbox": _box(o.bbox)})
        intruders = [{"bbox": _box(i["bbox"]), "reason": i.get("reason", "")} for i in seg.intruders]
        return objs, objects, intruders

    async def _pose(self, rt) -> tuple:
        """(pose dict, at_observe) from the arm's current end position."""
        p = await self.arm.get_end_position()
        hx, hy = rt.ppa.HOME_XY
        at = max(abs(p.x - hx), abs(p.y - hy), abs(p.z - rt.ppa.ALIGN_Z)) <= OBSERVE_TOL_MM
        return {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1), "theta": round(p.theta, 1)}, at

    async def _look(self) -> None:
        """
        One idle look: only from the observe pose and only if the arm did not move while we looked (another
        driver may be using it), so a cached frame always matches where the arm is.
        """
        rt = _runtime()
        pose, at = await self._pose(rt)
        if not at:
            self._seen_raw = None
            self.seen = {"ts": time.time(), "at_observe": False, "pose": pose, "objects": [], "intruders": [],
                         "reason": "arm is not at the observe pose"}
            return
        t_cap = time.time()
        res = await self.camera.get_images()
        rgb, depth = rt.ppa.decode(res[0] if isinstance(res, tuple) else res)
        if rgb is None or depth is None:
            raise RuntimeError("camera returned no colour + depth frame")
        t0 = time.monotonic()
        objs, objects, intruders = await asyncio.to_thread(self._perceive, rt, rgb, depth)
        perceive_ms = round((time.monotonic() - t0) * 1000)
        pose2, still = await self._pose(rt)
        if self._busy() or not still or max(abs(pose2[k] - pose[k]) for k in ("x", "y", "z")) > 2.0:
            return  # arm moved meanwhile: this frame says nothing reliable about where things are now
        self._seen_raw = {"ts": t_cap, "objs": objs, "rgb": rgb, "depth": depth, "pose": pose}
        self.seen = {"ts": t_cap, "at_observe": True, "pose": pose, "objects": objects, "intruders": intruders,
                     "perceive_ms": perceive_ms}

    async def _watch_loop(self) -> None:
        """Idle perception, about once a second, once warm and while no job runs."""
        while True:
            await asyncio.sleep(WATCH_PERIOD_S)
            if not self.warm or self._busy():
                continue
            try:
                await self._look()
            except Exception as e:
                self._seen_raw = None
                self.seen = {"ts": time.time(), "at_observe": None, "objects": [], "intruders": [],
                             "error": f"{type(e).__name__}: {e}"[:200]}

    def _seen_image(self) -> Dict[str, ValueTypes]:
        """Latest idle frame as a small JPEG (base64) with the seen boxes scaled to it, for a live dashboard."""
        raw = self._seen_raw
        if raw is None:
            return {"ok": False, "reason": self.seen.get("reason") or self.seen.get("error") or "nothing seen yet"}
        if self._jpeg_cache[0] != raw["ts"]:
            from PIL import Image

            im = Image.fromarray(raw["rgb"])
            scale = min(1.0, SEEN_IMAGE_WIDTH / im.width)
            if scale < 1.0:
                im = im.resize((SEEN_IMAGE_WIDTH, round(im.height * scale)))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=75)
            self._jpeg_cache = (raw["ts"], base64.b64encode(buf.getvalue()).decode(), scale)
        ts, b64, scale = self._jpeg_cache
        seen = self.seen if self.seen.get("ts") == ts else {"objects": [], "intruders": []}
        return {"ok": True, "ts": ts, "jpeg_b64": b64, "scale": scale,
                "objects": [{**o, "bbox": [x * scale for x in o["bbox"]]} for o in seen["objects"]],
                "intruders": [{**i, "bbox": [x * scale for x in i["bbox"]]} for i in seen["intruders"]]}

    # ---- jobs --------------------------------------------------------------------------------------

    async def _run(self, kind: str, text: Optional[str], target: Optional[dict] = None,
                   decide_ms: Optional[int] = None) -> dict:
        """Run one grasp or reset to completion and return its summary; never raises."""
        fast = False
        try:
            rt = _runtime()
            rig = await self._get_rig()
            if kind == "grasp":
                # Fast path needs AGENT1's execute(..., target=...): skip its first observe and use ours
                fast = target is not None and "target" in inspect.signature(rt.executor.execute).parameters
                if target is not None and not fast and not self._no_target_warned:
                    LOGGER.warning("executor.execute has no target= parameter yet: full observe instead of the fast path")
                    self._no_target_warned = True
                kwargs = {"target": target} if fast else {}
                doc = await rt.executor.execute(rig, text, self.store, mode="demo", **kwargs)
            else:
                doc = await rt.executor.clear_square(rig, self.store)
            summary = {k: doc.get(k) for k in SUMMARY_KEYS if doc.get(k) is not None}
        except Exception as e:
            LOGGER.error(f"{kind} failed: {type(e).__name__}: {e}")
            summary = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            try:
                if self.rig is not None:
                    await self.rig.lift()
            except Exception:
                pass
        if kind == "grasp":
            summary.update(fast_path=fast, decide_ms=decide_ms)
        # job id lets a DoCommand caller (the dashboard in module mode) match this result to its request
        self.last = {"kind": kind, "text": text, "job": self.job_info.get("job"), "finished": time.time(), **summary}
        return summary

    async def _start(self, kind: str, text: Optional[str], wait: bool, target: Optional[dict] = None,
                     decide_ms: Optional[int] = None) -> Dict[str, ValueTypes]:
        if self._busy():
            return {"accepted": False, "error": "busy", "running": dict(self.job_info)}
        job_id = uuid.uuid4().hex[:12]
        self.job_info = {"job": job_id, "kind": kind, "text": text or "", "started": time.time(),
                         "fast_path": target is not None}
        self.job = asyncio.get_event_loop().create_task(self._run(kind, text, target, decide_ms))
        if wait:
            return {"accepted": True, "job": job_id, **(await self.job)}
        return {"accepted": True, "job": job_id}

    async def _grasp(self, text: str, wait: bool) -> Dict[str, ValueTypes]:
        """
        Decide from the cached idle frame when it is fresh: start at once, or say at once that it is not there.
        Otherwise (no fresh frame, arm away from the observe pose) fall back to the executor's own observe.
        """
        if self._busy():
            return {"ok": False, "accepted": False, "error": "busy", "running": dict(self.job_info)}
        raw = self._seen_raw
        age = time.time() - raw["ts"] if raw else None
        if raw is None or age > SEEN_FRESH_S:
            why = "no fresh frame from the observe pose" if raw is None else f"cached frame is {age:.1f} s old"
            res = await self._start("grasp", text, wait)
            return {**res, "ok": bool(res.get("accepted")), "decided_from_cache": False, "fast_path": False,
                    "decide_ms": None, "why_slow": why}
        rt = _runtime()
        t0 = time.monotonic()
        pick = await asyncio.to_thread(rt.clip_embed.best, raw["objs"], text, self.store)
        decide_ms = round((time.monotonic() - t0) * 1000)
        obj = pick.get("obj")
        if obj is None or pick.get("ambiguous"):
            status = "not_found" if obj is None else "ambiguous"
            reason = (f"no {text!r} on the table" + (f" ({pick['reason']})" if pick.get("reason") else "")
                      if obj is None else f"two objects match {text!r} about equally; be more specific")
            self.last = {"kind": "grasp", "text": text, "job": None, "finished": time.time(), "status": status,
                         "fail_phase": "find", "fail_reason": reason, "decide_ms": decide_ms, "decided_from_cache": True}
            return {"ok": False, "accepted": False, "status": status, "reason": reason, "decide_ms": decide_ms,
                    "decided_from_cache": True}
        target = {**pick, "rgb": raw["rgb"], "depth": raw["depth"], "pose": raw["pose"], "ts": raw["ts"]}
        # fast_path: the executor will skip its own first observe (needs AGENT1's execute(..., target=...))
        fast = "target" in inspect.signature(rt.executor.execute).parameters
        res = await self._start("grasp", text, wait, target=target, decide_ms=decide_ms)
        if not res.get("accepted"):
            return {**res, "ok": False}  # became busy while deciding
        return {**res, "ok": True, "decided_from_cache": True, "fast_path": fast, "decide_ms": decide_ms,
                "target": {"id": int(obj.id), "label": pick.get("recognised"),
                           "score": round(float(pick.get("score", 0.0)), 3), "bbox": _box(obj.bbox)}}

    def _status(self) -> Dict[str, ValueTypes]:
        seen = dict(self.seen)
        if seen.get("ts"):
            seen["age_s"] = round(time.time() - seen["ts"], 1)
        return {"busy": self._busy(), "running": dict(self.job_info) if self._busy() else {}, "last": dict(self.last),
                "store": "atlas" if self.store.online else "jsonl", "worker_id": self.worker_id,
                "library": dict(self.library), "warm": self.warm, "warm_s": self.warm_s,
                "warm_error": self.warm_error, "seen": seen}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        if command.get("status"):
            return self._status()
        if command.get("seen_image"):
            return self._seen_image()
        if "grasp" in command:
            text = _runtime().clean(str(command.get("grasp") or ""))
            if not text:
                return {"ok": False, "accepted": False, "error": "say what to grasp, e.g. {\"grasp\": \"green square\"}"}
            return await self._grasp(text, bool(command.get("wait")))
        if command.get("reset"):
            return await self._start("reset", None, bool(command.get("wait")))
        return {"error": "unknown command; use {\"grasp\": \"<description>\"}, {\"reset\": true}, {\"status\": true} "
                         "or {\"seen_image\": true}"}

    # ---- background loops --------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Dashboard worker panel: what this service is doing, every HEARTBEAT_S."""
        while True:
            try:
                busy = self._busy()
                phase = (f"{self.job_info.get('kind')}: {self.job_info.get('text') or 'square'}" if busy
                         else "idle" if self.warm else "warming up")
                err = self.last.get("error") if not busy else None
                # consumer: whether this instance takes dashboard commands, so the dashboard can warn when two
                # consumers are live at once (EXEC_PLAN: exactly one)
                self.store.heartbeat(self.worker_id, phase, error=err, last_status=self.last.get("status"),
                                     consumer=bool(self.attrs.get("consume_dashboard_commands", False)))
            except Exception as e:
                LOGGER.warning(f"heartbeat failed: {e}")
            await asyncio.sleep(HEARTBEAT_S)

    async def _command_loop(self) -> None:
        """Run commands queued from the dashboard ("__reset__" or an object description), one at a time."""
        while True:
            try:
                if not self._busy():
                    cmd = self.store.next_command()
                    if cmd is not None:
                        text = (cmd.get("text") or "").strip()
                        if text == "__reset__":
                            res = await self._start("reset", None, wait=True)
                        else:
                            res = await self._grasp(_runtime().clean(text), wait=True)
                        status = res.get("status") or ("failed" if not res.get("accepted") else "done")
                        self.store.finish_command(cmd["_id"], status, res.get("_id"), res.get("label"))
                        continue
            except Exception as e:
                LOGGER.warning(f"command loop: {type(e).__name__}: {e}")
            await asyncio.sleep(1.0)

    async def close(self) -> None:
        for t in self.tasks + ([self.job] if self.job else []):
            t.cancel()
