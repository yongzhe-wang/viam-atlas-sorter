"""
Viam generic service `sorter-team:sorter:grasp-by-description`: describe an object, it goes in the blue square.

WHY: other robot builders should get our whole pipeline (perception + CLIP choice + depth-guided pick + reset
+ Atlas logging) by adding one service to their machine config, without our scripts or a laptop. Running
inside viam-server also removes the laptop <-> robot WebRTC link that dropped repeatedly during our trials:
the arm, gripper and camera are reached through viam-server's local connections.

Always ready: run.sh installs deps and downloads the CLIP weights before the module starts. At startup a
background task loads torch, CLIP (eval, all CPU threads), perception and the object library into memory
and runs one dummy image + text pass ("warm" in status); until then a grasp answers "warming up" at once.
While idle with the arm at the observe pose, the service looks at the table about once a second (every 2 s
when a look takes over 800 ms) and caches what it recognises ("seen", with per-step "seen_ms"). A grasp then
decides from that cached frame in milliseconds ("decide_ms") and either starts moving at once or answers at
once that the object is not on the table.

DoCommand:
  {"grasp": "green square"}          decide + start in the background -> {"ok": true, "accepted": true, "job": id,
                                     "decide_ms": ..., "target": {...}} or {"ok": false, "reason": "no ... on the table"}
  {"grasp": "...", "wait": true}     same, but block until done -> the attempt summary
  {"preview": "green square"}        the same decision without moving: what would be picked, or why not
  {"reset": true}                    clear the square (pick-out, wipe fallback, verify empty)
  {"status": true}                   busy / running / last / warm / seen / library / store mode
  {"seen_image": true}               last idle frame as base64 JPEG (<= 640 px) with the seen boxes scaled to it
  {"table_map": true}                latest whole-table map (no embeddings), object sightings, sweep progress
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
SLOW_WATCH_PERIOD_S = 2.0  # ... every 2 s when one look takes longer than SLOW_LOOP_MS, so jobs are not starved
SLOW_LOOP_MS = 800
SEEN_FRESH_S = 1.5  # a grasp decides from the cached frame only when it is at most this old (VIAM CENTERAL)
OBSERVE_TOL_MM = 15.0  # the arm counts as "at the observe pose" within this of HOME_XY / ALIGN_Z ...
OBSERVE_TOL_DEG = 1.0  # ... and theta 0: AGENT1's pixel->world mapping for a target holds only there
SEEN_IMAGE_WIDTH = 640
WARM_RETRY_S = 30.0
TIMELAPSE_EVERY_S = 5.0  # rolling timelapse for a backup demo video: one frame every 5 s ...
TIMELAPSE_KEEP_S = 2 * 3600.0  # ... kept for the last 2 hours
MAP_FRESH_S = 60.0  # a whole-table map younger than this is trusted for "where is it" (VIAM CENTERAL)


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
    quickly (warm-up then loads everything in the background). One place, so tests can swap it for stubs.
    """
    from sorter import clip_embed, executor, perception, table_map, worker

    return SimpleNamespace(ppa=executor.ppa, executor=executor, clean=worker.clean, perception=perception,
                           clip_embed=clip_embed, table_map=table_map)


def _check_xy(attrs: dict, key: str) -> None:
    v = attrs.get(key)
    if v is not None and not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
        raise ValueError(f"'{key}' must be a list of two numbers, got {v!r}")


def _box(b) -> list:
    return [round(float(x), 1) for x in b]


def _ms(t0: float) -> int:
    return round((time.monotonic() - t0) * 1000)


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
        self.library = self.store.library_stats()
        self.worker_id = f"module:{config.name}"
        self.rig = None
        self.job: Optional[asyncio.Task] = None
        self.job_info: Dict[str, Any] = {}
        self.last: Dict[str, Any] = {}
        self.warm, self.warm_ms, self.warm_error = False, None, None
        self.watch_period = WATCH_PERIOD_S
        self.seen: Dict[str, Any] = {}  # summary of the latest idle look, for status / dashboard
        self._seen_raw = None  # {"ts", "recs", "objs", "rgb", "depth", "pose"} behind self.seen, for a fast grasp
        self._jpeg_cache = (None, None, 1.0)  # (frame ts, base64 jpeg, scale) for seen_image
        self._no_target_warned = False
        self.timelapse_dir = (Path(module_data) if module_data else self.store.data) / "timelapse"
        loop = asyncio.get_event_loop()
        self.tasks = [loop.create_task(self._heartbeat_loop()), loop.create_task(self._warmup()),
                      loop.create_task(self._watch_loop()), loop.create_task(self._timelapse_loop())]
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

    def _warm_sync(self) -> None:
        """
        Everything the first command would otherwise pay for: imports, CLIP in eval mode on every CPU thread,
        the object library as one normalised in-memory matrix, and one dummy image + text pass (first-call
        allocation). The weights themselves were downloaded by run.sh into $VIAM_MODULE_DATA/cache.
        """
        import numpy as np
        import open_clip  # noqa: F401  a missing package should fail warm-up, not the first command
        import torch

        rt = _runtime()
        rt.clip_embed.embed_images([np.zeros((64, 64, 3), np.uint8)])  # loads the model (eval) on first call
        torch.set_num_threads(os.cpu_count() or 1)  # after the load, which sets its own thread count
        rt.clip_embed.embed_images([np.zeros((224, 224, 3), np.uint8)])
        rt.clip_embed.embed_text("warm up")
        self.store.library_matrix()
        self.library = self.store.library_stats()

    async def _warmup(self) -> None:
        """Run _warm_sync in the background right after start; retry every WARM_RETRY_S until it succeeds."""
        while not self.warm:
            t0 = time.monotonic()
            try:
                await asyncio.to_thread(self._warm_sync)
                self.warm, self.warm_ms, self.warm_error = True, _ms(t0), None
                LOGGER.info(f"{self.name}: warm in {self.warm_ms} ms ({self.library['views']} library views)")
            except Exception as e:
                self.warm_error = f"{type(e).__name__}: {e}"[:300]
                LOGGER.error(f"{self.name}: warm-up failed, retrying in {WARM_RETRY_S:.0f} s: {self.warm_error}")
                await asyncio.sleep(WARM_RETRY_S)

    def _perceive(self, rt, rgb, depth) -> tuple:
        """
        Targets in one frame with their library labels: (recs, [summaries], [intruders], step timings ms).

        recs = clip_embed.recognise_all() output (one CLIP batch for all crops), reused by best() at grasp time.
        Same filter as executor.find_target: nothing cut by the image border, nothing already in the square.
        """
        ms = {}
        t = time.monotonic()
        seg = rt.perception.segment_full(rgb, depth)
        ms["segment"] = _ms(t)
        t = time.monotonic()
        hole = rt.executor.hole_mask(rgb)
        ms["square"] = _ms(t)
        t = time.monotonic()
        if hasattr(rt.clip_embed, "recognise_all"):
            recs = rt.clip_embed.recognise_all(rgb, depth, self.store, seg=seg)
        else:  # older clip_embed without the batched call
            objs = list(seg.objects)
            recs = []
            for o, emb in zip(objs, rt.clip_embed.embed_images([o.crop for o in objs])):
                lab, names, conf = rt.clip_embed.recognise(emb, self.store, colour=o.colour, height_mm=o.height_mm)
                recs.append({"obj": o, "label": lab, "names": names, "score": conf, "emb": emb})
        recs = [r for r in recs if not r["obj"].touches_edge
                and (hole is None or (r["obj"].mask & hole).sum() < 0.5 * r["obj"].mask.sum())]
        ms["clip"] = _ms(t)
        objects = [{"id": int(r["obj"].id), "label": r["label"],
                    "aliases": [n for n in (r.get("names") or []) if n != r["label"]],
                    "confidence": round(float(r.get("score") or 0.0), 3), "colour": r["obj"].colour,
                    "height_mm": round(float(r["obj"].height_mm), 1), "bbox": _box(r["obj"].bbox)} for r in recs]
        intruders = [{"bbox": _box(i["bbox"]), "reason": i.get("reason", "")} for i in seg.intruders]
        return recs, objects, intruders, ms

    async def _pose(self, rt) -> tuple:
        """(pose dict, at_observe) from the arm's current end position."""
        p = await self.arm.get_end_position()
        hx, hy = rt.ppa.HOME_XY
        dtheta = abs(((p.theta + 180.0) % 360.0) - 180.0)
        at = (max(abs(p.x - hx), abs(p.y - hy), abs(p.z - rt.ppa.ALIGN_Z)) <= OBSERVE_TOL_MM
              and dtheta <= OBSERVE_TOL_DEG)
        return {"x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1), "theta": round(p.theta, 2)}, at

    async def _look(self) -> None:
        """
        One idle look: only from the observe pose and only if the arm did not move while we looked (another
        driver may be using it), so a cached frame always matches where the arm is. Records step timings and
        slows the loop to SLOW_WATCH_PERIOD_S when a look takes over SLOW_LOOP_MS.
        """
        rt = _runtime()
        pose, at = await self._pose(rt)
        if not at:
            self._seen_raw = None
            self.seen = {"ts": time.time(), "at_observe": False, "pose": pose, "objects": [], "intruders": [],
                         "reason": "arm is not at the observe pose"}
            return
        t_all = time.monotonic()
        t_cap = time.time()
        res = await self.camera.get_images()
        rgb, depth = rt.ppa.decode(res[0] if isinstance(res, tuple) else res)
        frame_ms = _ms(t_all)
        if rgb is None or depth is None:
            raise RuntimeError("camera returned no colour + depth frame")
        recs, objects, intruders, ms = await asyncio.to_thread(self._perceive, rt, rgb, depth)
        ms = {"frame": frame_ms, **ms, "total": _ms(t_all)}
        self.watch_period = SLOW_WATCH_PERIOD_S if ms["total"] > SLOW_LOOP_MS else WATCH_PERIOD_S
        pose2, still = await self._pose(rt)
        if self._busy() or not still or max(abs(pose2[k] - pose[k]) for k in ("x", "y", "z")) > 2.0:
            return  # arm moved meanwhile: this frame says nothing reliable about where things are now
        self._seen_raw = {"ts": t_cap, "recs": recs, "objs": [r["obj"] for r in recs], "rgb": rgb, "depth": depth,
                          "pose": pose}
        self.seen = {"ts": t_cap, "at_observe": True, "pose": pose, "objects": objects, "intruders": intruders,
                     "seen_ms": ms, "watch_period_s": self.watch_period}
        # Arm confirmed still: keep each recognised object's last-seen table xy current (AGENT3's where_is)
        if hasattr(rt.clip_embed, "record_sightings"):
            await asyncio.to_thread(rt.clip_embed.record_sightings, self.store, recs, pose, None, "idle")

    async def _watch_loop(self) -> None:
        """Idle perception, every watch_period, once warm and while no job runs."""
        while True:
            await asyncio.sleep(self.watch_period)
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

    # ---- timelapse ---------------------------------------------------------------------------------

    def _timelapse_save(self, jpeg: bytes, ts: float, label: str) -> None:
        """Downscale, stamp time + what the robot is doing, save as <epoch_ms>.jpg, drop frames older than 2 h."""
        from PIL import Image, ImageDraw

        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if im.width > SEEN_IMAGE_WIDTH:
            im = im.resize((SEEN_IMAGE_WIDTH, round(im.height * SEEN_IMAGE_WIDTH / im.width)))
        text = f"{time.strftime('%H:%M:%S', time.localtime(ts))}  {label}"
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(255, 255, 255))
        self.timelapse_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.timelapse_dir / f".{int(ts * 1000)}.tmp"
        im.save(tmp, "JPEG", quality=75)
        tmp.replace(self.timelapse_dir / f"{int(ts * 1000)}.jpg")
        cutoff = (ts - TIMELAPSE_KEEP_S) * 1000
        for p in self.timelapse_dir.glob("*.jpg"):
            if p.stem.isdigit() and int(p.stem) < cutoff:
                p.unlink(missing_ok=True)

    async def _timelapse_loop(self) -> None:
        """
        One colour frame every TIMELAPSE_EVERY_S, during jobs too (read-only camera), into
        $VIAM_MODULE_DATA/timelapse/, so a backup demo video can be cut from the last 2 hours.
        """
        while True:
            await asyncio.sleep(TIMELAPSE_EVERY_S)
            try:
                try:
                    res = await self.camera.get_images(filter_source_names=["color"])
                except Exception:
                    res = await self.camera.get_images()
                images = res[0] if isinstance(res, tuple) else res
                jpeg = next((im.data for im in images if im.mime_type == "image/jpeg"), None)
                if jpeg is None:
                    continue
                label = (f"{self.job_info.get('kind')}: {self.job_info.get('text') or 'square'}" if self._busy()
                         else "idle")
                await asyncio.to_thread(self._timelapse_save, jpeg, time.time(), label)
            except Exception as e:
                LOGGER.warning(f"timelapse frame skipped: {type(e).__name__}: {e}")

    def _timelapse_stats(self) -> dict:
        stamps = sorted(int(p.stem) for p in self.timelapse_dir.glob("*.jpg") if p.stem.isdigit()) \
            if self.timelapse_dir.exists() else []
        return {"frames": len(stamps), "span_s": round((stamps[-1] - stamps[0]) / 1000) if stamps else 0,
                "dir": str(self.timelapse_dir)}

    # ---- deciding --------------------------------------------------------------------------------------

    def _not_ready(self) -> Optional[Dict[str, ValueTypes]]:
        """Immediate answer while warm-up has not finished, so a command never blocks on loading CLIP."""
        if self.warm:
            return None
        reason = "warming up: loading CLIP and the object library"
        if self.warm_error:
            reason += f" (last attempt failed: {self.warm_error}; retrying)"
        return {"ok": False, "accepted": False, "status": "warming_up", "reason": reason}

    async def _decide(self, text: str) -> tuple:
        """
        clip_embed.best() on the cached idle frame: (pick, decide_ms, raw), or (None, None, raw) when there is no
        fresh frame. Reuses the frame's recognise_all() output, so no crop is embedded again.
        """
        raw = self._seen_raw
        if raw is None or time.time() - raw["ts"] > SEEN_FRESH_S:
            return None, None, raw
        rt = _runtime()
        t0 = time.monotonic()
        if "recognised" in inspect.signature(rt.clip_embed.best).parameters:
            pick = await asyncio.to_thread(rt.clip_embed.best, None, text, self.store, recognised=raw["recs"])
        else:
            pick = await asyncio.to_thread(rt.clip_embed.best, raw["objs"], text, self.store)
        return pick, _ms(t0), raw

    @staticmethod
    def _refusal(pick: dict, text: str) -> Optional[tuple]:
        """(status, reason) when the decision is not to pick anything, else None."""
        if pick.get("obj") is None:
            return "not_found", f"no {text!r} on the table" + (f" ({pick['reason']})" if pick.get("reason") else "")
        if pick.get("ambiguous"):
            return "ambiguous", f"two objects match {text!r} about equally; be more specific"
        return None

    @staticmethod
    def _target_summary(pick: dict) -> dict:
        o = pick["obj"]
        return {"id": int(o.id), "label": pick.get("recognised"), "score": round(float(pick.get("score", 0.0)), 3),
                "bbox": _box(o.bbox)}

    async def _from_map(self, text: str) -> Optional[dict]:
        """
        best_on_map() on the whole-table map when it is at most MAP_FRESH_S old:
        {"found", "xy", "label", "status", "reason", "decide_ms", "map_age_s"}, or None when there is no fresh map.
        """
        m = self.store.get_table_map()
        if not m or m["age_s"] > MAP_FRESH_S:
            return None
        t0 = time.monotonic()
        b = await asyncio.to_thread(_runtime().table_map.best_on_map, m, text, self.store)
        out = {"decide_ms": _ms(t0), "map_age_s": m["age_s"]}
        if b.get("map_obj") is not None and not b.get("ambiguous"):
            return {**out, "found": True, "xy": [round(float(v), 1) for v in b["xy"]], "label": b["map_obj"].get("label"),
                    "score": round(float(b.get("score") or 0.0), 3)}
        if b.get("ambiguous"):
            return {**out, "found": False, "status": "ambiguous",
                    "reason": f"two objects on the table match {text!r} about equally; be more specific"}
        return {**out, "found": False, "status": "not_found",
                "reason": f"no {text!r} on the table (whole-table map {m['age_s']:.0f} s old)"}

    async def _preview(self, text: str) -> Dict[str, ValueTypes]:
        """
        What a grasp for `text` would do, without moving (dashboard Preview): pick from the cached frame, else go
        to where a fresh table map saw it, else it would scan the table.
        """
        busy_or_cold = self._not_ready()
        if busy_or_cold:
            return busy_or_cold
        pick, decide_ms, raw = await self._decide(text)
        refusal = self._refusal(pick, text) if pick is not None else None
        if pick is not None and not refusal:
            return {"ok": True, "via": "frame", "target": self._target_summary(pick), "decide_ms": decide_ms,
                    "decided_from_cache": True, "frame_ts": raw["ts"]}
        if refusal and refusal[0] == "ambiguous":
            return {"ok": False, "status": "ambiguous", "reason": refusal[1], "decide_ms": decide_ms,
                    "decided_from_cache": True}
        found = await self._from_map(text)
        if found is None:
            return {"ok": False, "via": "sweep", "status": "would_scan",
                    "reason": f"{text!r} is not in view and there is no fresh table map: a grasp would scan the table"}
        if found["found"]:
            x, y = found["xy"]
            return {"ok": True, "via": "map", "target_xy": found["xy"], "label": found["label"],
                    "decide_ms": found["decide_ms"], "map_age_s": found["map_age_s"],
                    "message": f"remembered at ({x:.0f}, {y:.0f}) on the table map"}
        return {"ok": False, "status": found["status"], "reason": found["reason"], "decide_ms": found["decide_ms"]}

    def _sweep_progress(self, k: int, n: int):
        """
        Executor callback during a table sweep: shows "pose k/N" in status. Returns a finished future, so it
        works whether the executor calls it plainly or awaits it.
        """
        self.job_info["sweep"] = {"k": int(k), "n": int(n)}
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut

    # ---- jobs --------------------------------------------------------------------------------------

    async def _run(self, kind: str, text: Optional[str], target: Optional[dict] = None,
                   decide_ms: Optional[int] = None, target_xy: Optional[list] = None, sweep: bool = False) -> dict:
        """
        Run one grasp or reset to completion and return its summary; never raises.

        Grasp, in order of what is known: target (from the cached frame) -> executor skips its own first look;
        target_xy (from a fresh table map) -> executor looks there and picks; sweep -> executor scans the whole
        table, we build and save the map (AGENT3's map_from_frames) and pick from it. Each step needs the matching
        executor entry point (AGENT1); without it the executor's own search (find_target) is used instead.
        """
        via = "search"
        try:
            rt = _runtime()
            rig = await self._get_rig()
            if kind == "grasp":
                params = inspect.signature(rt.executor.execute).parameters
                if sweep and hasattr(rt.executor, "sweep_table"):
                    frames = await rt.executor.sweep_table(rig, progress=self._sweep_progress)
                    m = await asyncio.to_thread(rt.table_map.map_from_frames, frames, self.store)
                    self.store.put_table_map(m)
                    b = await asyncio.to_thread(rt.table_map.best_on_map, m, text, self.store)
                    if b.get("map_obj") is None or b.get("ambiguous"):
                        status = "ambiguous" if b.get("ambiguous") else "not_found"
                        summary = {"status": status, "fail_phase": "find", "via": "sweep",
                                   "fail_reason": f"no {text!r} on the table after scanning it" if status == "not_found"
                                   else f"two objects match {text!r} about equally after scanning the table"}
                        self.last = {"kind": kind, "text": text, "job": self.job_info.get("job"),
                                     "finished": time.time(), **summary}
                        return summary
                    target_xy, via = [float(v) for v in b["xy"]], "sweep"
                kwargs = {}
                if target is not None and "target" in params:
                    kwargs, via = {"target": target}, "frame"
                elif target_xy is not None and "target_xy" in params:
                    kwargs = {"target_xy": tuple(target_xy)}
                    via = via if via == "sweep" else "map"
                elif (target is not None or target_xy is not None) and not self._no_target_warned:
                    LOGGER.warning("executor.execute lacks target=/target_xy=: using its own search instead")
                    self._no_target_warned = True
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
            summary.update(fast_path=via == "frame", via=via, decide_ms=decide_ms)
        # job id lets a DoCommand caller (the dashboard in module mode) match this result to its request
        self.last = {"kind": kind, "text": text, "job": self.job_info.get("job"), "finished": time.time(), **summary}
        return summary

    async def _start(self, kind: str, text: Optional[str], wait: bool, target: Optional[dict] = None,
                     decide_ms: Optional[int] = None, target_xy: Optional[list] = None,
                     sweep: bool = False) -> Dict[str, ValueTypes]:
        if self._busy():
            return {"accepted": False, "error": "busy", "running": dict(self.job_info)}
        job_id = uuid.uuid4().hex[:12]
        via = "frame" if target is not None else "map" if target_xy is not None else "sweep" if sweep else "search"
        self.job_info = {"job": job_id, "kind": kind, "text": text or "", "started": time.time(), "via": via,
                         "fast_path": target is not None, "target_xy": target_xy}
        self.job = asyncio.get_event_loop().create_task(self._run(kind, text, target, decide_ms, target_xy, sweep))
        if wait:
            return {"accepted": True, "job": job_id, **(await self.job)}
        return {"accepted": True, "job": job_id}

    async def _grasp(self, text: str, wait: bool) -> Dict[str, ValueTypes]:
        """
        Find the object fast, in this order (VIAM CENTERAL): the cached idle frame -> go at once; a fresh
        whole-table map (<= MAP_FRESH_S) -> go to where it saw it; otherwise scan the table, then pick. Answers at
        once when the object is ambiguous, or when a fresh map says it is not on the table.
        """
        if self._busy():
            return {"ok": False, "accepted": False, "error": "busy", "running": dict(self.job_info)}
        cold = self._not_ready()
        if cold:
            return cold
        rt = _runtime()
        pick, decide_ms, raw = await self._decide(text)
        refusal = self._refusal(pick, text) if pick is not None else None
        if pick is not None and not refusal:
            target = {**pick, "rgb": raw["rgb"], "depth": raw["depth"], "pose": raw["pose"], "ts": raw["ts"]}
            # fast_path: the executor will skip its own first observe (needs AGENT1's execute(..., target=...))
            fast = "target" in inspect.signature(rt.executor.execute).parameters
            res = await self._start("grasp", text, wait, target=target, decide_ms=decide_ms)
            if not res.get("accepted"):
                return {**res, "ok": False}  # became busy while deciding
            return {**res, "ok": True, "via": "frame", "decided_from_cache": True, "fast_path": fast,
                    "decide_ms": decide_ms, "target": self._target_summary(pick)}
        if refusal and refusal[0] == "ambiguous":
            self.last = {"kind": "grasp", "text": text, "job": None, "finished": time.time(), "status": "ambiguous",
                         "fail_phase": "find", "fail_reason": refusal[1], "decide_ms": decide_ms, "decided_from_cache": True}
            return {"ok": False, "accepted": False, "status": "ambiguous", "reason": refusal[1], "decide_ms": decide_ms,
                    "decided_from_cache": True}
        # Not in the cached frame (or no fresh frame): the whole-table map, else a sweep
        found = await self._from_map(text)
        if found is not None and not found["found"]:
            self.last = {"kind": "grasp", "text": text, "job": None, "finished": time.time(), "status": found["status"],
                         "fail_phase": "find", "fail_reason": found["reason"], "decide_ms": found["decide_ms"]}
            return {"ok": False, "accepted": False, "status": found["status"], "reason": found["reason"],
                    "decide_ms": found["decide_ms"], "via": "map"}
        if found is not None:
            x, y = found["xy"]
            res = await self._start("grasp", text, wait, target_xy=found["xy"], decide_ms=found["decide_ms"])
            return {**res, "ok": bool(res.get("accepted")), "via": "map", "status_text": "going_to_map",
                    "message": f"remembered at ({x:.0f}, {y:.0f}) on the table map · going there",
                    "target_xy": found["xy"], "label": found["label"], "map_age_s": found["map_age_s"],
                    "decide_ms": found["decide_ms"]}
        has_sweep = hasattr(rt.executor, "sweep_table")
        res = await self._start("grasp", text, wait, sweep=True)
        return {**res, "ok": bool(res.get("accepted")), "via": "sweep" if has_sweep else "search",
                "status_text": "scanning" if has_sweep else "searching",
                "message": "scanning the table…" if has_sweep else "searching the observe positions…",
                "why": "not in the cached frame" if pick is not None else "no fresh frame from the observe pose"}

    def _status(self) -> Dict[str, ValueTypes]:
        seen = dict(self.seen)
        if seen.get("ts"):
            seen["age_s"] = round(time.time() - seen["ts"], 1)
        return {"busy": self._busy(), "running": dict(self.job_info) if self._busy() else {}, "last": dict(self.last),
                "store": "atlas" if self.store.online else "jsonl", "worker_id": self.worker_id,
                "library": dict(self.library), "warm": self.warm, "warm_ms": self.warm_ms,
                "warm_error": self.warm_error, "seen": seen, "timelapse": self._timelapse_stats()}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        if command.get("status"):
            return self._status()
        if command.get("seen_image"):
            return self._seen_image()
        if command.get("table_map"):
            # Dashboard top-down view: latest whole-table map (no embeddings) + each object's latest sighting
            m = self.store.get_table_map(with_emb=False)
            return {"ok": m is not None, "map": m or {}, "sightings": self.store.list_sightings()[:40],
                    "sweep": dict(self.job_info.get("sweep") or {}) if self._busy() else {},
                    "now": time.time()}  # robot clock, so the dashboard can age sightings without clock skew
        for key in ("grasp", "preview"):
            if key in command:
                text = _runtime().clean(str(command.get(key) or ""))
                if not text:
                    return {"ok": False, "accepted": False, "error": f"say what to {key}, e.g. {{\"{key}\": \"green square\"}}"}
                return await (self._grasp(text, bool(command.get("wait"))) if key == "grasp" else self._preview(text))
        if command.get("reset"):
            return await self._start("reset", None, bool(command.get("wait")))
        return {"error": "unknown command; use {\"grasp\": \"<description>\"}, {\"preview\": \"<description>\"}, "
                         "{\"reset\": true}, {\"status\": true}, {\"seen_image\": true} or {\"table_map\": true}"}

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
