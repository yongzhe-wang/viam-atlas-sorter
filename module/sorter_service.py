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
  {"scan_table": true}               scan the whole table into the live map (only on request: a grasp never scans
                                     first; it searches and picks at the first view that shows the object)
  {"status": true}                   busy / running / last / warm / seen / library / store mode / live map stats
  {"seen_image": true}               last idle frame as base64 JPEG (<= 640 px) with the seen boxes scaled to it
  {"table_map": true}                the live table map (no embeddings; each object's seen_ts), sightings, sweep progress
  {"clear_map": true}                forget the table (someone rearranged it): live map + sightings + cached frame;
                                     the old map and sightings go to the Store's history. Refused while a job runs.
  {"rebuild_map": {"since": ts}}     rebuild the live map from the observation log (every frame folded in is
                                     logged) since ts, default the current epoch; needs AGENT3's fold_observations
The map also starts empty at every module start: the table may have changed while the module was down.
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
import json
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

from module.live_map import LiveMap

NAMESPACE = "sorter-team"  # must match "module_id" in meta.json; change both to your registry namespace
LOGGER = getLogger(__name__)
SUPPORTED_SQUARES = ("blue",)  # executor/pick_place_auto find the drop square by blue tape
HEARTBEAT_S = 5.0  # dashboard marks a worker offline after 15 s (agreed with AGENT1)
# Object library shipped in the archive by module/build.sh (views.jsonl + img/); read-only, versioned
BUNDLE_DIR = Path(__file__).resolve().parent.parent / "library"
# build.sh writes which commit an archive holds; status reports it, so everyone can see what the robot runs
BUILD_INFO_PATH = Path(__file__).resolve().parent.parent / "BUILD_INFO.json"
# find_path / square_path (AGENT1 0d3c817): how the executor really found the object and the square, for status.last
SUMMARY_KEYS = ("_id", "status", "label", "colour", "score", "height_mm", "duration_s", "fail_phase", "fail_reason",
                "find_path", "find_views", "square_path", "objects", "check")
# A grasp that fails by accident (slipped, closed on air) is tried this many times in all, each from a fresh
# search {USER 2026-10-29 "I ALSO WANT RETRY WHEN FAIL"} [CONFIDENCE: SINGLE-SRC 90% — 3 is my pick: a 4th
# try past ~45 s of retries loses the demo minute anyway]
GRASP_ATTEMPTS = 3
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
MAP_FRESH_S = 60.0  # a live-map object seen this recently is gone to directly; a sweep this recent covers the table
MAP_PERSIST_S = 2.0  # live map -> Store.put_table_map at most this often (VIAM CENTERAL)
MAP_TASK_EVERY_S = 1.0  # during a task, at most one executor frame per second goes to the live map (servo keeps CPU)
IDLE_SIG_PX = 12  # idle pictures differ when a box centre moved this many pixels (or labels / intruders changed)
IDLE_MAP_KEEPALIVE_S = 20.0  # ... else one idle picture this often still goes into the map and the log
OBS_THIN_EVERY_S = 60.0  # how often the observation log checks free disk space (thins idle images under 5 GB)
SWEEPS_KEEP = 30  # saved sweep frame sets (AGENT1, ~8 MB each: 4 x jpg + 1280x720 uint16 depth): keep the newest 30


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


def _build_info() -> dict:
    """{commit, dirty, built_at} of this archive (module/build.sh), or a note when running from a checkout."""
    try:
        return json.loads(BUILD_INFO_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return {"commit": None, "note": "not a built archive (no BUILD_INFO.json)"}


def _runtime():
    """
    The arm/perception stack, imported on first use so viam-server can start the module and validate config
    quickly (warm-up then loads everything in the background). One place, so tests can swap it for stubs.
    """
    from sorter import clip_embed, drop_square, executor, perception, table_map, worker

    return SimpleNamespace(ppa=executor.ppa, executor=executor, clean=worker.clean, perception=perception,
                           clip_embed=clip_embed, table_map=table_map, drop_square=drop_square)


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
        if data_dir is not None:
            # AGENT1's executor saves every sweep frame under $SORTER_DATA_DIR/sweeps, read when it is imported
            # (warm-up, later): next to the Store data, kept across versions, not inside the replaced package
            os.environ.setdefault("SORTER_DATA_DIR", str(data_dir))
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
        self._look_lock = asyncio.Lock()  # one look at a time: the idle loop and a grasp that needs a fresh frame
        self.live: Optional[LiveMap] = None  # the live table map, once warm and if AGENT3's update_map exists
        self._intr: Optional[dict] = None  # camera intrinsics for update_map, read once through the Rig
        self._sweeping = False  # while a sweep runs, every frame goes to the live map (none dropped)
        self._task_map_at = 0.0  # when the last other executor frame went to the live map
        self._thinned_at = 0.0  # when the observation log last checked free disk space
        self._idle_sig, self._idle_fed_at = None, 0.0  # last idle picture fed to the live map: what it showed, when
        self._no_target_warned = False
        self.timelapse_dir = (Path(module_data) if module_data else self.store.data) / "timelapse"
        loop = asyncio.get_event_loop()
        # Start from an empty table: the saved map and sightings go to history before anything can find from them
        self._forgotten = loop.create_task(self._forget_table("module start"))
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

    async def _intrinsics(self) -> dict:
        """Camera intrinsics {fx, fy, cx, cy} for update_map, read once (by the Rig, from the camera)."""
        if self._intr is None:
            rig = await self._get_rig()
            self._intr = {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy}
        return self._intr

    def _on_map_frame(self, rig, rgb, depth, pose: dict) -> None:
        """
        Rig.on_map_frame (AGENT1): called inside the arm's control loop for every frame with depth, with the pose
        it was taken at (theta 0 only while ppa.MAP_ONLY_THETA0). It must not block, so it only hands the frame to
        the live map's worker: during a sweep every frame (keep, none dropped, applied while the arm moves on);
        otherwise at most one per MAP_TASK_EVERY_S while no update runs, so update_map's 150-300 ms per frame
        does not compete with every servo probe for the CPU. The source is the executor's phase (servo, check...).
        """
        if self.live is None or depth is None:
            return
        if not self._sweeping:
            if self.live.busy() or time.time() - self._task_map_at < MAP_TASK_EVERY_S:
                return
            self._task_map_at = time.time()
        rt = _runtime()
        intr = self._intr or {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy}
        phase = rt.executor.phase_now(rig) if hasattr(rt.executor, "phase_now") else "task"
        self.live.offer(rgb, depth, pose, intr, "sweep" if self._sweeping else phase, keep=self._sweeping)

    async def _forget_table(self, reason: str) -> dict:
        """
        Module start (VIAM CENTERAL: "remember to clear [the map] if we restart"; the table may have changed while
        the module was down): the saved map and the sightings move to the Store's history, an empty map becomes
        the latest. Returns that empty map, which the live map starts from. A Store error only skips the history.
        """
        try:
            empty = await asyncio.to_thread(self.store.clear_table_map, reason)
            n = await asyncio.to_thread(self.store.clear_sightings)
            await asyncio.to_thread(self.store.observation_stats)  # one pass over the log now, not in a status call
            LOGGER.info(f"{self.name}: table cleared at {reason}: {empty['cleared_objects']} map objects and "
                        f"{n} sightings moved to history")
            return empty
        except Exception as e:
            LOGGER.warning(f"{self.name}: could not move the saved map / sightings to history: {type(e).__name__}: {e}")
            now = time.time()
            return {"ts": now, "objects": [], "frames": [], "cleared_ts": now, "cleared_reason": reason,
                    "cleared_objects": 0}

    async def _start_live_map(self) -> None:
        """The one live map (AGENT3's table_map.update_map), empty from the start; part of warm-up."""
        if self.live is None:
            initial = await self._forgotten
            tm = _runtime().table_map
            self.live = LiveMap(self.store, tm.update_map, MAP_PERSIST_S, initial=initial, empty_fn=tm.empty_map)
            LOGGER.info(f"{self.name}: live table map on, empty since the module started")

    async def _rebuild_map(self, since: Optional[float]) -> Dict[str, ValueTypes]:
        """
        {"rebuild_map": {"since": ts}}: the live map from the observation log since ts (default: the current
        epoch), folded by AGENT3's table_map.fold_observations, e.g. after a crash. Refused while a job runs.
        """
        fold = getattr(_runtime().table_map, "fold_observations", None)
        if fold is None:
            return {"ok": False, "error": "rebuilding needs table_map.fold_observations (AGENT3)"}
        if self._busy():
            return {"ok": False, "error": "busy", "running": dict(self.job_info)}
        if self.live is None:
            return {"ok": False, "error": "warming up"}
        try:
            async with self._look_lock:
                res = await self.live.rebuild(float(since if since is not None else self.live.epoch), fold)
        except Exception as e:  # e.g. AGENT3's fold refusing a record without its observation fields
            return {"ok": False, "error": f"rebuild failed: {type(e).__name__}: {e}"[:300]}
        LOGGER.info(f"{self.name}: map rebuilt from {res['observations']} observations: {res['objects']} objects")
        return {"ok": True, **res}

    async def _clear_map(self, reason: str) -> Dict[str, ValueTypes]:
        """
        {"clear_map": true}: someone rearranged the table, so forget it: the live map (old one to history), every
        sighting, the cached idle frame. The next grasp decides from a fresh frame or sweeps, as usual.
        Refused while a job runs, since it is reading and updating the map.
        """
        if self._busy():
            return {"ok": False, "error": "busy", "running": dict(self.job_info)}
        async with self._look_lock:  # no idle look half done across the clear
            if self.live is not None:
                empty = await self.live.clear(reason)
            else:
                empty = await asyncio.to_thread(self.store.clear_table_map, reason)
            n = await asyncio.to_thread(self.store.clear_sightings)
            self._seen_raw = None
            self.seen = {"ts": time.time(), "at_observe": None, "objects": [], "intruders": [], "reason": "map cleared"}
        LOGGER.info(f"{self.name}: map cleared ({reason}): {empty['cleared_objects']} objects, {n} sightings to history")
        return {"ok": True, "cleared_objects": empty["cleared_objects"], "cleared_sightings": n,
                "cleared_ts": empty["cleared_ts"]}

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
                await self._start_live_map()
                self.warm, self.warm_ms, self.warm_error = True, _ms(t0), None
                LOGGER.info(f"{self.name}: warm in {self.warm_ms} ms ({self.library['views']} library views)")
            except Exception as e:
                self.warm_error = f"{type(e).__name__}: {e}"[:300]
                LOGGER.error(f"{self.name}: warm-up failed, retrying in {WARM_RETRY_S:.0f} s: {self.warm_error}")
                await asyncio.sleep(WARM_RETRY_S)

    @staticmethod
    def _delivered(rt, o, hole, pose, intr, square) -> bool:
        """
        Already in the drop square, by the executor's look_here rule (AGENT1 bc07e1e): in this frame (in_hole), or
        its grasp point's table xy inside the live map's estimated full square (AGENT3's drop_square: any rotation,
        also while only part of the paper is in view). Never fails a look: an error counts as not delivered.
        """
        try:
            if hole is not None and rt.executor.in_hole(o, hole):
                return True
            ds = getattr(rt, "drop_square", None)
            if not square or ds is None or pose is None or intr is None or not hasattr(rt.executor, "grasp_px"):
                return False
            gu, gv, _ = rt.executor.grasp_px(o, pose, intr)
            return bool(ds.contains(square, rt.perception.world_xy(pose, intr, gu, gv, o.top_depth_mm)))
        except Exception as e:
            LOGGER.debug(f"in-square test skipped: {type(e).__name__}: {e}")
            return False

    def _perceive(self, rt, rgb, depth, pose: dict = None, intr: dict = None, square: dict = None) -> tuple:
        """
        Targets in one frame with their library labels:
        (recs, [summaries], [intruders], step timings ms, seg, all_recs).

        recs = clip_embed.recognise_all() output (one CLIP batch for all crops), reused by best() at grasp time.
        Same filter as executor.find_target: nothing cut by the image border, nothing already in the square.
        all_recs (with the edge-cut "partial" ones, AGENT3) and seg go to update_map, which then neither segments
        nor embeds again (38 ms instead of ~210 ms, AGENT3's measurement) and can keep edge sightings.
        """
        ms = {}
        t = time.monotonic()
        seg = rt.perception.segment_full(rgb, depth)
        ms["segment"] = _ms(t)
        t = time.monotonic()
        # With depth, a square cut by the picture's edge (paper pushed) still gives a hole (AGENT1 ce17ecf); without
        # it, the yellow block already in the square was a frame-path candidate (AGENT3)
        hole = rt.executor.hole_mask(rgb, depth)
        ms["square"] = _ms(t)
        t = time.monotonic()
        if hasattr(rt.clip_embed, "recognise_all"):
            partial = {"include_partial": True} if "include_partial" in inspect.signature(rt.clip_embed.recognise_all).parameters else {}
            recs = rt.clip_embed.recognise_all(rgb, depth, self.store, seg=seg, **partial)
        else:  # older clip_embed without the batched call
            objs = list(seg.objects)
            recs = []
            for o, emb in zip(objs, rt.clip_embed.embed_images([o.crop for o in objs])):
                lab, names, conf = rt.clip_embed.recognise(emb, self.store, colour=o.colour, height_mm=o.height_mm)
                recs.append({"obj": o, "label": lab, "names": names, "score": conf, "emb": emb})
        all_recs = recs
        # Same rule as the executor's look_here: nothing already in the drop square is a candidate (_delivered)
        recs = [r for r in recs if not r.get("partial") and not r["obj"].touches_edge
                and not self._delivered(rt, r["obj"], hole, pose, intr, square)]
        ms["clip"] = _ms(t)
        objects = [{"id": int(r["obj"].id), "label": r["label"],
                    "aliases": [n for n in (r.get("names") or []) if n != r["label"]],
                    "confidence": round(float(r.get("score") or 0.0), 3), "colour": r["obj"].colour,
                    "height_mm": round(float(r["obj"].height_mm), 1), "bbox": _box(r["obj"].bbox)} for r in recs]
        intruders = [{"bbox": _box(i["bbox"]), "reason": i.get("reason", "")} for i in seg.intruders]
        return recs, objects, intruders, ms, seg, all_recs

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
        square = (self.live.map or {}).get("square") if self.live is not None else None
        recs, objects, intruders, ms, seg, all_recs = await asyncio.to_thread(
            self._perceive, rt, rgb, depth, pose, await self._intrinsics(), square)
        ms = {"frame": frame_ms, **ms, "total": _ms(t_all)}
        self.watch_period = SLOW_WATCH_PERIOD_S if ms["total"] > SLOW_LOOP_MS else WATCH_PERIOD_S
        pose2, still = await self._pose(rt)
        if self._busy() or not still or max(abs(pose2[k] - pose[k]) for k in ("x", "y", "z")) > 2.0:
            return  # arm moved meanwhile: this frame says nothing reliable about where things are now
        self._seen_raw = {"ts": t_cap, "recs": recs, "objs": [r["obj"] for r in recs], "rgb": rgb, "depth": depth,
                          "pose": pose}
        self.seen = {"ts": t_cap, "at_observe": True, "pose": pose, "objects": objects, "intruders": intruders,
                     "seen_ms": ms, "watch_period_s": self.watch_period}
        if self.live is not None and self._idle_changed(all_recs, seg):  # reusing this frame's recognition
            self.live.offer(rgb, depth, pose, await self._intrinsics(), "idle", recognised=all_recs, seg=seg)
        # Arm confirmed still: keep each recognised object's last-seen table xy current (AGENT3's where_is)
        if hasattr(rt.clip_embed, "record_sightings"):
            await asyncio.to_thread(rt.clip_embed.record_sightings, self.store, recs, pose, None, "idle")

    def _idle_changed(self, recs: list, seg) -> bool:
        """
        Should this idle picture go into the live map (and so the observation log)? Yes when what it shows changed
        (labels and box centres to IDLE_SIG_PX, intruders) or IDLE_MAP_KEEPALIVE_S passed (seen_ts stays fresh).
        WHY: a still table at 1 frame/s is ~40 MB/h of log records for nothing (AGENT3); a frame left out of the
        log must stay out of the map too, so live and replay agree.
        """
        sig = tuple(sorted((r.get("label") or "", round((r["obj"].bbox[0] + r["obj"].bbox[2]) / 2 / IDLE_SIG_PX),
                            round((r["obj"].bbox[1] + r["obj"].bbox[3]) / 2 / IDLE_SIG_PX)) for r in recs))
        sig += (len(getattr(seg, "intruders", []) or []),)
        now = time.time()
        if sig == self._idle_sig and now - self._idle_fed_at < IDLE_MAP_KEEPALIVE_S:
            return False
        self._idle_sig, self._idle_fed_at = sig, now
        return True

    async def _watch_loop(self) -> None:
        """Idle perception, every watch_period, once warm and while no job runs."""
        while True:
            await asyncio.sleep(self.watch_period)
            if not self.warm or self._busy():
                continue
            try:
                async with self._look_lock:
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
                label = (self._job_label() if self._busy()
                         else "idle")
                await asyncio.to_thread(self._timelapse_save, jpeg, time.time(), label)
                await asyncio.to_thread(self._prune_sweeps)
                if time.time() - self._thinned_at >= OBS_THIN_EVERY_S:
                    self._thinned_at = time.time()
                    freed = await asyncio.to_thread(self.store.thin_observations)
                    if freed:
                        LOGGER.warning(f"{self.name}: disk under 5 GB free: {freed / 1e6:.0f} MB of old idle frames thinned")
            except Exception as e:
                LOGGER.warning(f"timelapse frame skipped: {type(e).__name__}: {e}")

    def _prune_sweeps(self) -> None:
        """Keep the newest SWEEPS_KEEP saved sweeps (timestamped dirs), so the robot's disk does not fill up."""
        root = getattr(_runtime().executor, "SWEEP_DIR", None)
        if root is None or not Path(root).is_dir():
            return
        dirs = sorted((d for d in Path(root).iterdir() if d.is_dir()), key=lambda d: d.name)
        for d in dirs[:-SWEEPS_KEEP]:
            shutil.rmtree(d, ignore_errors=True)

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

    async def _fresh_look(self) -> None:
        """
        Look now when the cached frame is older than SEEN_FRESH_S, so a command sent right after a task (the arm
        just got back to the observe pose) decides from the table as it is, instead of waiting for the next idle
        tick and falling back to the map or a sweep (VIAM CENTERAL: "no fresh frame from the observe pose" after a
        reset). Off the observe pose _look() only records why; a running job is never interrupted.
        """
        async with self._look_lock:  # the idle loop mid-look: wait for it and use its frame
            raw = self._seen_raw
            if self._busy() or (raw is not None and time.time() - raw["ts"] <= SEEN_FRESH_S):
                return
            try:
                await self._look()
            except Exception as e:
                LOGGER.warning(f"look before deciding failed: {type(e).__name__}: {e}")

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
        return {"id": int(o.id), "label": pick.get("recognised"), "colour": getattr(o, "colour", None),
                "score": round(float(pick.get("score", 0.0)), 3), "bbox": _box(o.bbox)}

    async def _from_map(self, text: str) -> Optional[dict]:
        """
        The live map as a hint, never as an extra step (user: "just search, and if found just start grasp; do NOT
        have an initial step"). best_on_map() names an object -> {"found": True, "xy", "label", "seen_s_ago",
        "stale", "look_first", ...}: the search looks there first, however long ago it was seen (stale: more than
        MAP_FRESH_S; AGENT1: an old xy is still the best first look, and the search goes on if it is gone). Only in
        the drop square (AGENT3) -> {"found": False, "status": "already_in_square", "reason"}. Else None: search.
        """
        now = time.time()
        m = self.live.map
        if not m or not m.get("objects"):
            return None
        t0 = time.monotonic()
        b = await asyncio.to_thread(_runtime().table_map.best_on_map, m, text, self.store)
        out = {"decide_ms": _ms(t0), "map_age_s": round(now - float(m.get("ts") or 0.0), 1)}
        o = b.get("map_obj")
        if o is not None and not b.get("ambiguous"):
            seen_ago = now - float(o.get("seen_ts") or m.get("ts") or 0.0)
            return {**out, "found": True, "xy": [round(float(v), 1) for v in b["xy"]], "label": o.get("label"),
                    "colour": o.get("colour"), "score": round(float(b.get("score") or 0.0), 3),
                    "seen_s_ago": round(seen_ago, 1), "stale": seen_ago > MAP_FRESH_S,
                    "look_first": bool(b.get("look_first"))}
        if b.get("already_in_square"):
            # AGENT3: the named object is only in the drop square (delivered): say so at once, never search for it
            # or pick it out again
            return {**out, "found": False, "status": "already_in_square", "reason": b.get("reason")}
        return None

    @staticmethod
    def _map_message(found: dict) -> str:
        """'remembered at (x, y) on the table map, seen N s ago' (or 'last seen ... N min ago' when stale)."""
        x, y, ago = found["xy"][0], found["xy"][1], found["seen_s_ago"]
        when = f"{ago:.0f} s ago" if ago < 90 else f"{ago / 60:.0f} min ago"
        edge = " (only seen at a picture's edge)" if found.get("look_first") else ""
        if found.get("stale"):
            return f"last seen at ({x:.0f}, {y:.0f}) {when}{edge}"
        return f"remembered at ({x:.0f}, {y:.0f}) on the table map, seen {when}{edge}"

    async def _preview(self, text: str) -> Dict[str, ValueTypes]:
        """
        What a grasp for `text` would do, without moving (dashboard Preview): pick from the cached frame, else
        look first where the live map saw it, else search the table (and grasp at the first view that shows it).
        """
        busy_or_cold = self._not_ready()
        if busy_or_cold:
            return busy_or_cold
        await self._fresh_look()  # same frame a grasp right now would decide from
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
            return {"ok": False, "via": "search", "status": "would_search",
                    "reason": f"{text!r} is not in view: a grasp would search the table and pick it at the first "
                              f"view that shows it"}
        if found["found"]:
            return {"ok": True, "via": "map", "target_xy": found["xy"], "label": found["label"],
                    "decide_ms": found["decide_ms"], "map_age_s": found["map_age_s"], "seen_s_ago": found["seen_s_ago"],
                    "look_first": found["look_first"], "stale": found["stale"],
                    "message": self._map_message(found) + ": a grasp would look there first"
                               + (", then search the table" if found["stale"] else "")}
        return {"ok": False, "status": found["status"], "reason": found["reason"], "decide_ms": found["decide_ms"]}

    @staticmethod
    def _done_future():
        """A finished future: executor callbacks work whether the executor calls them plainly or awaits them."""
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut

    def _sweep_progress(self, k: int, n: int):
        """Executor callback during a table scan ({"scan_table"}): "scanning table… pose k/N" in status."""
        self.job_info["sweep"] = {"k": int(k), "n": int(n)}
        return self._done_future()

    def _search_progress(self, k: int, n: int, path: Optional[str] = None, found: bool = False):
        """
        Executor callback after each view of search_and_find (AGENT1): "searching… view k/N", and once best()
        accepts, "found at view k, grasping" (the search stops there and the pick starts at once).
        """
        self.job_info["search"] = {"k": int(k), "n": int(n), "path": path, "found": bool(found)}
        return self._done_future()

    def _job_label(self) -> str:
        """What the running job is, for heartbeat, timelapse and seen_image captions."""
        kind = self.job_info.get("kind")
        what = {"reset": "square", "scan": "table"}.get(kind) or self.job_info.get("text") or ""
        return f"{kind}: {what}"

    # ---- jobs --------------------------------------------------------------------------------------

    async def _scan(self, rt, rig) -> dict:
        """
        {"scan_table": true}, only on request (a grasp never scans first): every pose of the executor's
        sweep_table, every frame into the live map (none dropped), then the map counts as covering the table.
        """
        t0 = time.monotonic()
        self._sweeping = True  # sweep frames reach the live map through the Rig hook, none dropped
        try:
            await rt.executor.sweep_table(rig, progress=self._sweep_progress)
        finally:
            self._sweeping = False
        await self.live.drain()
        await self.live.mark_swept()
        return {"status": "done", "objects": len((self.live.map or {}).get("objects") or []),
                "duration_s": round(time.monotonic() - t0, 1)}

    async def _run(self, kind: str, text: Optional[str], target: Optional[dict] = None,
                   decide_ms: Optional[int] = None, target_xy: Optional[list] = None,
                   chosen: Optional[dict] = None, skip_current: bool = False) -> dict:
        """
        Run one grasp, reset or table scan to completion and return its summary; never raises.

        Grasp, from what is known: target (the cached frame) -> the executor picks at once; target_xy (a fresh
        live-map entry) -> AGENT1's search_and_find looks there first; neither -> it searches the table. The
        search stops at the first view whose best() accepts and the pick starts right there: no scan first.
        chosen: {"label", "colour"} of the object decided on (map path); the frame path fills it here.
        skip_current: a fresh idle frame from where the arm still is already showed it is not in view, so the
        search starts at the hint / first pose instead of looking there again (AGENT1: ~3.6 s saved).
        The summary's "label" is that recognised library name and "colour" its colour, because the executor's
        own "label" falls back to the colour (VIAM CENTERAL: a blue screwdriver came back as "label": "black").
        Retries (_retryable): up to GRASP_ATTEMPTS in all, each later one searching afresh; job_info["attempt"]
        says which try is running, and the summary's "attempts" lists every try when there was more than one.
        """
        if target is not None:
            chosen = {"label": target.get("recognised"), "colour": getattr(target.get("obj"), "colour", None)}
        tries = GRASP_ATTEMPTS if kind == "grasp" else 1
        attempts: list = []
        for n in range(1, tries + 1):
            self.job_info.update(attempt=n, attempts_max=tries)
            # skip_current only for the first try: a failed try moved the arm, so a retry looks from where it is
            summary, run_via = await self._once(kind, text, target, target_xy, skip_current and n == 1)
            # The first attempt's path names the job (frame / map / search); retries always search
            via = run_via if n == 1 else via
            if n == tries or not self._retryable(kind, summary):
                break
            attempts.append({k: summary.get(k) for k in ("status", "fail_phase", "fail_reason", "duration_s")})
            LOGGER.info(f"grasp {text!r} attempt {n}/{tries} failed ({summary.get('fail_reason')}): retrying")
            # The failed try moved things (a slipped object lands somewhere else): search again from the
            # observe pose instead of reusing the old frame or map position
            target = target_xy = None
        if attempts:
            summary["attempts"] = attempts + [{k: summary.get(k) for k in ("status", "fail_phase", "fail_reason",
                                                                           "duration_s")}]
        if kind == "grasp":
            summary.update(fast_path=via == "frame", via=via, decide_ms=decide_ms)
            # Something was picked (or found): name it by the decided library label, keep the colour apart
            if summary.get("label") is not None and chosen:
                summary.update(label=chosen.get("label") or summary["label"],
                               colour=summary.get("colour") or chosen.get("colour"))
        # job id lets a DoCommand caller (the dashboard in module mode) match this result to its request
        self.last = {"kind": kind, "text": text, "job": self.job_info.get("job"), "finished": time.time(), **summary}
        return summary

    @staticmethod
    def _retryable(kind: str, summary: dict) -> bool:
        """
        Whether another grasp attempt can fix this failure.

        WHY: the user asked for it after the green cylinder slipped out during the carry and the job just ended
        {USER 2026-10-29 "I ALSO WANT RETRY WHEN FAIL"}. Retried: status "failed" while picking or carrying
        (slipped, closed on air, the standoff check saw something else). Not retried: every other status (not
        found, ambiguous, too wide, too close, square full are answers, not accidents); a refused move or a dead
        link (the same move fails again); a failed release check (the object may already be in the square, and
        a retry would take it back out); a failed recovery (it may still be holding something).
        """
        if kind != "grasp" or summary.get("status") != "failed" or summary.get("fail_phase") == "check":
            return False
        reason = str(summary.get("fail_reason") or "")
        return not any(s in reason for s in ("MoveRefused", "link", "recovery failed", "still holding"))

    async def _once(self, kind: str, text: Optional[str], target: Optional[dict],
                    target_xy: Optional[list], skip_current: bool = False) -> tuple:
        """One grasp attempt, reset or table scan -> (summary, via); never raises (an exception becomes "error")."""
        via = "scan" if kind == "scan" else "search"
        try:
            rt = _runtime()
            rig = await self._get_rig()
            # Every frame the executor takes from here on also updates the live map (AGENT1's hook in Rig.frame)
            rig.on_map_frame = lambda rgb, depth, pose: self._on_map_frame(rig, rgb, depth, pose)
            # AGENT1: the executor reads the live map for the newest drop square and free drop spots
            rig.live_map = lambda: self.live.map if self.live is not None else None
            if kind == "grasp":
                params = inspect.signature(rt.executor.execute).parameters
                kwargs = {"progress": self._search_progress} if "progress" in params else {}
                if target is not None and "target" in params:
                    kwargs["target"], via = target, "frame"
                elif target_xy is not None and "target_xy" in params:
                    kwargs["target_xy"], via = tuple(target_xy), "map"
                elif (target is not None or target_xy is not None) and not self._no_target_warned:
                    LOGGER.warning("executor.execute lacks target=/target_xy=: using its own search instead")
                    self._no_target_warned = True
                if skip_current and target is None and "skip_current" in params:
                    kwargs["skip_current"] = True
                doc = await rt.executor.execute(rig, text, self.store, mode="demo", **kwargs)
            elif kind == "scan":
                doc = await self._scan(rt, rig)
            else:
                doc = await rt.executor.clear_square(rig, self.store)
            summary = {k: doc.get(k) for k in SUMMARY_KEYS if doc.get(k) is not None}
        except Exception as e:
            # The cause too: Rig.call's "link still down after N reconnects" hid a refused move (VIAM CENTERAL)
            cause = e.__cause__ or e.__context__
            err = f"{type(e).__name__}: {e}" + (f" <- {type(cause).__name__}: {cause}" if cause is not None else "")
            LOGGER.error(f"{kind} failed: {err}")
            summary = {"status": "error", "error": err[:300]}
            try:
                if self.rig is not None:
                    await self.rig.lift()
                    # End at the observe pose like every executor task, so the idle loop sees the table again
                    if hasattr(_runtime().executor, "back_to_observe"):
                        await _runtime().executor.back_to_observe(self.rig)
            except Exception:
                pass
        return summary, via

    async def _start(self, kind: str, text: Optional[str], wait: bool, target: Optional[dict] = None,
                     decide_ms: Optional[int] = None, target_xy: Optional[list] = None,
                     chosen: Optional[dict] = None, skip_current: bool = False) -> Dict[str, ValueTypes]:
        if self._busy():
            return {"accepted": False, "error": "busy", "running": dict(self.job_info)}
        job_id = uuid.uuid4().hex[:12]
        via = ("scan" if kind == "scan" else "frame" if target is not None else "map" if target_xy is not None
               else "search")
        self.job_info = {"job": job_id, "kind": kind, "text": text or "", "started": time.time(), "via": via,
                         "fast_path": target is not None, "target_xy": target_xy}
        self.job = asyncio.get_event_loop().create_task(
            self._run(kind, text, target, decide_ms, target_xy, chosen, skip_current))
        if wait:
            return {"accepted": True, "job": job_id, **(await self.job)}
        return {"accepted": True, "job": job_id}

    async def _grasp(self, text: str, wait: bool) -> Dict[str, ValueTypes]:
        """
        Start picking at once (user: "just search, and if found just start grasp; do NOT have an initial step"):
        in the cached idle frame -> go; a fresh live-map entry -> search there first; otherwise search the table.
        The search grasps at the first view that shows it. Answers at once only when the cached frame shows two
        equal matches, or when the object is already in the drop square.
        """
        if self._busy():
            return {"ok": False, "accepted": False, "error": "busy", "running": dict(self.job_info)}
        cold = self._not_ready()
        if cold:
            return cold
        rt = _runtime()
        await self._fresh_look()
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
        # The fresh idle frame was decided on and does not show it; the arm has not moved since (only jobs move
        # it): the search need not look from here again
        seen_miss = pick is not None and pick.get("obj") is None
        found = await self._from_map(text)
        if found is not None and not found["found"]:  # already in the drop square
            self.last = {"kind": "grasp", "text": text, "job": None, "finished": time.time(), "status": found["status"],
                         "fail_phase": "find", "fail_reason": found["reason"], "decide_ms": found["decide_ms"]}
            return {"ok": False, "accepted": False, "status": found["status"], "reason": found["reason"],
                    "decide_ms": found["decide_ms"], "via": "map"}
        if found is not None:
            res = await self._start("grasp", text, wait, target_xy=found["xy"], decide_ms=found["decide_ms"],
                                    chosen={"label": found["label"], "colour": found.get("colour")},
                                    skip_current=seen_miss)
            return {**res, "ok": bool(res.get("accepted")), "via": "map", "status_text": "searching",
                    "message": self._map_message(found) + " · looking there first"
                               + (", then searching" if found["stale"] else ""), "stale": found["stale"],
                    "target_xy": found["xy"], "label": found["label"], "map_age_s": found["map_age_s"],
                    "seen_s_ago": found["seen_s_ago"], "look_first": found["look_first"],
                    "decide_ms": found["decide_ms"]}
        res = await self._start("grasp", text, wait, skip_current=seen_miss)
        return {**res, "ok": bool(res.get("accepted")), "via": "search", "status_text": "searching",
                "message": "searching the table… (grasps at the first view that shows it)",
                "why": "not in the cached frame" if pick is not None
                else f"no fresh frame from the observe pose ({self.seen.get('reason') or self.seen.get('error') or 'no look yet'})"}

    def _status(self) -> Dict[str, ValueTypes]:
        seen = dict(self.seen)
        if seen.get("ts"):
            seen["age_s"] = round(time.time() - seen["ts"], 1)
        return {"busy": self._busy(), "running": dict(self.job_info) if self._busy() else {}, "last": dict(self.last),
                "store": "atlas" if self.store.online else "jsonl", "worker_id": self.worker_id,
                "library": dict(self.library), "warm": self.warm, "warm_ms": self.warm_ms,
                "warm_error": self.warm_error, "seen": seen, "timelapse": self._timelapse_stats(),
                "map": self.live.summary() if self.live is not None else {"live": False},
                "observations": self.store.observation_stats(), "build": _build_info()}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        if command.get("status"):
            return self._status()
        if command.get("seen_image"):
            return self._seen_image()
        if "rebuild_map" in command:
            arg = command.get("rebuild_map")
            return await self._rebuild_map(arg.get("since") if isinstance(arg, dict) else None)
        if command.get("clear_map"):
            return await self._clear_map(str(command.get("reason") or "clear_map command"))
        if command.get("scan_table"):
            cold = self._not_ready()
            return cold or await self._start("scan", None, bool(command.get("wait")))
        if command.get("table_map"):
            # Dashboard top-down view: the live map (no embeddings, each object with its seen_ts; up to
            # MAP_PERSIST_S newer than the saved one) + each object's latest sighting
            m = self.live.light() if self.live is not None else self.store.get_table_map(with_emb=False)
            return {"ok": bool(m), "map": m or {}, "sightings": self.store.list_sightings()[:40],
                    "sweep": dict(self.job_info.get("sweep") or {}) if self._busy() else {},
                    "search": dict(self.job_info.get("search") or {}) if self._busy() else {},
                    "live": self.live.summary() if self.live is not None else {"live": False},
                    "observations": self.store.observation_stats()["count"],
                    "now": time.time()}  # robot clock, so the dashboard can age objects without clock skew
        for key in ("grasp", "preview"):
            if key in command:
                text = _runtime().clean(str(command.get(key) or ""))
                if not text:
                    return {"ok": False, "accepted": False, "error": f"say what to {key}, e.g. {{\"{key}\": \"green square\"}}"}
                return await (self._grasp(text, bool(command.get("wait"))) if key == "grasp" else self._preview(text))
        if command.get("reset"):
            return await self._start("reset", None, bool(command.get("wait")))
        return {"error": "unknown command; use {\"grasp\": \"<description>\"}, {\"preview\": \"<description>\"}, "
                         "{\"reset\": true}, {\"scan_table\": true}, {\"clear_map\": true}, {\"rebuild_map\": {}}, "
                         "{\"status\": true}, "
                         "{\"seen_image\": true} or {\"table_map\": true}"}

    # ---- background loops --------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Dashboard worker panel: what this service is doing, every HEARTBEAT_S."""
        while True:
            try:
                busy = self._busy()
                phase = (self._job_label() if busy
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
        for t in self.tasks + [self._forgotten] + ([self.job] if self.job else []):
            t.cancel()
        if self.live is not None:
            await self.live.close()
