"""
Viam generic service `sorter-team:sorter:grasp-by-description`: describe an object, it goes in the blue square.

WHY: other robot builders should get our whole pipeline (perception + CLIP choice + depth-guided pick + reset
+ Atlas logging) by adding one service to their machine config, without our scripts or a laptop. Running
inside viam-server also removes the laptop <-> robot WebRTC link that dropped repeatedly during our trials:
the arm, gripper and camera are reached through viam-server's local connections.

DoCommand:
  {"grasp": "green square"}          start a pick-and-place in the background -> {"accepted": true, "job": id}
  {"grasp": "...", "wait": true}     same, but block until done -> the attempt summary
  {"reset": true}                    clear the square (pick-out, wipe fallback, verify empty)
  {"status": true}                   busy / current job / last result / store mode
One job at a time; a second request while busy returns {"accepted": false, "error": "busy"}.

Config attributes (all optional except where the defaults don't match your machine):
  arm, gripper, camera            resource names, default "arm", "gripper", "cam"; declared as dependencies
  square_colour                   colour of the tape square to drop into; only "blue" is supported today
  mongodb_uri                     Atlas connection string; else env MONGODB_URI; else local JSONL files
  consume_dashboard_commands      true -> also run commands queued from the dashboard (replaces sorter/worker.py)
  home_xy [x, y]                  observe pose of the flange in mm (default from scripts/pick_place_auto.py)
  jaw_tip_px [u, v], tip_below_cam_mm   jaw-axis calibration for your camera mount (see module/README.md)

Arm logic is AGENT1's (scripts/pick_place_auto.py Rig, sorter/executor.py); this file only adapts it.
"""
import asyncio
import time
import uuid
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
SUMMARY_KEYS = ("_id", "status", "label", "score", "height_mm", "duration_s", "fail_phase", "fail_reason")


def _runtime():
    """
    Import the arm/perception stack on first use (torch + CLIP take seconds), so viam-server can start the
    module and validate config quickly. Returns (pick_place_auto, executor, worker.clean).
    """
    from sorter import executor, worker

    return executor.ppa, executor, worker.clean


def _check_xy(attrs: dict, key: str) -> None:
    v = attrs.get(key)
    if v is not None and not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
        raise ValueError(f"'{key}' must be a list of two numbers, got {v!r}")


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
        so there is no separate reconfigure step.
        """
        from sorter.store import Store

        self = cls(config.name)
        attrs = struct_to_dict(config.attributes)
        self.attrs = attrs
        self.arm = dependencies[Arm.get_resource_name(attrs.get("arm", "arm"))]
        self.gripper = dependencies[Gripper.get_resource_name(attrs.get("gripper", "gripper"))]
        self.camera = dependencies[Camera.get_resource_name(attrs.get("camera", "cam"))]
        self.store = Store(uri=attrs.get("mongodb_uri"))
        self.worker_id = f"module:{config.name}"
        self.rig = None
        self.job: Optional[asyncio.Task] = None
        self.job_info: Dict[str, Any] = {}
        self.last: Dict[str, Any] = {}
        self.tasks = [asyncio.get_event_loop().create_task(self._heartbeat_loop())]
        if attrs.get("consume_dashboard_commands", False):
            self.tasks.append(asyncio.get_event_loop().create_task(self._command_loop()))
        LOGGER.info(f"{config.name}: store {'Atlas' if self.store.online else 'local JSONL'}, "
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
        ppa, executor, _ = _runtime()
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

    # ---- jobs --------------------------------------------------------------------------------------

    async def _run(self, kind: str, text: Optional[str]) -> dict:
        """Run one grasp or reset to completion and return its summary; never raises."""
        try:
            _, executor, _ = _runtime()
            rig = await self._get_rig()
            if kind == "grasp":
                doc = await executor.execute(rig, text, self.store, mode="demo")
            else:
                doc = await executor.clear_square(rig, self.store)
            summary = {k: doc.get(k) for k in SUMMARY_KEYS if doc.get(k) is not None}
        except Exception as e:
            LOGGER.error(f"{kind} failed: {type(e).__name__}: {e}")
            summary = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            try:
                if self.rig is not None:
                    await self.rig.lift()
            except Exception:
                pass
        # job id lets a DoCommand caller (the dashboard in module mode) match this result to its request
        self.last = {"kind": kind, "text": text, "job": self.job_info.get("job"), "finished": time.time(), **summary}
        return summary

    async def _start(self, kind: str, text: Optional[str], wait: bool) -> Dict[str, ValueTypes]:
        if self.job is not None and not self.job.done():
            return {"accepted": False, "error": "busy", "running": dict(self.job_info)}
        job_id = uuid.uuid4().hex[:12]
        self.job_info = {"job": job_id, "kind": kind, "text": text or "", "started": time.time()}
        self.job = asyncio.get_event_loop().create_task(self._run(kind, text))
        if wait:
            return {"accepted": True, "job": job_id, **(await self.job)}
        return {"accepted": True, "job": job_id}

    def _status(self) -> Dict[str, ValueTypes]:
        busy = self.job is not None and not self.job.done()
        return {"busy": busy, "running": dict(self.job_info) if busy else {}, "last": dict(self.last),
                "store": "atlas" if self.store.online else "jsonl", "worker_id": self.worker_id}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        if command.get("status"):
            return self._status()
        if "grasp" in command:
            _, _, clean = _runtime()
            text = clean(str(command.get("grasp") or ""))
            if not text:
                return {"accepted": False, "error": "say what to grasp, e.g. {\"grasp\": \"green square\"}"}
            return await self._start("grasp", text, bool(command.get("wait")))
        if command.get("reset"):
            return await self._start("reset", None, bool(command.get("wait")))
        return {"error": "unknown command; use {\"grasp\": \"<description>\"}, {\"reset\": true} or {\"status\": true}"}

    # ---- background loops --------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Dashboard worker panel: what this service is doing, every HEARTBEAT_S."""
        while True:
            try:
                busy = self.job is not None and not self.job.done()
                phase = f"{self.job_info.get('kind')}: {self.job_info.get('text') or 'square'}" if busy else "idle"
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
                if self.job is None or self.job.done():
                    cmd = self.store.next_command()
                    if cmd is not None:
                        text = (cmd.get("text") or "").strip()
                        if text == "__reset__":
                            res = await self._start("reset", None, wait=True)
                        else:
                            _, _, clean = _runtime()
                            res = await self._start("grasp", clean(text), wait=True)
                        self.store.finish_command(cmd["_id"], res.get("status", "failed"), res.get("_id"), res.get("label"))
                        continue
            except Exception as e:
                LOGGER.warning(f"command loop: {type(e).__name__}: {e}")
            await asyncio.sleep(1.0)

    async def close(self) -> None:
        for t in self.tasks + ([self.job] if self.job else []):
            t.cancel()
