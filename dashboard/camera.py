"""
Live camera feed for the dashboard: read-only get_images about once a second, with detected-object boxes.

WHY: the dashboard must show what the arm sees without ever commanding the arm, and must survive the
laptop <-> robot WebRTC link dropping (it did twice in ~10 minutes). This class runs inside
dashboard/camera_worker.py, a separate process, because Viam's native WebRTC layer once aborted the whole
process during a reconnect; app.py restarts the worker when that happens.
{DASH.LOG 2026-10-29 "MEMORY ALLOCATION OF 8302792106237310565 BYTES FAILED" AFTER "LOST CONNECTION TO MACHINE. ATTEMPTING TO RECONNECT"}
[CONFIDENCE: CONFIRMED 90% — observed once on the laptop; process exited 134 (SIGABRT)]

Boxes come from AGENT3's sorter.perception when it is importable and depth is aligned to colour.
Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY, VIAM_CAMERA (default "cam")
"""
import asyncio
import importlib
import io
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PERIOD_S = 1.0
PERCEPTION_RETRY_S = 30.0  # perception.py is still being written; re-try the import this often
FRAME_TIMEOUT_S = 10.0
DEPTH_EVERY_S = 5.0  # boxes refresh this often; between refreshes the last boxes are drawn on new colour frames
COLOR_SOURCE = "color"  # get_images source names on sfhack4: "color", "depth"
# Fields clip_embed.rank and the dashboard read from each object; masks are too big to hand over
OBJ_FIELDS = ("id", "bbox", "centroid_px", "yaw_deg", "length_px", "width_px", "height_mm", "colour")


def decode(images) -> tuple:
    """
    (rgb array, depth mm array or None) from a get_images() result.

    Depth arrives as Viam's DEPTHMAP format: 8-byte magic, uint64 width, uint64 height, big-endian uint16 mm.
    {SCRIPTS/PICK_PLACE_AUTO.PY "DEPTH ARRIVES AS VIAM'S DEPTHMAP FORMAT: 8-BYTE MAGIC, UINT64 WIDTH, UINT64 HEIGHT"}
    [CONFIDENCE: CONFIRMED 90% — decoded this way on sfhack4 by AGENT1]
    """
    rgb, depth = None, None
    for im in images:
        if im.mime_type == "image/jpeg":
            rgb = np.asarray(Image.open(io.BytesIO(im.data)).convert("RGB"))
        elif "dep" in im.mime_type and im.data[:8] == b"DEPTHMAP":
            w = int.from_bytes(im.data[8:16], "big")
            h = int.from_bytes(im.data[16:24], "big")
            depth = np.frombuffer(im.data[24:], dtype=">u2").reshape(h, w).astype(float)
    return rgb, depth


def _write_atomic(path: Path, data: bytes) -> None:
    """Write then rename, so the web server never reads a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class CameraFeed:
    """Holds the latest annotated JPEG, detected objects and connection status."""

    def __init__(self):
        self.jpeg = None
        self.frame_ts = 0.0
        self.connected = False
        self.error = None
        self.objects = []  # perception Obj list from the latest frame: the only things the arm may target
        self.excluded = []  # [(Obj, reason)]: cut by the image edge or outside the workspace, drawn grey
        self.intruders = []  # [{"bbox", "reason"}]: e.g. a hand in the workspace (the arm pauses), drawn red
        self.table_depth = None
        self.perception = None
        self.perception_error = "not loaded yet"
        self._perception_tried = 0.0
        self.highlight = None  # {"id", "text", "score", "until"} from a text preview
        self._dumped_ts = 0.0

    def status(self) -> dict:
        return {"connected": self.connected, "error": self.error,
                "frame_age_s": round(time.time() - self.frame_ts, 1) if self.frame_ts else None,
                "perception": self.perception is not None, "perception_error": self.perception_error,
                "objects": len(self.objects), "excluded": len(self.excluded), "intruders": len(self.intruders)}

    def _load_perception(self) -> None:
        if self.perception is not None or time.time() - self._perception_tried < PERCEPTION_RETRY_S:
            return
        self._perception_tried = time.time()
        try:
            self.perception = importlib.import_module("sorter.perception")
            self.perception_error = None
        except Exception as e:
            self.perception_error = f"{type(e).__name__}: {e}"[:200]

    def _segment(self, rgb, depth) -> None:
        """
        Run perception: segment_full() for targets + excluded + intruders, else plain segment() (targets only).

        Depth/colour alignment is perception's job (it checks once and falls back), so no shape guard here.
        {AGENT3 53E4B75 "SEGMENT_FULL().EXCLUDED AS (OBJ, REASON) WITH NEGATIVE IDS"}
        [CONFIDENCE: CONFIRMED 90% — Segmentation dataclass read in sorter/perception.py]
        """
        self._load_perception()
        if self.perception is None or depth is None:
            self.objects, self.excluded, self.intruders = [], [], []
            return
        try:
            if hasattr(self.perception, "segment_full"):
                seg = self.perception.segment_full(rgb, depth)
                self.table_depth = seg.table_depth_mm
                self.objects, self.excluded, self.intruders = list(seg.objects), list(seg.excluded), list(seg.intruders)
            else:
                res = self.perception.segment(rgb, depth)
                self.table_depth, self.objects = res[0], list(res[1])
                self.excluded, self.intruders = [], []
            self.perception_error = None
        except Exception as e:
            self.perception_error = f"segment failed: {type(e).__name__}: {e}"[:200]
            self.objects, self.excluded, self.intruders = [], [], []

    def _render(self, rgb) -> bytes:
        """
        Colour frame with boxes: excluded objects grey with the reason, intruders red, targets yellow, and the
        previewed target thick and green. Drawn in that order so targets stay on top.
        """
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)

        def tag(box, text, fill, outline, width):
            u0, v0, u1, v1 = [int(x) for x in box]
            draw.rectangle([u0, v0, u1, v1], outline=outline, width=width)
            draw.rectangle([u0, max(v0 - 22, 0), u0 + 9 * len(text), max(v0 - 22, 0) + 22], fill=fill)
            draw.text((u0 + 4, max(v0 - 20, 2)), text, fill=(0, 0, 0))

        for o, reason in self.excluded:
            short = "edge" if "edge" in reason else "outside" if "outside" in reason else reason[:16]
            tag(o.bbox, f"ignored: {short}", (190, 190, 190), (150, 150, 150), 2)
        for it in self.intruders:
            tag(it["bbox"], "hand? arm paused", (255, 110, 100), (230, 50, 40), 4)
        hl = self.highlight if self.highlight and self.highlight["until"] > time.time() else None
        for o in self.objects:
            u0, v0, u1, v1 = [int(x) for x in o.bbox]
            target = hl is not None and o.id == hl["id"]
            color = (40, 220, 90) if target else (255, 200, 0)
            draw.rectangle([u0, v0, u1, v1], outline=color, width=6 if target else 3)
            label = f"{hl['text']} {hl['score']:.2f}" if target else f"#{o.id} {getattr(o, 'height_mm', 0):.0f}mm"
            draw.rectangle([u0, max(v0 - 22, 0), u0 + 9 * len(label), max(v0 - 22, 0) + 22], fill=color)
            draw.text((u0 + 4, max(v0 - 20, 2)), label, fill=(0, 0, 0))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=75)
        return out.getvalue()

    def read_highlight(self, out: Path) -> None:
        """Pick up a preview highlight written by the web server (highlight.json)."""
        try:
            self.highlight = json.loads((out / "highlight.json").read_text())
        except (FileNotFoundError, ValueError):
            pass

    def dump(self, out: Path) -> None:
        """
        Mirror state into files for the web server: frame.jpg, state.json, objs.pkl (crops for ranking).

        state.json is rewritten every call so its `written` time doubles as a heartbeat.
        """
        state = {**self.status(), "written": time.time(), "table_depth": self.table_depth,
                 "objects": [{k: _plain(getattr(o, k, None)) for k in OBJ_FIELDS if k != "colour"}
                             for o in self.objects]}
        if self.frame_ts and self.frame_ts != self._dumped_ts:
            _write_atomic(out / "frame.jpg", self.jpeg)
            objs = [{**{k: getattr(o, k, None) for k in OBJ_FIELDS}, "crop": getattr(o, "crop", None)}
                    for o in self.objects]
            _write_atomic(out / "objs.pkl", pickle.dumps(objs))
            self._dumped_ts = self.frame_ts
        _write_atomic(out / "state.json", json.dumps(state).encode())

    async def _frames(self, robot) -> None:
        """Camera loop on an open connection; raises on any error so the caller reconnects."""
        from viam.components.camera import Camera

        cam = Camera.from_robot(robot, os.environ.get("VIAM_CAMERA", "cam"))
        last_depth = 0.0
        while True:
            self._load_perception()
            # Depth (~1.8 MB) only every DEPTH_EVERY_S for boxes; colour alone (~70 KB) in between, since
            # the laptop link timed out even on single colour+depth frames
            want_depth = self.perception is not None and time.time() - last_depth >= DEPTH_EVERY_S
            kw = {} if want_depth else {"filter_source_names": [COLOR_SOURCE]}
            res = await asyncio.wait_for(cam.get_images(**kw), timeout=FRAME_TIMEOUT_S)
            rgb, depth = decode(res[0] if isinstance(res, tuple) else res)
            if rgb is not None:
                if depth is not None:
                    last_depth = time.time()
                    await asyncio.to_thread(self._segment, rgb, depth)
                self.jpeg = await asyncio.to_thread(self._render, rgb)
                self.frame_ts = time.time()
            await asyncio.sleep(PERIOD_S)

    async def run(self, camera_on: bool = True, bridge=None) -> None:
        """
        Connect, run the camera loop and/or the module bridge on that one connection, reconnect with backoff
        when either fails. This process never moves the arm itself; the bridge only forwards dashboard
        commands to our module's DoCommand.
        """
        from viam.robot.client import RobotClient

        backoff = 2.0
        while True:
            env = [os.environ.get(k) for k in ("VIAM_MACHINE_ADDRESS", "VIAM_API_KEY_ID", "VIAM_API_KEY")]
            if not all(env):
                self.error = "VIAM_MACHINE_ADDRESS / VIAM_API_KEY_ID / VIAM_API_KEY not set"
                await asyncio.sleep(5)
                continue
            robot = None
            try:
                # Our loop is the only reconnect logic: the SDK's own checker raced it before the native abort.
                # Sessions off: camera reads need none, and a module job must not be cancelled when this
                # laptop link drops (the module runs it on the robot).
                opts = RobotClient.Options.with_api_key(api_key=env[2], api_key_id=env[1], check_connection_interval=0,
                                                        attempt_reconnect_interval=0, disable_sessions=True)
                robot = await asyncio.wait_for(RobotClient.at_address(env[0], opts), timeout=30)
                self.connected, self.error, backoff = True, None, 2.0
                loops = ([self._frames(robot)] if camera_on else []) + ([bridge.run(robot)] if bridge else [])
                await _first_failure(loops)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.connected = False
                # TimeoutError has an empty message; say what timed out
                detail = str(e) or f"no answer within {FRAME_TIMEOUT_S:.0f} s, reconnecting"
                self.error = f"{type(e).__name__}: {detail}"[:200]
            finally:
                if robot is not None:
                    try:
                        await robot.close()
                    except Exception:
                        pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def _first_failure(coros: list) -> None:
    """Run loops together; when one raises, cancel the rest and re-raise (the caller reconnects)."""
    if not coros:
        await asyncio.Event().wait()  # nothing to run; keep the connection open
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            t.result()
    finally:
        for t in tasks:
            t.cancel()


class ModuleBridge:
    """
    Dashboard <-> our Viam module (generic service) over the SDK, without the Store command queue.

    WHY: without MONGODB_URI the laptop and the robot cannot share the queue (offline JSONL is per machine),
    but the module's DoCommand works over the existing robot connection. Runs in this subprocess so an SDK
    abort cannot take the web server down.

    File protocol in <out>: the web server writes module_req/<id>.json {"command": {...}}; this loop calls
    DoCommand and writes module_res/<id>.json; module_status.json holds the latest {"status": true} answer.
    """

    STATUS_EVERY_S = 1.5
    CMD_TIMEOUT_S = 15.0
    MAX_FAILURES = 5  # consecutive failed calls -> raise, so the connection is rebuilt

    def __init__(self, out: Path, name: str):
        self.out, self.name = out, name
        (out / "module_req").mkdir(parents=True, exist_ok=True)
        (out / "module_res").mkdir(parents=True, exist_ok=True)

    async def _call(self, svc, command: dict, timeout: float) -> dict:
        return _jsonable(await asyncio.wait_for(svc.do_command(command), timeout=timeout))

    async def run(self, robot) -> None:
        from viam.services.generic import Generic

        svc = Generic.from_robot(robot, self.name)
        failures, last_poll = 0, 0.0
        while True:
            for req_path in sorted((self.out / "module_req").glob("*.json")):
                try:
                    req = json.loads(req_path.read_text())
                except (FileNotFoundError, ValueError):
                    continue
                try:
                    res, failures = await self._call(svc, req["command"], self.CMD_TIMEOUT_S), 0
                except Exception as e:
                    failures += 1
                    res = {"accepted": False, "error": f"{type(e).__name__}: {e or 'no answer'}"[:300]}
                _write_atomic(self.out / "module_res" / req_path.name, json.dumps(res, default=str).encode())
                req_path.unlink(missing_ok=True)
            if time.time() - last_poll >= self.STATUS_EVERY_S:
                last_poll = time.time()
                try:
                    state, failures = {"ok": True, "status": await self._call(svc, {"status": True}, 10.0)}, 0
                except Exception as e:
                    failures += 1
                    state = {"ok": False, "error": f"{type(e).__name__}: {e or 'no answer'}"[:300]}
                state["ts"] = time.time()
                _write_atomic(self.out / "module_status.json", json.dumps(state, default=str).encode())
            if failures >= self.MAX_FAILURES:
                raise ConnectionError(f"module '{self.name}' failed {failures} calls in a row")
            await asyncio.sleep(0.2)


def _jsonable(v):
    """DoCommand results (Mapping / Sequence / numbers) -> plain JSON-serialisable Python."""
    if isinstance(v, dict) or hasattr(v, "items"):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _plain(v):
    """numpy scalars/tuples -> JSON-safe Python values."""
    if isinstance(v, (tuple, list, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v
