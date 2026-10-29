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
        self.objects = []  # perception Obj list from the latest frame
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
                "objects": len(self.objects)}

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
        """Run perception if possible; accepts segment() returning (table, objs) or an object with .objects."""
        self._load_perception()
        if self.perception is None or depth is None:
            self.objects = []
            return
        if depth.shape[:2] != rgb.shape[:2]:
            # Boxes would land in the wrong place on an unaligned depth grid
            self.perception_error = "depth not aligned to colour (align_color_depth off)"
            self.objects = []
            return
        try:
            res = self.perception.segment(rgb, depth)
            if isinstance(res, tuple):
                self.table_depth, self.objects = res[0], list(res[1])
            else:
                self.table_depth, self.objects = getattr(res, "table_depth", None), list(res.objects)
            self.perception_error = None
        except Exception as e:
            self.perception_error = f"segment failed: {type(e).__name__}: {e}"[:200]
            self.objects = []

    def _render(self, rgb) -> bytes:
        """Colour frame with object boxes; the previewed target is drawn thick and green."""
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
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

    async def run(self) -> None:
        """Connect, stream frames, reconnect with backoff on any error. Never sends motion commands."""
        from viam.components.camera import Camera
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
                # Sessions are off because this client only reads the camera.
                opts = RobotClient.Options.with_api_key(api_key=env[2], api_key_id=env[1], check_connection_interval=0,
                                                        attempt_reconnect_interval=0, disable_sessions=True)
                robot = await asyncio.wait_for(RobotClient.at_address(env[0], opts), timeout=30)
                cam = Camera.from_robot(robot, os.environ.get("VIAM_CAMERA", "cam"))
                self.connected, self.error, backoff = True, None, 2.0
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
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.connected = False
                # TimeoutError has an empty message; say what timed out
                detail = str(e) or f"no frame within {FRAME_TIMEOUT_S:.0f} s, reconnecting"
                self.error = f"{type(e).__name__}: {detail}"[:200]
            finally:
                if robot is not None:
                    try:
                        await robot.close()
                    except Exception:
                        pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _plain(v):
    """numpy scalars/tuples -> JSON-safe Python values."""
    if isinstance(v, (tuple, list, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v
