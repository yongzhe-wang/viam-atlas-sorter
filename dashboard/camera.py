"""
Robot link for the dashboard: one connection to the machine and at most two DoCommands a second to our module, mirrored
into files the web server (dashboard/app.py) reads, however many browsers are open.

WHY: the dashboard must never slow the module {VIAM CENTERAL 2026-10-30 "DASHBOARD 对 MODULE 的请求每秒 ≤ 2 次，超时 3 S，
失败了不重试刷屏"}: each tick sends one call, a pending command first, else {"dashboard": {"since": ts}} (status, live map,
arm pose and the picture if newer: module/dashboard_view.py); a failed call waits longer before the next. An older module
without {"dashboard"} gets its three commands in turn, one per tick.
This runs in dashboard/camera_worker.py, a separate process, because Viam's native WebRTC layer once aborted the whole
process during a reconnect; app.py restarts the worker when that happens.
{DASH.LOG 2026-10-29 "MEMORY ALLOCATION OF 8302792106237310565 BYTES FAILED" AFTER "LOST CONNECTION TO MACHINE. ATTEMPTING TO RECONNECT"}
[CONFIDENCE: CONFIRMED 90% — observed once on the laptop; process exited 134 (SIGABRT)]

Files in <out>: module_status.json {ok, ts, status, arm, estop, phase}; module_table_map.json {map, search, sweep, live,
now (robot clock), fetched_at}; module_frame.jpg + module_frame.json {ts, got, w, h, source, pose, boxes, view};
link.json {connected, error, written}; commands: module_req/<id>.json -> module_res/<id>.json.
Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY.
"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path


def _write_atomic(path: Path, data: bytes) -> None:
    """Write then rename, so the web server never reads a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, obj) -> None:
    _write_atomic(path, json.dumps(obj, default=str).encode())


class Link:
    """The connection to the machine: connect, run the bridge on it, reconnect with backoff when it fails."""

    CONNECT_TIMEOUT_S = 30.0

    def __init__(self):
        self.connected, self.error = False, None

    def dump(self, out: Path) -> None:
        """link.json, rewritten every call so its `written` time doubles as this process's heartbeat."""
        _write_json(out / "link.json", {"connected": self.connected, "error": self.error, "written": time.time()})

    async def run(self, bridge) -> None:
        """This process never moves the arm itself; the bridge only forwards dashboard commands to our module."""
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
                # Sessions off: a module job must not be cancelled when this laptop link drops (it runs on the robot).
                opts = RobotClient.Options.with_api_key(api_key=env[2], api_key_id=env[1], check_connection_interval=0,
                                                        attempt_reconnect_interval=0, disable_sessions=True)
                robot = await asyncio.wait_for(RobotClient.at_address(env[0], opts), timeout=self.CONNECT_TIMEOUT_S)
                self.connected, self.error, backoff = True, None, 2.0
                await bridge.run(robot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.connected = False
                # TimeoutError has an empty message; say what timed out
                self.error = f"{type(e).__name__}: {e or 'no answer, reconnecting'}"[:200]
            finally:
                if robot is not None:
                    try:
                        await robot.close()
                    except Exception:
                        pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


class ModuleBridge:
    """Dashboard <-> our Viam module (generic service) over the SDK: one call per tick, see the module docstring."""

    POLL_EVERY_S = 0.5  # one call per tick, a command or the poll: at most 2 calls a second to the module
    CALL_TIMEOUT_S = 3.0
    CMD_TIMEOUT_S = 15.0  # a command (grasp, preview, teach) may take a fresh look before it answers
    MAX_FAILURES = 5  # failed calls in a row -> raise, so the connection is rebuilt
    LEGACY = ({"status": True}, {"seen_image": True}, {"table_map": True})

    def __init__(self, out: Path, name: str):
        self.out, self.name = out, name
        (out / "module_req").mkdir(parents=True, exist_ok=True)
        (out / "module_res").mkdir(parents=True, exist_ok=True)
        self.since = 0.0  # ts of the newest picture this side has: the module sends a JPEG only when it has a newer one
        self.legacy = False  # the module answered "unknown command" to {"dashboard"}
        self.turn = 0
        self.calls = 0

    async def _call(self, svc, command: dict, timeout: float) -> dict:
        self.calls += 1
        return _jsonable(await asyncio.wait_for(svc.do_command(command), timeout=timeout))

    def _frame(self, f: dict, got: float) -> None:
        """A new picture: the JPEG, then its meta (the web server keys on the meta file)."""
        _write_atomic(self.out / "module_frame.jpg", base64.b64decode(f["jpeg_b64"]))
        _write_json(self.out / "module_frame.json", {**{k: v for k, v in f.items() if k != "jpeg_b64"}, "got": got})
        self.since = float(f["ts"])

    async def _poll(self, svc) -> None:
        """One {"dashboard"} call into the three files; an older module: the next of its three commands."""
        now = time.time()
        if not self.legacy:
            res = await self._call(svc, {"dashboard": {"since": self.since}}, self.CALL_TIMEOUT_S)
            if "unknown command" in str(res.get("error", "")):
                self.legacy = True
                return
            _write_json(self.out / "module_status.json", {"ok": True, "ts": now, "status": res.get("status") or {},
                                                          **{k: res.get(k) for k in ("arm", "estop", "phase")}})
            _write_json(self.out / "module_table_map.json", {**(res.get("map") or {}), "fetched_at": now})
            if (res.get("frame") or {}).get("jpeg_b64"):
                self._frame(res["frame"], now)
            return
        command = self.LEGACY[self.turn % len(self.LEGACY)]
        self.turn += 1
        res = await self._call(svc, command, self.CALL_TIMEOUT_S)
        if command.get("status"):
            _write_json(self.out / "module_status.json", {"ok": True, "ts": now, "status": res})
        elif command.get("table_map"):
            _write_json(self.out / "module_table_map.json", {**res, "fetched_at": now})
        elif res.get("ok") and res.get("ts") != self.since:
            boxes = ([{"bbox": o["bbox"], "label": o.get("label"), "names": [o["label"]] if o.get("label") else [],
                       "score": o.get("confidence"), "kind": "object"} for o in res.get("objects") or []]
                     + [{"bbox": i["bbox"], "kind": "intruder"} for i in res.get("intruders") or []])
            self._frame({"ts": res["ts"], "jpeg_b64": res["jpeg_b64"], "source": "idle", "boxes": boxes}, now)

    async def _command(self, svc, req_path: Path) -> None:
        """A dashboard command -> its answer file; a failed call answers with the error, then counts as a failure."""
        try:
            req = json.loads(req_path.read_text())
        except (FileNotFoundError, ValueError):
            return
        try:
            res = await self._call(svc, req["command"], self.CMD_TIMEOUT_S)
        except Exception as e:
            _write_json(self.out / "module_res" / req_path.name, {"accepted": False, "error": f"{type(e).__name__}: {e or 'no answer'}"[:300]})
            raise
        finally:
            req_path.unlink(missing_ok=True)
        _write_json(self.out / "module_res" / req_path.name, res)

    async def run(self, robot, svc=None) -> None:
        if svc is None:
            from viam.services.generic import Generic

            svc = Generic.from_robot(robot, self.name)
        failures = 0
        while True:
            t0 = time.monotonic()
            req = min((self.out / "module_req").glob("*.json"), default=None)
            try:
                await (self._command(svc, req) if req is not None else self._poll(svc))
                failures = 0
            except Exception:
                failures += 1
                if failures >= self.MAX_FAILURES:
                    raise ConnectionError(f"module '{self.name}' failed {failures} calls in a row")
                # No retry storm: each failure in a row waits longer before the next call
                await asyncio.sleep(min(self.POLL_EVERY_S * 2 ** failures, 5.0))
            await asyncio.sleep(max(0.0, self.POLL_EVERY_S - (time.monotonic() - t0)))


def _jsonable(v):
    """DoCommand results (Mapping / Sequence / numbers) -> plain JSON-serialisable Python."""
    if isinstance(v, dict) or hasattr(v, "items"):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
