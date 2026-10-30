"""
The dashboard's robot link (dashboard/camera.py ModuleBridge) on a fake module, no robot: at most 2 calls a second
including commands, 3 s timeout, waits longer after each failure, the picture only when newer, an older module's
three commands in turn. {VIAM CENTERAL 2026-10-30 "DASHBOARD 对 MODULE 的请求每秒 ≤ 2 次，超时 3 S，失败了不重试刷屏"}
"""
import asyncio
import base64
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.camera import ModuleBridge  # noqa: E402

JPEG = base64.b64encode(b"\xff\xd8 not really a jpeg \xff\xd9").decode()


class FakeModule:
    def __init__(self, legacy=False):
        self.legacy, self.calls, self.fail, self.frame_ts, self.hang = legacy, [], False, 100.0, False

    async def do_command(self, command):
        self.calls.append((time.monotonic(), command))
        if self.hang:
            await asyncio.sleep(10)
        if self.fail:
            raise ConnectionError("link down")
        if "dashboard" in command:
            if self.legacy:
                return {"error": "unknown command; use ..."}
            since = command["dashboard"]["since"]
            frame = {"ts": self.frame_ts} if since >= self.frame_ts else {"ts": self.frame_ts, "jpeg_b64": JPEG, "w": 640, "h": 360,
                                                                            "boxes": [], "source": "idle"}
            return {"status": {"busy": False}, "map": {"map": {"objects": []}, "now": 1.0}, "arm": {"x": 250.0}, "estop": False,
                    "frame": frame}
        if command.get("status"):
            return {"busy": False, "store": "jsonl"}
        if command.get("seen_image"):
            return {"ok": True, "ts": self.frame_ts, "jpeg_b64": JPEG, "objects": [{"bbox": [1, 2, 3, 4], "label": "tape", "confidence": 0.5}],
                    "intruders": []}
        if command.get("table_map"):
            return {"ok": True, "map": {"objects": []}, "now": 1.0}
        if "grasp" in command:
            return {"accepted": True, "job": "j1"}
        return {"error": "unknown command"}


async def run_for(bridge, svc, seconds):
    task = asyncio.create_task(bridge.run(None, svc))
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, ConnectionError) as e:
        return e
    return None


async def main():
    out = Path(tempfile.mkdtemp())
    svc, bridge = FakeModule(), ModuleBridge(out, "sorter")
    (out / "module_req" / "1.json").write_text(json.dumps({"command": {"grasp": "tape"}}))
    await run_for(bridge, svc, 4.0)
    rate = len(svc.calls) / 4.0
    assert rate <= 2.05, rate
    assert svc.calls[0][1] == {"grasp": "tape"} and json.loads((out / "module_res" / "1.json").read_text())["accepted"]
    sinces = [c["dashboard"]["since"] for _, c in svc.calls[1:]]
    assert sinces[0] == 0.0 and all(s == 100.0 for s in sinces[1:]), sinces
    st, fr = json.loads((out / "module_status.json").read_text()), json.loads((out / "module_frame.json").read_text())
    assert st["ok"] and st["arm"] == {"x": 250.0} and st["estop"] is False and fr["ts"] == 100.0 and "jpeg_b64" not in fr
    assert (out / "module_frame.jpg").read_bytes().startswith(b"\xff\xd8")
    print(f"1. {rate:.2f} calls/s with a command in the first tick; the picture sent once, then since={sinces[-1]:.0f}")

    svc.calls.clear()
    svc.fail = True
    err = await run_for(bridge, svc, 6.0)
    gaps = [b[0] - a[0] for a, b in zip(svc.calls, svc.calls[1:])]
    assert len(svc.calls) <= 5 and all(g2 > g1 for g1, g2 in zip(gaps, gaps[1:])), gaps
    assert isinstance(err, ConnectionError) or len(svc.calls) < 5
    print(f"2. failing module: {len(svc.calls)} calls in 6 s, gaps {[round(g, 1) for g in gaps]} s growing, then a reconnect")

    svc.fail, svc.hang = False, True
    svc.calls.clear()
    t0 = time.monotonic()
    await run_for(bridge, svc, 3.5)
    assert len(svc.calls) == 1 and time.monotonic() - t0 >= 3.4
    print("3. a call that hangs is given up after 3 s")

    old, legacy = FakeModule(legacy=True), ModuleBridge(Path(tempfile.mkdtemp()), "sorter")
    await run_for(legacy, old, 3.2)
    kinds = [next(iter(c)) for _, c in old.calls]
    assert kinds[:4] == ["dashboard", "status", "seen_image", "table_map"] and len(kinds) <= 7, kinds
    meta = json.loads((legacy.out / "module_frame.json").read_text())
    assert meta["boxes"][0]["label"] == "tape" and json.loads((legacy.out / "module_status.json").read_text())["status"]["store"] == "jsonl"
    print(f"4. an older module: {kinds}: its three commands in turn, still <= 2 calls/s")


asyncio.run(main())
print("test_dashboard_link OK")
