"""
Demo dashboard: three live views of the robot through our Viam module: the command box, what the arm's camera sees,
and the module's own map of the table.

WHY: the user: "for the dashboard i just want three things, one is the command stuff, one is the view of the arm and
one is the map of the objects it internal record", then "all dynamic" and "also optimize the dashboard" (VIAM CENTERAL
2026-10-30). The robot link runs in a child process (dashboard/camera_worker.py, restarted if the Viam SDK aborts) that
asks the module one {"dashboard"} DoCommand at most twice a second however many browsers are open; this server only
reads the files it writes. The page asks /api/live once a second (status, command, map, all in one) and /api/frame.jpg
for the picture, which answers 304 until there is a new one. The page never moves the arm: commands go to the module.

Run from the repo root (laptop): python dashboard/app.py
Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY; DASH_MODULE_NAME (default "sorter"); SORTER_DATA (where the
link files go, default <repo>/data); HOST / PORT; DASH_LINK=0 starts no robot link (something else writes the files,
e.g. a simulator).
"""
import asyncio
import json
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CAM_DIR = Path(os.environ.get("SORTER_DATA") or ROOT / "data") / "camera"
MODULE_NAME = os.environ.get("DASH_MODULE_NAME", "sorter")
RESET_COMMAND = "__reset__"  # Reset button: the module clears the drop target
SCAN_COMMAND = "__scan__"  # "Scan table": the module looks over the whole table into its live map
MAX_TEXT = 200
LINK_STALE_S = 4.0  # the module answers every 0.5 s; older than this = link or module down
MODULE_CALL_TIMEOUT_S = 25.0  # one command round trip through the robot-link process
MAP_HIGHLIGHT_S = 20.0
FRAME_WAIT_S = 2.0  # a page that has the newest picture waits this long for the next one before a 304
REACH_MM = (120, 650)  # the half ring the arm code lets the arm reach, drawn on the map

camproc = {"pid": None, "restarts": 0, "last_exit": None}
module_cmds = deque(maxlen=5)  # commands sent this session, newest first; the page shows the newest
_map_highlight = {}  # {"text", "label", "xy", "until"}: what the last preview / grasp resolved to
_frame_cache = {"mtime": None, "data": None, "etag": None, "meta": None}  # module_frame.* read once per new picture


async def supervise_link() -> None:
    """Keep dashboard/camera_worker.py running; restart with backoff when it exits (reset after a minute up)."""
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DASH_MODULE_NAME": MODULE_NAME}
    backoff = 1.0
    while True:
        started = time.time()
        with open(CAM_DIR / "worker.log", "wb") as log:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "dashboard.camera_worker", str(CAM_DIR), cwd=str(ROOT), stdout=log, stderr=log,
                env=env)
            camproc["pid"] = proc.pid
            try:
                rc = await proc.wait()
            except asyncio.CancelledError:
                proc.terminate()
                raise
        camproc.update(pid=None, restarts=camproc["restarts"] + 1, last_exit=rc)
        backoff = 1.0 if time.time() - started > 60 else min(backoff * 2, 15.0)
        await asyncio.sleep(backoff)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(supervise_link()) if os.environ.get("DASH_LINK", "1") != "0" else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Sorter dashboard", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


class TeachIn(BaseModel):
    name: str
    alias_of: Optional[str] = None
    point: Optional[List[float]] = None  # where the object was clicked, 0-1 of the picture


def _clean(text: str) -> str:
    text = " ".join(text.split())
    if not text or len(text) > MAX_TEXT:
        raise HTTPException(400, f"text must be 1-{MAX_TEXT} characters")
    return text


def _read(name: str) -> dict:
    try:
        return json.loads((CAM_DIR / name).read_text())
    except (FileNotFoundError, ValueError):
        return {}


async def module_call(command: dict) -> dict:
    """One DoCommand through the robot-link process (file request/response); 504 if it does not answer."""
    rid = f"{time.time():.3f}-{uuid.uuid4().hex[:8]}"
    req, res = CAM_DIR / "module_req" / f"{rid}.json", CAM_DIR / "module_res" / f"{rid}.json"
    req.parent.mkdir(parents=True, exist_ok=True)
    tmp = req.with_suffix(".tmp")
    tmp.write_text(json.dumps({"command": command}))
    tmp.replace(req)  # the bridge only picks up complete *.json files
    t0 = time.time()
    while time.time() - t0 < MODULE_CALL_TIMEOUT_S:
        if res.exists():
            out = json.loads(res.read_text())
            res.unlink(missing_ok=True)
            return out
        await asyncio.sleep(0.1)
    req.unlink(missing_ok=True)
    raise HTTPException(504, "the robot link did not answer; is the module running and the machine online?")


def _set_map_highlight(res: dict, text: str) -> None:
    """Remember what a module answer pointed at (map xy or recognised label), to ring it on the map and the picture."""
    target = res.get("target") or {}
    xy, label = res.get("target_xy"), res.get("label") or target.get("label")
    # Every new answer replaces the ring: a miss must not leave the previous hit ringed
    _map_highlight.clear()
    if res.get("ok") is not False and (xy or label):
        _map_highlight.update(text=text, label=label, xy=xy, until=time.time() + MAP_HIGHLIGHT_S)


def _sync_command(s: dict) -> Optional[dict]:
    """
    The newest command, updated from the module's status: running while its job is the module's current job (a list
    with its progress and items so far), then the module's last result once that names the same job.
    """
    if not module_cmds:
        return None
    c = module_cmds[0]
    running, last = s.get("running") or {}, s.get("last") or {}
    if c["status"] == "running":
        if s.get("busy") and running.get("job") == c.get("job"):
            c.update(progress=running.get("progress"), items=running.get("items") or c.get("items"))
        elif last.get("job") and last.get("job") == c.get("job"):
            c.update(status=last.get("status") or "done", fail_reason=last.get("fail_reason") or last.get("error"),
                     duration_s=last.get("duration_s"), items=last.get("items"), stopped_by=last.get("stopped_by"),
                     progress=None, label=None if last.get("items") else last.get("label"))
    return c


def _slim_map(d: dict) -> Optional[dict]:
    """The module's live map for drawing: ages on the robot's clock, no observation detail."""
    m = d.get("map")
    if not m:
        return None
    robot_now = (d.get("now") or time.time()) + (time.time() - (d.get("fetched_at") or time.time()))
    keep = ("id", "label", "names", "xy", "partial", "in_square", "superseded", "score", "colour")
    objs = [{**{k: o.get(k) for k in keep}, "age_s": round(robot_now - (o.get("seen_ts") or m.get("ts") or robot_now), 1)}
            for o in m.get("objects") or [] if o.get("xy")]
    return {"objects": objs, "square": m.get("square"), "bucket": m.get("bucket"),
            "age_s": round(robot_now - (m.get("ts") or robot_now), 1),
            "cleared_s_ago": round(robot_now - m["cleared_ts"], 1) if m.get("cleared_ts") else None}


@app.get("/")
def index():
    # no-cache: after a dashboard restart with new code, a plain reload must show it
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/live")
def live():
    """Everything but the picture, in one answer the page asks for once a second."""
    now = time.time()
    st, mp, fr, link = _read("module_status.json"), _read("module_table_map.json"), _read("module_frame.json"), _read("link.json")
    age = now - st["ts"] if st.get("ts") else None
    s = st.get("status") or {}
    up = age is not None and age <= LINK_STALE_S and bool(st.get("ok"))
    running = s.get("running") or {}
    seen = s.get("seen") or {}
    hl = dict(_map_highlight) if _map_highlight.get("until", 0) > now else None
    lib = s.get("library") or {}
    return {
        "now": now,
        "link": {"up": up, "age_s": round(age, 1) if age is not None else None,
                 "error": None if up else link.get("error") or ("robot link starting" if age is None else "no answer from the module"),
                 "restarts": camproc["restarts"]},
        "busy": bool(s.get("busy")), "warm": s.get("warm"), "estop": st.get("estop"), "phase": st.get("phase"),
        "running": {k: running.get(k) for k in ("kind", "text", "progress", "item", "items", "search", "sweep", "teach",
                                               "attempt", "attempts_max")} if s.get("busy") else None,
        "seen": {"objects": len(seen.get("objects") or []), "intruders": len(seen.get("intruders") or []),
                 "at_observe": seen.get("at_observe")},
        "command": _sync_command(s),
        "library": {"store": s.get("store"), "views": (s.get("atlas") or {}).get("views") or lib.get("views")},
        "teach": bool((s.get("teach") or {}).get("enabled")),
        "map": _slim_map(mp), "search": mp.get("search") or None, "sweep": mp.get("sweep") or None,
        "map_failing": bool((mp.get("live") or {}).get("failing")),
        "arm": st.get("arm"), "view": fr.get("view"), "frame_age_s": round(now - fr["got"], 1) if fr.get("got") else None,
        "highlight": hl, "reach_mm": REACH_MM}


def _frame_mtime() -> Optional[int]:
    try:
        return (CAM_DIR / "module_frame.json").stat().st_mtime_ns
    except FileNotFoundError:
        return None


@app.get("/api/frame.jpg")
async def frame(request: Request):
    """
    The newest picture, read from disk once per new picture and shared by every browser. ETag = its capture time.
    A page that already has it (If-None-Match) waits here until the next one arrives, up to FRAME_WAIT_S, else gets 304:
    one request per new picture instead of polling, and the picture is on screen as soon as the robot link has it.
    X-Frame-Meta carries the boxes of this very picture (URL-encoded JSON).
    """
    mtime = _frame_mtime()
    if mtime is None:
        raise HTTPException(503, "no picture yet")
    # The picture this page has, by its own ETag: other pages refresh the shared cache meanwhile
    have = _frame_cache["mtime"] if request.headers.get("if-none-match") == _frame_cache["etag"] else None
    deadline = time.monotonic() + FRAME_WAIT_S
    while mtime == have and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        mtime = _frame_mtime() or mtime
    if mtime != _frame_cache["mtime"]:
        meta = _read("module_frame.json")
        try:
            data = (CAM_DIR / "module_frame.jpg").read_bytes()
        except FileNotFoundError:
            raise HTTPException(503, "no picture yet")
        slim = {k: meta.get(k) for k in ("ts", "w", "h", "source", "boxes")}
        _frame_cache.update(mtime=mtime, data=data, etag=f'"{meta.get("ts")}"', meta=quote(json.dumps(slim)))
    headers = {"ETag": _frame_cache["etag"], "Cache-Control": "no-cache", "X-Frame-Meta": _frame_cache["meta"]}
    if request.headers.get("if-none-match") == _frame_cache["etag"]:
        return Response(status_code=304, headers=headers)
    return Response(_frame_cache["data"], media_type="image/jpeg", headers=headers)


def _new_command(text: str) -> dict:
    entry = {"_id": uuid.uuid4().hex, "text": text, "ts": time.time(), "status": "sending"}
    module_cmds.appendleft(entry)
    return entry


def _answered(entry: dict, res: dict) -> None:
    """What the module said at once: a job started, nothing to do (and why), busy, or refused."""
    if res.get("accepted"):
        entry.update(status="running", job=res.get("job"), items=res.get("items"), ignored=res.get("ignored"))
    elif res.get("status"):
        entry.update(status=res["status"], fail_reason=res.get("reason") or res.get("error"))
    else:
        busy = res.get("error") == "busy"
        entry.update(status="busy" if busy else "rejected", fail_reason=None if busy else res.get("error"))


@app.post("/api/commands")
async def add_command(body: TextIn):
    """Grasp (a description, or several), Reset or Scan table: DoCommand on the module, which runs it in the background."""
    text = body.text.strip()
    text = text if text in (RESET_COMMAND, SCAN_COMMAND) else _clean(text)
    entry = _new_command(text)
    try:
        res = await module_call({"reset": True} if text == RESET_COMMAND else {"scan_table": True}
                                if text == SCAN_COMMAND else {"grasp": text})
    except HTTPException as e:
        entry.update(status="error", fail_reason=e.detail)
        raise
    _set_map_highlight(res, text)
    _answered(entry, res)
    return {"id": entry["_id"], "module": res}


@app.post("/api/teach")
async def teach(body: TeachIn):
    """Teach (in "more"): DoCommand {"teach"} on the module, shown like a grasp."""
    name = _clean(body.name)
    entry = _new_command(f"teach: {name}")
    command = {"teach": name}
    if body.alias_of and body.alias_of.strip():
        command["alias_of"] = body.alias_of.strip()
    if body.point and len(body.point) == 2:
        command["point"] = body.point
    try:
        res = await module_call(command)
    except HTTPException as e:
        entry.update(status="error", fail_reason=e.detail)
        raise
    _answered(entry, res)
    return {"id": entry["_id"], "module": res}


@app.post("/api/preview")
async def preview(body: TextIn):
    """What the arm would pick for the text, without moving: the module decides from its cached frame and map."""
    text = _clean(body.text)
    res = await module_call({"preview": text})
    _set_map_highlight(res, text)
    return {"module": res}


@app.post("/api/clear_map")
async def clear_map():
    """Someone rearranged the table: the module forgets its live map and every last-seen position."""
    _map_highlight.clear()
    return {"module": await module_call({"clear_map": True, "reason": "dashboard Clear map"})}


if __name__ == "__main__":
    import uvicorn

    # Loopback by default; set HOST=0.0.0.0 to let teammates on the LAN open it
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
