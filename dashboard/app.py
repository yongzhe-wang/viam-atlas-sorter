"""
Demo dashboard: live camera, "grasp ..." command box, command status, attempts and success-rate chart.

WHY: the demo is "type grasp green square, watch the arm do it, see it logged in Atlas". The dashboard only
reads the camera and writes commands to the Store; the arm worker (AGENT1) claims and executes them, so
this process can never move the arm. The camera runs in a child process (dashboard/camera_worker.py) that
is restarted if the Viam SDK aborts, so the page itself never goes down.

Run from the repo root (laptop):
  python -m uvicorn dashboard.app:app --port 8000          # or: python dashboard/app.py
Env: VIAM_* (camera), MONGODB_URI (Atlas; else data/*.jsonl), SORTER_DATA (JSONL dir),
     ATLAS_CHART_URLS (comma-separated Atlas Charts embed URLs), HOST / PORT for `python dashboard/app.py`,
     DASH_DIRECT_CAMERA=0 to never connect to the robot from this machine (use the worker's frames only),
     DASH_COMMAND_PATH=queue|module (default queue), DASH_MODULE_NAME (default "sorter").

Command path: "queue" writes commands to the Store for a worker to claim (needs Atlas when the worker runs on
another machine). "module" sends them straight to our Viam module's DoCommand ({"grasp": text} / {"reset":
true}) over the robot link and shows its {"status": true} answers, so the demo needs no Atlas at all.

Picture source: the arm worker's latest frame from the Store (Store.put_frame) when it is under 30 s old,
else this machine's own camera connection, else the stale worker frame. Once Atlas is live and the worker
runs on the Linux box next to the arm, the laptop needs no robot connection at all.
"""
import asyncio
import io
import json
import os
import pickle
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sorter.store import Store  # noqa: E402

RESET_COMMAND = "__reset__"  # EXEC_PLAN "Clear square" button; executed by the arm worker
SCAN_COMMAND = "__scan__"  # "Scan table" button: the module scans the whole table into its live map (module mode)
MAX_TEXT = 200
PREVIEW_HIGHLIGHT_S = 15.0
STALE_S = 5.0  # camera state older than this means the camera process is down or restarting
WORKER_FRAME_FRESH_S = 30.0  # prefer the worker's frame over our own camera while it is this fresh
WORKER_OFFLINE_S = 15.0  # agreed with AGENT1: heartbeat at least every 5 s
DIRECT_CAMERA = os.environ.get("DASH_DIRECT_CAMERA", "1") != "0"
COMMAND_PATH = "module" if os.environ.get("DASH_COMMAND_PATH", "queue") == "module" else "queue"
MODULE_MODE = COMMAND_PATH == "module"
MODULE_NAME = os.environ.get("DASH_MODULE_NAME", "sorter")
MODULE_STALE_S = 6.0  # status is polled every 1.5 s; older than this = link or module down
MODULE_CALL_TIMEOUT_S = 25.0  # one DoCommand round trip through the robot-link process

store = Store()
CAM_DIR = store.data / "camera"
camproc = {"pid": None, "restarts": 0, "last_exit": None, "enabled": DIRECT_CAMERA or MODULE_MODE}
module_cmds = deque(maxlen=30)  # module mode: commands sent this session, newest first
_worker_render = {"key": None, "jpeg": None}  # last annotated worker frame, keyed by (worker, ts)


async def supervise_camera() -> None:
    """
    Keep dashboard/camera_worker.py running; restart with backoff when it exits.

    Backoff resets after a worker has run for a minute, so a crash every few hours restarts at once.
    """
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    # What the robot-link process runs: the camera, the module bridge, or both (one shared connection)
    env = {**os.environ, "DASH_CAMERA": "1" if DIRECT_CAMERA else "0", "DASH_MODULE_NAME": MODULE_NAME if MODULE_MODE else ""}
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
    task = asyncio.create_task(supervise_camera()) if DIRECT_CAMERA or MODULE_MODE else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Sorter dashboard", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


def _clean(text: str) -> str:
    text = " ".join(text.split())
    if not text or len(text) > MAX_TEXT:
        raise HTTPException(400, f"text must be 1-{MAX_TEXT} characters")
    return text


def _cam_state() -> dict:
    """Latest state of our own camera process, marked disconnected when its heartbeat is stale."""
    if not DIRECT_CAMERA:
        return {"connected": False, "error": "direct camera off (DASH_DIRECT_CAMERA=0)", "objects": [],
                "frame_age_s": None, "process": camproc}
    try:
        state = json.loads((CAM_DIR / "state.json").read_text())
    except (FileNotFoundError, ValueError):
        state = {"connected": False, "error": "camera process starting", "objects": [], "written": 0}
    if time.time() - state.get("written", 0) > STALE_S:
        state["connected"] = False
        state["error"] = f"camera process restarting (last exit code {camproc['last_exit']})"
    state["frame_age_s"] = state.get("frame_age_s")
    return {**state, "process": camproc}


def _render_worker_frame(data: bytes, doc: dict, highlight: dict = None) -> bytes:
    """
    Worker frame with the worker's boxes (already in this image's pixels), same colours as the direct camera:
    ignored objects grey, intruders red, targets yellow on top. Shape agreed with AGENT1: meta.objects with
    "ignored": true for excluded ones, meta.intruders as a separate list. `highlight` {"id", "text", "score"}
    draws that object thick and green: what a Preview says the arm would pick.
    """
    from PIL import Image, ImageDraw

    meta = doc.get("meta") or {}
    objs, intruders = meta.get("objects") or [], meta.get("intruders") or []
    if not objs and not intruders:
        return data
    im = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(im)

    def tag(box, text, fill, width):
        u0, v0, u1, v1 = [int(x) for x in box]
        draw.rectangle([u0, v0, u1, v1], outline=fill, width=width)
        draw.rectangle([u0, max(v0 - 14, 0), u0 + 6 * len(text) + 4, max(v0 - 14, 0) + 14], fill=fill)
        draw.text((u0 + 2, max(v0 - 13, 1)), text, fill=(0, 0, 0))

    for o in sorted(objs, key=lambda o: not o.get("ignored")):  # ignored first, so targets end up on top
        if not o.get("bbox"):
            continue
        if o.get("ignored"):
            tag(o["bbox"], "ignored", (170, 170, 170), 1)
            continue
        if highlight and o.get("id") is not None and o.get("id") == highlight.get("id"):
            tag(o["bbox"], f"{highlight['text']} {float(highlight.get('score') or 0):.2f}", (40, 220, 90), 4)
            continue
        label = str(o.get("label") or f"#{o.get('id', '')}")
        if o.get("score") is not None:
            label += f" {float(o['score']):.2f}"
        tag(o["bbox"], label, (255, 200, 0), 2)
    for it in intruders:
        if it.get("bbox"):
            tag(it["bbox"], "hand? arm paused", (240, 70, 60), 3)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=80)
    return out.getvalue()


MODULE_FRAME_FRESH_S = 3.0  # module mode: show the module's own labelled frame while it is this fresh


def _module_frame():
    """(jpeg with the module's recognised labels drawn, age_s) from module_frame.*, or None."""
    img, meta_path = CAM_DIR / "module_frame.jpg", CAM_DIR / "module_frame.json"
    try:
        meta = json.loads(meta_path.read_text())
        data = img.read_bytes()
    except (FileNotFoundError, ValueError):
        return None
    age = time.time() - meta_path.stat().st_mtime  # laptop clock: the module's own ts may be skewed
    try:
        hl = json.loads((CAM_DIR / "module_highlight.json").read_text())
        hl = hl if hl.get("until", 0) > time.time() else None
    except (FileNotFoundError, ValueError):
        hl = None
    key = ("module", meta.get("ts"), hl and (hl.get("id"), hl.get("until")))
    if _worker_render["key"] != key:
        objs = [{"id": o.get("id"), "bbox": o["bbox"], "label": o.get("label") or "unknown",
                 "score": o.get("confidence")} for o in meta.get("objects") or []]
        _worker_render.update(key=key, jpeg=_render_worker_frame(data, {"meta": {"objects": objs,
                                                                                 "intruders": meta.get("intruders")}}, hl))
    return _worker_render["jpeg"], age


def _pick_frame():
    """(jpeg, source description, age_s) by the policy in the module docstring, or None."""
    if MODULE_MODE:
        # What the module itself sees and recognises, labelled: the audience sees "green square" before typing
        mf = _module_frame()
        if mf and mf[1] <= MODULE_FRAME_FRESH_S:
            return mf[0], f"module {MODULE_NAME} (robot)", mf[1]
    worker = store.get_frame()
    worker_age = time.time() - worker[1].get("ts", 0) if worker else None
    direct = CAM_DIR / "frame.jpg"
    direct_age = time.time() - direct.stat().st_mtime if DIRECT_CAMERA and direct.exists() else None
    if worker and (worker_age <= WORKER_FRAME_FRESH_S or direct_age is None):
        data, doc = worker
        key = (doc.get("worker"), doc.get("ts"))
        if _worker_render["key"] != key:
            _worker_render.update(key=key, jpeg=_render_worker_frame(data, doc))
        return _worker_render["jpeg"], f"worker {doc.get('worker')}", worker_age
    if direct_age is not None:
        return direct.read_bytes(), "direct camera", direct_age
    return None


@app.get("/")
def index():
    # no-cache: after a dashboard restart with new code, a plain reload must show it (a cached page did not)
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    urls = [u.strip() for u in os.environ.get("ATLAS_CHART_URLS", "").split(",") if u.strip()]
    cam = _cam_state()
    cam["objects"] = len(cam.get("objects", []))
    picked = _pick_frame()
    frame = {"source": picked[1], "age_s": round(picked[2], 1)} if picked else None
    out = {"atlas": store.online, "queued": store.queued_counts(), "camera": cam, "frame": frame,
           "workers": workers(), "command_path": COMMAND_PATH, "charts": urls, "now": time.time()}
    if MODULE_MODE:
        # Perception runs on the robot inside the module: report what it sees there, not our (off) camera
        seen = ((_module_state().get("status") or {}).get("seen") or {})
        out["module_seen"] = {"objects": len(seen.get("objects") or []), "intruders": len(seen.get("intruders") or []),
                              "at_observe": seen.get("at_observe")}
    return out


def _module_state() -> dict:
    """Latest {"status": true} answer from the module (written by the robot-link process), with age_s."""
    try:
        state = json.loads((CAM_DIR / "module_status.json").read_text())
    except (FileNotFoundError, ValueError):
        return {"ok": False, "error": "no answer from the module yet", "age_s": None, "status": {}}
    state["age_s"] = round(time.time() - state.get("ts", 0), 1)
    state.setdefault("status", {})
    return state


def _module_worker() -> dict:
    """The module as a worker-panel row: busy job as phase, link or job errors as error; it takes our commands."""
    st = _module_state()
    s = st.get("status") or {}
    running, last = s.get("running") or {}, s.get("last") or {}
    online = bool(st.get("ok")) and st["age_s"] is not None and st["age_s"] <= MODULE_STALE_S
    if s.get("busy"):
        phase = f"{running.get('kind')}: {running.get('text') or 'square'}"
    elif s.get("warm") is False:
        phase = "warming up"
    else:
        # Always-ready module: say what it currently recognises on the table
        seen = [o.get("label") for o in (s.get("seen") or {}).get("objects") or [] if o.get("label")]
        phase = "idle · sees " + ", ".join(seen) if seen else "idle"
    error = None if st.get("ok") else st.get("error")
    if not error and not s.get("busy") and last.get("status") in ("error", "failed"):
        error = last.get("error") or last.get("fail_reason")
    return {"worker": f"module:{MODULE_NAME} (direct)", "phase": phase, "holding": None, "error": error,
            "age_s": st["age_s"] if st["age_s"] is not None else 1e9, "online": online, "consumer": True}


def _sync_module_cmds() -> None:
    """
    Update sent commands from the module's status: running while its job is the module's current job, then the
    module's last result once that names the same job (fallback for an older module without job ids: the
    first result finished after the command was sent).
    """
    s = _module_state().get("status") or {}
    running, last = s.get("running") or {}, s.get("last") or {}
    for c in module_cmds:
        if c["status"] != "running":
            continue
        if s.get("busy") and running.get("job") == c.get("job"):
            continue
        same = last.get("job") == c.get("job") if last.get("job") else (last.get("finished") or 0) >= c["ts"]
        if last and same:
            c.update(status=last.get("status") or "done", finished=last.get("finished"), attempt_id=last.get("_id"),
                     fail_reason=last.get("fail_reason") or last.get("error"), duration_s=last.get("duration_s"),
                     chosen={"label": last.get("label"), "colour": last.get("colour"), "score": last.get("score")}
                     if last.get("label") or last.get("colour") else None)


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


@app.get("/api/workers")
def workers():
    """
    Each arm worker's latest heartbeat, with `online` = heard from within WORKER_OFFLINE_S and `consumer` =
    it takes dashboard commands.

    WHY: EXEC_PLAN allows exactly one command consumer; two would race for the arm. A heartbeat may say
    consumer=true/false; without the flag the command worker (sfhack4-worker) counts as a consumer and the
    Viam module (module:<name>) does not, because the module heartbeats even with consume_dashboard_commands off.
    """
    out = [_module_worker()] if MODULE_MODE else []
    for w in store.worker_status():
        consumer = w.get("consumer")
        if consumer is None:
            consumer = not str(w.get("worker", "")).startswith("module:")
        out.append({**w, "online": w["age_s"] <= WORKER_OFFLINE_S, "consumer": bool(consumer)})
    return out


@app.get("/api/snapshot.jpg")
def snapshot():
    picked = _pick_frame()
    if picked is None:
        raise HTTPException(503, "no frame yet")
    data, source, age = picked
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Frame-Source": source, "X-Frame-Age": f"{age:.1f}"})


@app.get("/api/objects")
def objects():
    return _cam_state().get("objects", [])


@app.post("/api/commands")
async def add_command(body: TextIn):
    """Queue mode: Store command for a worker. Module mode: DoCommand on the module, which runs it in the background."""
    text = body.text.strip()
    text = text if text in (RESET_COMMAND, SCAN_COMMAND) else _clean(text)
    if not MODULE_MODE:
        if text == SCAN_COMMAND:
            raise HTTPException(400, "Scan table needs the Viam module (DASH_COMMAND_PATH=module)")
        return {"id": store.add_command(text)}
    entry = {"_id": uuid.uuid4().hex, "text": text, "ts": time.time(), "status": "sending"}
    module_cmds.appendleft(entry)
    try:
        res = await module_call({"reset": True} if text == RESET_COMMAND else {"scan_table": True}
                                if text == SCAN_COMMAND else {"grasp": text})
    except HTTPException as e:
        entry.update(status="error", fail_reason=e.detail)
        raise
    entry["decide_ms"] = res.get("decide_ms")
    _set_map_highlight(res, text)
    if res.get("accepted"):
        target = res.get("target") or {}
        entry.update(status="running", job=res.get("job"),
                     chosen={"label": target.get("label"), "colour": target.get("colour"), "score": target.get("score")}
                     if target else None)
    elif res.get("status") in ("not_found", "ambiguous", "already_in_square"):
        # Always-ready module answers at once from its cached frame: nothing started, here is why
        entry.update(status=res["status"], fail_reason=res.get("reason"))
    else:
        busy = res.get("error") == "busy"
        # the "busy" badge says it all; a refusal for any other reason shows its message
        entry.update(status="busy" if busy else "rejected", fail_reason=None if busy else res.get("error"))
    return {"id": entry["_id"], "module": res}


@app.get("/api/commands")
def commands(limit: int = 15):
    if MODULE_MODE:
        _sync_module_cmds()
        return list(module_cmds)[: min(limit, 100)]
    return store.list_commands(limit=min(limit, 100))


@app.post("/api/preview")
async def preview(body: TextIn):
    """
    Rank the objects currently in view against the text (no arm motion) and highlight the best one.

    Shows the audience which object "green square" resolves to before the arm moves. Module mode asks the
    module ({"preview": text}: its cached frame, CLIP already loaded on the robot), so this laptop never loads
    CLIP. Queue mode ranks the crops our own camera process saved for the latest frame (objs.pkl).
    """
    text = _clean(body.text)
    if MODULE_MODE:
        res = await module_call({"preview": text})
        if res.get("ok") and res.get("target"):
            t = res["target"]
            (CAM_DIR / "module_highlight.json").write_text(json.dumps(
                {"id": t.get("id"), "text": text, "score": t.get("score"), "until": time.time() + PREVIEW_HIGHLIGHT_S}))
        _set_map_highlight(res, text)
        return {"module": res}
    try:
        objs = [SimpleNamespace(**o) for o in pickle.loads((CAM_DIR / "objs.pkl").read_bytes())]
    except FileNotFoundError:
        objs = []
    objs = [o for o in objs if isinstance(o.crop, np.ndarray) and o.crop.size]
    if not objs:
        raise HTTPException(409, "no objects detected in the current frame")
    try:
        from sorter import clip_embed
    except Exception as e:
        raise HTTPException(503, f"clip_embed not available: {e}")
    ranked = await asyncio.to_thread(clip_embed.rank, objs, text, store)
    if not ranked:
        raise HTTPException(409, "nothing ranked")
    best, score = ranked[0]
    hl = {"id": best.id, "text": text, "score": float(score), "until": time.time() + PREVIEW_HIGHLIGHT_S}
    (CAM_DIR / "highlight.json").write_text(json.dumps(hl))
    return [{"id": o.id, "score": float(s)} for o, s in ranked]


MAP_HIGHLIGHT_S = 20.0
_map_highlight = {}  # {"text", "label", "xy", "until"}: the object the last preview / grasp resolved to


def _set_map_highlight(res: dict, text: str) -> None:
    """Remember what a module answer pointed at (map xy or recognised label), to ring it on the table map."""
    target = res.get("target") or {}
    xy, label = res.get("target_xy"), res.get("label") or target.get("label")
    # Every new answer replaces the ring: a miss ("ghost" not found) must not leave the previous hit ringed
    _map_highlight.clear()
    if res.get("ok") is not False and (xy or label):
        _map_highlight.update(text=text, label=label, xy=xy, until=time.time() + MAP_HIGHLIGHT_S)


@app.get("/api/table_map")
def table_map():
    """
    Top-down table view: the live table map (objects with table xy and age, no embeddings), each object's last
    sighting, and sweep progress. Module mode: from the module (via the robot link), ages on the robot's clock.
    """
    if MODULE_MODE:
        try:
            d = json.loads((CAM_DIR / "module_table_map.json").read_text())
        except (FileNotFoundError, ValueError):
            d = {}
        since = time.time() - d.get("fetched_at", time.time())  # keep ages counting between fetches
        robot_now = (d.get("now") or time.time()) + since
        m = d.get("map") or None
        sightings = d.get("sightings") or []
        running = ((_module_state().get("status") or {}).get("running") or {})
        sweep = running.get("sweep") or None
        search = {**running["search"], "text": running.get("text")} if running.get("search") else None
        n_obs, live = d.get("observations"), d.get("live")
    else:
        m, sightings, robot_now = store.get_table_map(with_emb=False), store.list_sightings(), time.time()
        sweep = next((w.get("sweep") for w in workers() if w.get("online") and w.get("sweep")), None)
        search, n_obs, live = None, store.observation_stats()["count"], None
    if m:
        m["age_s"] = round(robot_now - (m.get("ts") or robot_now), 1)
        # Live map: each object's own age (seen_ts); an older map without it -> the map's age for all
        for o in m.get("objects") or []:
            o["age_s"] = round(robot_now - (o.get("seen_ts") or m.get("ts") or robot_now), 1)
        if m.get("swept_ts"):
            m["swept_s_ago"] = round(robot_now - m["swept_ts"], 1)
        if m.get("cleared_ts"):
            m["cleared_s_ago"] = round(robot_now - m["cleared_ts"], 1)
    for s in sightings:
        s["age_s"] = round(robot_now - (s.get("ts") or robot_now), 1)
    hl = dict(_map_highlight) if _map_highlight.get("until", 0) > time.time() else None
    return {"map": m, "sightings": sightings, "sweep": sweep, "search": search, "reach_mm": [120, 650], "highlight": hl,
            "observations": n_obs, "live": live}


@app.post("/api/clear_map")
async def clear_map():
    """
    "Clear map" button: someone rearranged the table. Module mode: DoCommand clear_map (live map, sightings and
    the cached frame; old ones to the Store's history). Queue mode: the same on the Store directly.
    """
    _map_highlight.clear()
    if MODULE_MODE:
        return {"module": await module_call({"clear_map": True, "reason": "dashboard Clear map"})}
    empty = store.clear_table_map("dashboard Clear map")
    return {"ok": True, "cleared_objects": empty["cleared_objects"], "cleared_sightings": store.clear_sightings()}


@app.get("/api/labels")
def labels():
    """Known objects for the "you can ask for" list: [[label, *aliases], ...] from the view library."""
    return store.list_labels()


@app.get("/api/attempts")
def attempts(limit: int = 30):
    """Recent attempts; clip_frames = how many replay frames each has (0 = no Replay button)."""
    docs = store.list_attempts(limit=min(limit, 200))
    counts = store.clip_counts([d["_id"] for d in docs])
    return [{**d, "clip_frames": counts.get(d["_id"], 0)} for d in docs]


@app.get("/api/stats")
def stats():
    """
    Success over attempts (cumulative and rolling 10) and per label. Reset passes (clearing the square,
    EXEC_PLAN "Reset") are not grasps on request, so they are excluded here and counted separately.
    """
    every = store.list_attempts(limit=0, newest_first=False)
    picks = [a for a in every if a.get("mode") != "reset"]
    resets = [a for a in every if a.get("mode") == "reset"]
    series, ok = [], 0
    for i, a in enumerate(picks):
        ok += bool(a.get("success"))
        last = picks[max(0, i - 9): i + 1]
        series.append({"i": i + 1, "ts": a.get("ts"), "success": bool(a.get("success")),
                       "cum_rate": ok / (i + 1), "roll10": sum(bool(x.get("success")) for x in last) / len(last)})
    per_label = {}
    for a in picks:
        d = per_label.setdefault(a.get("label") or "unknown", {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += bool(a.get("success"))
    times = [a["duration_s"] for a in picks if a.get("success") and a.get("duration_s")]
    return {"n": len(picks), "ok": ok, "rate": ok / len(picks) if picks else None,
            "mean_success_s": sum(times) / len(times) if times else None, "series": series,
            "resets": {"n": len(resets), "ok": sum(bool(a.get("success")) for a in resets)},
            "per_label": [{"label": k, **v, "rate": v["ok"] / v["n"]} for k, v in sorted(per_label.items())]}


@app.get("/api/image/{coll}/{doc_id}/{field}")
def image(coll: str, doc_id: str, field: str):
    data = store.get_image(coll, doc_id, field)
    if data is None:
        raise HTTPException(404, "no image")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# ---- replay: every frame of an attempt (Store.put_clip_frame) -----------------------------------------

CLIP_MIN_MS, CLIP_MAX_MS = 100, 1500  # per-frame display time in exports: real gaps, clamped to stay watchable


def _clip_frames(attempt_id: str) -> list:
    frames = store.list_clip_frames(attempt_id)
    if not frames:
        raise HTTPException(404, "no replay frames for this attempt")
    return frames


def _clip_durations_ms(frames: list) -> list:
    """How long to show each frame: the real gap to the next one, clamped; the last frame holds 1.5 s."""
    out = []
    for a, b in zip(frames, frames[1:] + [None]):
        gap = (b["ts"] - a["ts"]) * 1000 if b else CLIP_MAX_MS
        out.append(int(min(max(gap, CLIP_MIN_MS), CLIP_MAX_MS)))
    return out


@app.get("/api/clips/{attempt_id}")
def clip(attempt_id: str):
    """Frame list for the player: [{"seq", "phase", "ts", "meta"}, ...]."""
    return _clip_frames(attempt_id)


@app.get("/api/clips/{attempt_id}/{seq}.jpg")
def clip_frame(attempt_id: str, seq: int):
    data = store.get_clip_frame(attempt_id, seq)
    if data is None:
        raise HTTPException(404, "no such frame")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _labelled_frames(attempt_id: str) -> tuple:
    """(PIL frames with "phase  +t s" stamped on them, durations ms), shared by the GIF and MP4 exports."""
    from PIL import Image, ImageDraw

    frames = _clip_frames(attempt_id)
    t0, images = frames[0]["ts"], []
    for f in frames:
        im = Image.open(io.BytesIO(store.get_clip_frame(attempt_id, f["seq"]))).convert("RGB")
        text = f"{f['phase']}  +{f['ts'] - t0:.1f} s"
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(255, 255, 255))
        images.append(im)
    return images, _clip_durations_ms(frames)


@app.get("/api/clips/{attempt_id}/download.gif")
def clip_gif(attempt_id: str):
    images, durations = _labelled_frames(attempt_id)
    buf = io.BytesIO()
    images[0].save(buf, "GIF", save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return Response(buf.getvalue(), media_type="image/gif",
                    headers={"Content-Disposition": f'attachment; filename="attempt-{attempt_id}.gif"'})


@app.get("/api/clips/{attempt_id}/download.mp4")
async def clip_mp4(attempt_id: str):
    """MP4 via ffmpeg's concat demuxer (per-frame durations); 503 when ffmpeg is not installed."""
    import shutil
    import tempfile

    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "ffmpeg is not installed on this machine; use the GIF download")
    images, durations = _labelled_frames(attempt_id)
    with tempfile.TemporaryDirectory() as tmp:
        lines = []
        for i, (im, ms) in enumerate(zip(images, durations)):
            name = f"{i:05d}.jpg"
            im.save(Path(tmp) / name, "JPEG", quality=90)
            lines += [f"file '{name}'", f"duration {ms / 1000:.3f}"]
        lines.append(f"file '{len(images) - 1:05d}.jpg'")  # concat demuxer needs the last file twice
        (Path(tmp) / "list.txt").write_text("\n".join(lines) + "\n")
        out = Path(tmp) / "out.mp4"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "25", "-movflags", "+faststart",
            "out.mp4", cwd=tmp, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0 or not out.exists():
            raise HTTPException(500, f"ffmpeg failed: {err.decode(errors='replace')[-300:]}")
        data = out.read_bytes()
    return Response(data, media_type="video/mp4",
                    headers={"Content-Disposition": f'attachment; filename="attempt-{attempt_id}.mp4"'})


if __name__ == "__main__":
    import uvicorn

    # Loopback by default; set HOST=0.0.0.0 to let teammates on the LAN open it
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
