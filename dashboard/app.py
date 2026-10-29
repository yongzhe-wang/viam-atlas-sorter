"""
Demo dashboard: live camera, "grasp ..." command box, command status, attempts and success-rate chart.

WHY: the demo is "type grasp green square, watch the arm do it, see it logged in Atlas". The dashboard only
reads the camera and writes commands to the Store; the arm worker (AGENT1) claims and executes them, so
this process can never move the arm. The camera runs in a child process (dashboard/camera_worker.py) that
is restarted if the Viam SDK aborts, so the page itself never goes down.

Run from the repo root (laptop):
  python -m uvicorn dashboard.app:app --port 8000          # or: python dashboard/app.py
Env: VIAM_* (camera), MONGODB_URI (Atlas; else data/*.jsonl), SORTER_DATA (JSONL dir),
     ATLAS_CHART_URLS (comma-separated Atlas Charts embed URLs), HOST / PORT for `python dashboard/app.py`.
"""
import asyncio
import json
import os
import pickle
import sys
import time
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
MAX_TEXT = 200
PREVIEW_HIGHLIGHT_S = 15.0
STALE_S = 5.0  # camera state older than this means the camera process is down or restarting

store = Store()
CAM_DIR = store.data / "camera"
camproc = {"pid": None, "restarts": 0, "last_exit": None}


async def supervise_camera() -> None:
    """
    Keep dashboard/camera_worker.py running; restart with backoff when it exits.

    Backoff resets after a worker has run for a minute, so a crash every few hours restarts at once.
    """
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    backoff = 1.0
    while True:
        started = time.time()
        with open(CAM_DIR / "worker.log", "wb") as log:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "dashboard.camera_worker", str(CAM_DIR), cwd=str(ROOT), stdout=log, stderr=log)
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
    task = asyncio.create_task(supervise_camera())
    yield
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
    """Latest camera state from the worker, marked disconnected when the heartbeat is stale."""
    try:
        state = json.loads((CAM_DIR / "state.json").read_text())
    except (FileNotFoundError, ValueError):
        state = {"connected": False, "error": "camera process starting", "objects": [], "written": 0}
    if time.time() - state.get("written", 0) > STALE_S:
        state["connected"] = False
        state["error"] = f"camera process restarting (last exit code {camproc['last_exit']})"
    state["frame_age_s"] = state.get("frame_age_s")
    return {**state, "process": camproc}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
def status():
    urls = [u.strip() for u in os.environ.get("ATLAS_CHART_URLS", "").split(",") if u.strip()]
    cam = _cam_state()
    cam["objects"] = len(cam.get("objects", []))
    return {"atlas": store.online, "queued": store.queued_counts(), "camera": cam, "charts": urls, "now": time.time()}


@app.get("/api/snapshot.jpg")
def snapshot():
    path = CAM_DIR / "frame.jpg"
    if not path.exists():
        raise HTTPException(503, "no frame yet")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/objects")
def objects():
    return _cam_state().get("objects", [])


@app.post("/api/commands")
def add_command(body: TextIn):
    text = body.text.strip()
    return {"id": store.add_command(RESET_COMMAND if text == RESET_COMMAND else _clean(text))}


@app.get("/api/commands")
def commands(limit: int = 15):
    return store.list_commands(limit=min(limit, 100))


@app.post("/api/preview")
async def preview(body: TextIn):
    """
    Rank the objects currently in view against the text (no arm motion) and highlight the best one.

    Shows the audience which object "green square" resolves to before the arm moves. Uses the crops the
    camera process saved for the latest frame (objs.pkl).
    """
    text = _clean(body.text)
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


@app.get("/api/attempts")
def attempts(limit: int = 30):
    return store.list_attempts(limit=min(limit, 200))


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


if __name__ == "__main__":
    import uvicorn

    # Loopback by default; set HOST=0.0.0.0 to let teammates on the LAN open it
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
