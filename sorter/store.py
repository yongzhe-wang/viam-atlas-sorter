"""
Data layer for the sorter: MongoDB Atlas when MONGODB_URI is set, otherwise local JSONL files that are
flushed to Atlas later.

WHY: the arm worker, perception and the dashboard all log through this one interface, so the demo runs
before the Atlas cluster exists and nothing is lost when the network drops. Ids are generated here
(uuid hex) in both modes, so a later flush is an idempotent upsert.

Collections (schemas: docs/EXEC_PLAN.md): attempts, calib_samples, views, commands.
Images (attempts.img_before / img_after, views.jpeg) are passed in as JPEG bytes or a file path. Online they
are stored inline as small JPEG binaries (downscaled to <= 640 px wide) so a dashboard on another machine
can show them; offline they are written to data/img/ and the doc keeps {"path": ...}.

Offline mode only shares data between processes on the same machine; the arm worker on the Linux box and
the dashboard on the laptop need Atlas to see each other's commands.

Env: MONGODB_URI, MONGODB_DB (default "sorter"), SORTER_DATA (default <repo>/data)
"""
import fcntl
import io
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

COLLECTIONS = ("attempts", "calib_samples", "views", "commands")
# Latest-value collections: one doc per worker, overwritten in place, never queued or flushed
LIVE_COLLECTIONS = ("status", "frames")
VECTOR_INDEX = "views_vec"  # Atlas Vector Search index on views.embedding (scripts/atlas_setup.py)
EMBED_DIM = 512
IMAGE_FIELDS = {"attempts": ("img_before", "img_after"), "views": ("jpeg",)}
MAX_IMG_WIDTH = 640


def _new_id() -> str:
    return uuid.uuid4().hex


def _small_jpeg_scaled(data: bytes) -> tuple:
    """(JPEG at most MAX_IMG_WIDTH wide, scale factor applied) so docs stay small (Atlas doc limit is 16 MB)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    scale = 1.0
    if im.width > MAX_IMG_WIDTH:
        scale = MAX_IMG_WIDTH / im.width
        im = im.resize((MAX_IMG_WIDTH, round(im.height * scale)))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=80)
    return out.getvalue(), scale


def _small_jpeg(data: bytes) -> bytes:
    return _small_jpeg_scaled(data)[0]


def _safe_name(worker: str) -> str:
    """Worker id usable as a file name (offline mode)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", worker)[:80] or "worker"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write then rename, so a reader in another process never sees a half-written file."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _with_date(doc: dict) -> dict:
    """Add `at` (a BSON Date from `ts`) for Atlas: Charts can bin time only on Date fields, not epoch floats."""
    from datetime import datetime, timezone

    if isinstance(doc.get("ts"), (int, float)):
        doc["at"] = datetime.fromtimestamp(doc["ts"], timezone.utc)
    return doc


def _cosine01(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity mapped to [0, 1] as (1 + cos) / 2, the same scale Atlas reports for cosine indexes."""
    m = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    return (1.0 + m @ q) / 2.0


class Store:
    """One object per process. `online` is True when writes go straight to Atlas."""

    def __init__(self, uri: str = None, db: str = None, data_dir: str = None):
        default_data = Path(__file__).resolve().parent.parent / "data"
        self.data = Path(data_dir or os.environ.get("SORTER_DATA") or default_data)
        (self.data / "img").mkdir(parents=True, exist_ok=True)
        uri = uri if uri is not None else os.environ.get("MONGODB_URI")
        self.db = None
        self._views_cache = None  # ((mtime_ns, size), views, matrix) for offline search_views
        if uri:
            from pymongo import MongoClient

            self.db = MongoClient(uri, serverSelectionTimeoutMS=8000)[db or os.environ.get("MONGODB_DB", "sorter")]
            self.flush()

    @property
    def online(self) -> bool:
        return self.db is not None

    # ---- low-level storage -------------------------------------------------------------------------

    def _file(self, coll: str) -> Path:
        return self.data / f"{coll}.jsonl"

    @contextmanager
    def _lock(self, coll: str):
        """Cross-process lock for read-modify-write on one collection's JSONL file (offline mode)."""
        with open(self.data / f".{coll}.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _append(self, coll: str, record: dict) -> None:
        with self._lock(coll), self._file(coll).open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _read_offline(self, coll: str) -> list:
        """All docs of a collection from JSONL. Commands are an event log (put/set), folded into final state."""
        path = self._file(coll)
        if not path.exists():
            return []
        docs, order = {}, []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("_op") == "set":
                if rec["_id"] in docs:
                    docs[rec["_id"]].update(rec["fields"])
                continue
            rec.pop("_op", None)
            if rec["_id"] not in docs:
                order.append(rec["_id"])
            docs[rec["_id"]] = rec
        return [docs[i] for i in order]

    def _image_in(self, value):
        """Normalise an image argument (bytes or path) to what the current mode stores."""
        if value is None or isinstance(value, dict):
            return value
        data = Path(value).read_bytes() if isinstance(value, (str, Path)) else bytes(value)
        data = _small_jpeg(data)
        if self.online:
            from bson import Binary

            return {"jpeg": Binary(data)}
        name = f"img/{_new_id()}.jpg"
        (self.data / name).write_bytes(data)
        return {"path": name}

    def _insert(self, coll: str, doc: dict) -> str:
        doc = dict(doc)
        doc.setdefault("_id", _new_id())
        doc.setdefault("ts", time.time())
        for field in IMAGE_FIELDS.get(coll, ()):
            if field in doc:
                doc[field] = self._image_in(doc[field])
        if self.online:
            self.db[coll].insert_one(_with_date(doc))
        else:
            self._append(coll, doc)
        return doc["_id"]

    # ---- contract (docs/EXEC_PLAN.md) --------------------------------------------------------------

    def log_attempt(self, doc: dict) -> str:
        """One pick-and-place attempt; returns its id."""
        return self._insert("attempts", doc)

    def log_calib(self, doc: dict) -> None:
        """One jaw-calibration sample (scripts/jaw_calib.py)."""
        self._insert("calib_samples", doc)

    def add_view(self, label: str, embedding, jpeg: bytes, meta: dict) -> str:
        """One labelled view of an object for retrieval; embedding is a 512-d CLIP vector."""
        vec = [float(x) for x in np.asarray(embedding, dtype=float).ravel()]
        if len(vec) != EMBED_DIM:
            raise ValueError(f"embedding has {len(vec)} dims, expected {EMBED_DIM}")
        return self._insert("views", {**(meta or {}), "label": label, "embedding": vec, "jpeg": jpeg})

    def _views_offline(self) -> tuple:
        """
        (views, embedding matrix) from data/views.jsonl, re-read only when the file changes.

        WHY: the module's idle perception searches the library for every object about once a second; parsing
        the ~1 MB file each time was most of that cost. Keyed on (mtime, size), so appends invalidate it.
        """
        path = self._file("views")
        try:
            st = path.stat()
        except FileNotFoundError:
            return [], None
        key = (st.st_mtime_ns, st.st_size)
        if self._views_cache is None or self._views_cache[0] != key:
            views = self._read_offline("views")
            matrix = np.array([v["embedding"] for v in views], dtype=float) if views else None
            self._views_cache = (key, views, matrix)
        return self._views_cache[1], self._views_cache[2]

    def search_views(self, embedding, k: int = 5, label: str = None) -> list:
        """
        Nearest labelled views, best first, each with "score" in [0, 1] (cosine, Atlas scale).

        Online: Atlas $vectorSearch on views.embedding (optionally pre-filtered by label).
        Offline: brute-force cosine over data/views.jsonl.
        Returned docs omit the embedding and image bytes.
        """
        q = np.asarray(embedding, dtype=float).ravel()
        if self.online:
            stage = {"index": VECTOR_INDEX, "path": "embedding", "queryVector": q.tolist(),
                     "numCandidates": max(50, 20 * k), "limit": k}
            if label:
                stage["filter"] = {"label": label}
            pipeline = [{"$vectorSearch": stage},
                        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                        {"$project": {"embedding": 0, "jpeg": 0}}]
            return list(self.db.views.aggregate(pipeline))
        views, matrix = self._views_offline()
        if label:
            keep = [i for i, v in enumerate(views) if v.get("label") == label]
            views, matrix = [views[i] for i in keep], (matrix[keep] if keep else None)
        if not views:
            return []
        scores = _cosine01(matrix, q)
        out = []
        for i in np.argsort(-scores)[:k]:
            v = {key: val for key, val in views[i].items() if key not in ("embedding", "jpeg")}
            v["score"] = float(scores[i])
            out.append(v)
        return out

    def list_labels(self) -> list:
        """
        Every object the library knows, as [[label, *aliases], ...]: label first, then its aliases (sorted,
        deduplicated across all of that label's views), ordered by label. Same result online and offline.

        WHY: clip_embed.library_names() needs which queries the library covers, and the dashboard shows the
        audience what it can ask for; both previously reached into views directly.
        """
        if self.online:
            rows = self.db.views.find({}, {"label": 1, "aliases": 1, "_id": 0})
        else:
            rows = self._read_offline("views")
        names = {}
        for r in rows:
            label = r.get("label")
            if not label:
                continue
            names.setdefault(label, set()).update(a for a in (r.get("aliases") or []) if a and a != label)
        return [[label, *sorted(names[label])] for label in sorted(names)]

    def library_stats(self) -> dict:
        """{"labels": N, "views": M} for the object library, so a deployment can show it loaded."""
        views = self.db.views.count_documents({}) if self.online else len(self._read_offline("views"))
        return {"labels": len(self.list_labels()), "views": views}

    def add_command(self, text: str) -> str:
        """Queue a dashboard command ("grasp green square"); status starts as pending."""
        now = time.time()
        return self._insert("commands", {"text": text, "status": "pending", "ts": now, "updated": now})

    def next_command(self):
        """
        Claim the oldest pending command (pending -> running) and return it, or None.

        Atomic: find_one_and_update online; file lock + event log offline, so two workers never claim one.
        """
        now = time.time()
        if self.online:
            from pymongo import ReturnDocument

            return self.db.commands.find_one_and_update(
                {"status": "pending"}, {"$set": {"status": "running", "started": now, "updated": now}},
                sort=[("ts", 1)], return_document=ReturnDocument.AFTER)
        with self._lock("commands"):
            pending = [c for c in self._read_offline("commands") if c["status"] == "pending"]
            if not pending:
                return None
            cmd = min(pending, key=lambda c: c["ts"])
            fields = {"status": "running", "started": now, "updated": now}
            with self._file("commands").open("a") as f:
                f.write(json.dumps({"_op": "set", "_id": cmd["_id"], "fields": fields}) + "\n")
        return {**cmd, **fields}

    def finish_command(self, cmd_id, status: str, attempt_id=None, chosen=None) -> None:
        """Close a command: status is e.g. done / failed / not_found; links the attempt and chosen object."""
        fields = {"status": status, "updated": time.time(), "finished": time.time()}
        if attempt_id is not None:
            fields["attempt_id"] = attempt_id
        if chosen is not None:
            fields["chosen"] = chosen
        if self.online:
            self.db.commands.update_one({"_id": cmd_id}, {"$set": fields})
        else:
            self._append("commands", {"_op": "set", "_id": cmd_id, "fields": fields})

    # ---- live worker state (agreed with AGENT1; one doc per worker, overwritten) ---------------------

    def heartbeat(self, worker: str, phase: str, holding=None, error=None, command_id=None, **extra) -> None:
        """
        Record what the arm worker is doing right now. Call on every phase change and at least every 5 s;
        the dashboard shows a worker as offline after 15 s without a heartbeat.

        phase: idle / observe / servo / descend / grab / carry / place / reset / paused_person / reconnecting /
        parked / error. holding: gripper is_holding_something() or None. extra: e.g. pose, attempt_id.
        """
        doc = {**extra, "_id": worker, "worker": worker, "ts": time.time(), "phase": phase,
               "holding": holding, "error": error, "command_id": command_id}
        if self.online:
            self.db.status.replace_one({"_id": worker}, _with_date(doc), upsert=True)
        else:
            (self.data / "status").mkdir(exist_ok=True)
            _write_atomic(self.data / "status" / f"{_safe_name(worker)}.json", json.dumps(doc, default=str).encode())

    def worker_status(self) -> list:
        """All workers' latest heartbeat, most recent first, each with age_s."""
        if self.online:
            docs = list(self.db.status.find({}))
        else:
            docs = []
            for p in (self.data / "status").glob("*.json"):
                try:
                    docs.append(json.loads(p.read_text()))
                except ValueError:
                    continue
        now = time.time()
        for d in docs:
            d.pop("at", None)
            d["age_s"] = round(now - d.get("ts", 0), 1)
        return sorted(docs, key=lambda d: d["age_s"])

    def put_frame(self, worker: str, jpeg: bytes, meta: dict = None) -> None:
        """
        Store the worker's latest observe frame (overwrites). meta["objects"][i]["bbox"] is in pixels of the
        image passed in; it is rescaled with the JPEG so the dashboard can draw boxes on the stored frame.
        """
        data, scale = _small_jpeg_scaled(bytes(jpeg))
        meta = dict(meta or {})
        # objects (targets + {"ignored": true}) and intruders both carry boxes in source-image pixels
        for key in ("objects", "intruders"):
            if scale != 1.0 and meta.get(key):
                meta[key] = [{**o, "bbox": [float(x) * scale for x in o["bbox"]]} if o.get("bbox") else o
                             for o in meta[key]]
        doc = {"_id": worker, "worker": worker, "ts": time.time(), "meta": meta}
        if self.online:
            from bson import Binary

            self.db.frames.replace_one({"_id": worker}, _with_date({**doc, "jpeg": Binary(data)}), upsert=True)
        else:
            folder = self.data / "frames"
            folder.mkdir(exist_ok=True)
            name = _safe_name(worker)
            _write_atomic(folder / f"{name}.jpg", data)
            _write_atomic(folder / f"{name}.json", json.dumps(doc, default=str).encode())

    def get_frame(self, worker: str = None):
        """(jpeg bytes, doc without image) of the newest worker frame (or that worker's), or None."""
        if self.online:
            query = {"_id": worker} if worker else {}
            doc = next(iter(self.db.frames.find(query).sort("ts", -1).limit(1)), None)
            if not doc:
                return None
            data = bytes(doc.pop("jpeg"))
            doc.pop("at", None)
            return data, doc
        folder = self.data / "frames"
        names = [_safe_name(worker)] if worker else [p.stem for p in folder.glob("*.json")]
        best = None
        for name in names:
            try:
                doc = json.loads((folder / f"{name}.json").read_text())
                data = (folder / f"{name}.jpg").read_bytes()
            except (FileNotFoundError, ValueError):
                continue
            if best is None or doc.get("ts", 0) > best[1].get("ts", 0):
                best = (data, doc)
        return best

    # ---- reads for the dashboard -------------------------------------------------------------------

    def _list(self, coll: str, limit: int, newest_first: bool = True) -> list:
        """Docs without embeddings or image bytes; image fields reduced to True when present."""
        if self.online:
            cur = self.db[coll].find({}, {"embedding": 0}).sort("ts", -1 if newest_first else 1)
            docs = list(cur.limit(limit) if limit else cur)
        else:
            docs = sorted(self._read_offline(coll), key=lambda d: d.get("ts", 0), reverse=newest_first)
            docs = docs[:limit] if limit else docs
        out = []
        for d in docs:
            d = {k: v for k, v in d.items() if k != "embedding"}
            for field in IMAGE_FIELDS.get(coll, ()):
                if field in d:
                    d[field] = bool(d[field])
            out.append(d)
        return out

    def list_attempts(self, limit: int = 50, newest_first: bool = True) -> list:
        return self._list("attempts", limit, newest_first)

    def list_commands(self, limit: int = 20) -> list:
        return self._list("commands", limit)

    def get_image(self, coll: str, doc_id: str, field: str):
        """JPEG bytes of one image field, or None."""
        if field not in IMAGE_FIELDS.get(coll, ()):
            return None
        if self.online:
            doc = self.db[coll].find_one({"_id": doc_id}, {field: 1})
        else:
            doc = next((d for d in self._read_offline(coll) if d["_id"] == doc_id), None)
        img = (doc or {}).get(field)
        if not img:
            return None
        if "jpeg" in img:
            return bytes(img["jpeg"])
        path = (self.data / img["path"]).resolve()
        # Only serve files from our own image folder
        if self.data.resolve() not in path.parents or not path.exists():
            return None
        return path.read_bytes()

    def queued_counts(self) -> dict:
        """Docs waiting in local JSONL for a flush (all zero once online and flushed)."""
        return {c: len(self._read_offline(c)) for c in COLLECTIONS}

    # ---- offline -> Atlas --------------------------------------------------------------------------

    def flush(self) -> dict:
        """
        Upsert every locally queued doc into Atlas, then archive the JSONL files. Safe to repeat.

        Images saved as files offline are loaded and stored inline. Returns counts per collection.
        """
        if not self.online:
            return {}
        from bson import Binary

        counts = {}
        for coll in COLLECTIONS:
            docs = self._read_offline(coll)
            for d in docs:
                for field in IMAGE_FIELDS.get(coll, ()):
                    img = d.get(field)
                    if isinstance(img, dict) and "path" in img and (self.data / img["path"]).exists():
                        d[field] = {"jpeg": Binary((self.data / img["path"]).read_bytes())}
                self.db[coll].replace_one({"_id": d["_id"]}, _with_date(d), upsert=True)
            if docs:
                self._file(coll).rename(self.data / f"{coll}.jsonl.flushed-{int(time.time())}")
            counts[coll] = len(docs)
        return counts
