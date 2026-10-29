"""
The module's one live table map: every frame the module sees updates it (VIAM CENTERAL: "keep a dynamic map for
finding; every picture should be used towards updating the internal map").

AGENT3's sorter.table_map.update_map does the perception and the bookkeeping (new objects, moved ones, negative
evidence for ones that are gone); this class only schedules it, so the arm never waits on it:
  - one update at a time, in a worker thread, on a deep copy of the map: readers always see a whole map;
  - newest frame wins: a frame offered while an update runs replaces the one waiting (counted as dropped);
    frames offered with keep=True (a sweep) queue instead and are never dropped;
  - the map is saved with Store.put_table_map at most every persist_s, so the Store's latest map is the live one;
  - it starts empty and is never reloaded from the Store: the table may have changed while the module was down
    (VIAM CENTERAL: "remember to clear [the map] if we restart"); clear() forgets it all on demand;
  - every frame it folds in is also appended to the Store's observation log (pose, what was detected, a small
    JPEG and depth), so the map is a fold over that log since `epoch` and rebuild() can replay it (VIAM CENTERAL:
    "all data should be saved to continuously form the map"). clear() moves the epoch; it never deletes the log.
"""
import asyncio
import copy
import inspect
import io
import time
from collections import deque
from logging import getLogger
from typing import Optional

import numpy as np

LOGGER = getLogger(__name__)
LOG_JPEG_WIDTH = 640  # observation log image width (VIAM CENTERAL: jpeg <= 640 px)
LOG_DEPTH_WIDTH = 320  # observation log depth: downsampled to 320 x 180 uint16 PNG (~60 KB)


def _small_jpeg(rgb) -> bytes:
    """The frame as a JPEG at most LOG_JPEG_WIDTH wide."""
    from PIL import Image

    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if im.width > LOG_JPEG_WIDTH:
        im = im.resize((LOG_JPEG_WIDTH, round(im.height * LOG_JPEG_WIDTH / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _small_depth(depth) -> tuple:
    """(16-bit PNG of the depth in mm at LOG_DEPTH_WIDTH wide, scale = original width / that width)."""
    from PIL import Image

    d = np.clip(np.nan_to_num(np.asarray(depth, dtype=np.float64)), 0, 65535).astype(np.uint16)
    scale = d.shape[1] / LOG_DEPTH_WIDTH
    im = Image.fromarray(d).resize((LOG_DEPTH_WIDTH, max(1, round(d.shape[0] / scale))), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), round(scale, 4)


class LiveMap:
    def __init__(self, store, update_fn, persist_s: float = 2.0, initial: Optional[dict] = None):
        self.store, self.update_fn, self.persist_s = store, update_fn, persist_s
        self.map: Optional[dict] = initial  # empty: the cleared map the module starts from (Store.clear_table_map)
        self.swept_ts = 0.0  # last whole-table sweep
        self.epoch = float((initial or {}).get("cleared_ts") or time.time())  # the fold over the log starts here
        self._log = hasattr(store, "log_observation")
        self._gen = 0  # bumped by clear(): an update still running on the old map is thrown away
        params = inspect.signature(update_fn).parameters
        self._takes_recognised, self._takes_seg = "recognised" in params, "seg" in params
        self._newest = None  # newest droppable frame waiting for the worker
        self._keep = deque()  # (frame, future): frames that must all be applied, in order (sweep)
        self._keep_current: Optional[asyncio.Future] = None  # the keep frame being applied right now
        self._task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._saved_at, self._dirty = 0.0, False
        self.stats = {"updates": 0, "dropped": 0, "errors": 0, "last_ms": None, "last_source": None,
                      "last_error": None, "logged": 0, "log_errors": 0}

    # ---- feeding --------------------------------------------------------------------------------------

    def offer(self, rgb, depth, pose: dict, intr: dict, source: str, recognised=None, seg=None,
              keep: bool = False) -> None:
        """
        Queue one frame for update_map and return at once. keep=False: replaces a frame still waiting.
        recognised / seg: that frame's recognise_all / segment_full output when the caller has it (idle loop),
        so update_map does not run them again.
        """
        frame = (rgb, depth, dict(pose), dict(intr), source, recognised, seg, time.time())
        if keep:
            self._keep.append((frame, asyncio.get_event_loop().create_future()))
        else:
            if self._newest is not None:
                self.stats["dropped"] += 1
            self._newest = frame
        if self._task is None or self._task.done():
            self._task = asyncio.get_event_loop().create_task(self._drain())

    def busy(self) -> bool:
        """An update is running or queued: a caller can skip work (e.g. a pose read) for a frame that would wait."""
        return self._task is not None and not self._task.done()

    async def drain(self) -> None:
        """
        Wait until every keep=True frame offered so far has been applied (after a sweep, before picking from the
        map). Droppable frames don't count: with pictures arriving all the time the worker may never be idle.
        """
        futs = [f for _, f in self._keep] + ([self._keep_current] if self._keep_current is not None else [])
        if futs:
            await asyncio.gather(*futs)

    async def _drain(self) -> None:
        """Worker: apply queued frames one by one, sweep frames first, then the newest droppable one."""
        while self._keep or self._newest is not None:
            if self._keep:
                frame, self._keep_current = self._keep.popleft()
            else:
                (frame, self._newest), self._keep_current = (self._newest, None), None
            try:
                await self._apply(frame)
            finally:
                if self._keep_current is not None and not self._keep_current.done():
                    self._keep_current.set_result(None)  # applied or failed: either way drain() may go on
                self._keep_current = None

    async def _apply(self, frame: tuple) -> None:
        """One update_map call in a worker thread; the result replaces the map, a failure is counted and skipped."""
        rgb, depth, pose, intr, source, recs, seg, taken = frame
        kw = {"recognised": recs} if recs is not None and self._takes_recognised else {}
        if seg is not None and self._takes_seg:
            kw["seg"] = seg
        snapshot, gen = self.map, self._gen  # map: replaced, never mutated in place, so stable in the thread
        t0 = time.monotonic()
        try:
            m = await asyncio.to_thread(lambda: self.update_fn(copy.deepcopy(snapshot), rgb, depth, pose, intr,
                                                               self.store, source, **kw))
        except Exception as e:
            self.stats["errors"] += 1
            self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
            LOGGER.warning(f"live map: update_map failed on a {source} frame: {self.stats['last_error']}")
            return
        if gen != self._gen:
            return  # the map was cleared while this frame (taken before the clear) was being applied
        if m is not None:
            m["ts"] = time.time()  # last update; each object keeps its own seen_ts
            m["swept_ts"] = self.swept_ts
            self.map = m
        self.stats.update(updates=self.stats["updates"] + 1, last_source=source,
                          last_ms=int((time.monotonic() - t0) * 1000))
        lu = (m or {}).get("last_update") or {}
        if self._log and not lu.get("skipped"):  # a skipped frame (same pose, < 0.5 s) added nothing to the map
            await asyncio.to_thread(self._log_observation, rgb, depth, pose, intr, source, taken, lu)
        self._dirty = True
        if self._persist_task is None or self._persist_task.done():
            self._persist_task = asyncio.get_event_loop().create_task(self._persist_loop())

    def _log_observation(self, rgb, depth, pose: dict, intr: dict, source: str, taken: float, lu: dict) -> None:
        """
        Append the frame just folded in to the Store's observation log (worker thread). What it saw comes from
        update_map's last_update (AGENT3: detections, square, intruders); the images are made small here.
        """
        try:
            png, scale = _small_depth(depth) if depth is not None else (None, None)
            doc = {"ts": taken, "source": source, "pose": pose, "intr": intr,
                   "detections": lu.get("detections") or [], "detected": lu.get("detected"),
                   "square": lu.get("square"), "intruders": lu.get("intruders") or [],
                   "hand_in_view": lu.get("hand_in_view"), "dropped": lu.get("dropped") or [],
                   "epoch": self.epoch, "map_version": f"{int(self.epoch * 1000)}:{self.stats['updates']}",
                   "depth_scale": scale}
            self.store.log_observation(doc, _small_jpeg(rgb) if rgb is not None else None, png)
            self.stats["logged"] += 1
        except Exception as e:
            self.stats["log_errors"] += 1
            LOGGER.warning(f"live map: observation not logged: {type(e).__name__}: {e}")

    async def rebuild(self, since: float, fold_fn) -> dict:
        """
        Replace the map with AGENT3's fold over the logged observations since `since` (e.g. after a crash):
        pending frames are dropped and an update in flight is thrown away, as for clear(). Returns the counts.
        """
        obs = await asyncio.to_thread(self.store.list_observations, since)
        m = await asyncio.to_thread(fold_fn, obs)
        self._gen += 1
        self._newest = None
        m = {**(m or {"objects": []}), "ts": time.time(), "epoch": since, "rebuilt_from": since, "swept_ts": 0.0}
        self.map, self.epoch, self.swept_ts = m, float(since), 0.0
        await asyncio.to_thread(self.store.put_table_map, self.map, True)
        self._dirty, self._saved_at = False, time.time()
        return {"observations": len(obs), "objects": len(m.get("objects") or [])}

    # ---- saving ---------------------------------------------------------------------------------------

    async def _persist_loop(self) -> None:
        """Save at most every persist_s; updates landing meanwhile ride along with the next save."""
        while self._dirty:
            await asyncio.sleep(max(0.0, self._saved_at + self.persist_s - time.time()))
            self._dirty = False
            try:
                await asyncio.to_thread(self.store.put_table_map, self.map, False)  # live saves skip the history
            except Exception as e:
                self._dirty = True  # retried at the next round
                LOGGER.warning(f"live map: save failed: {type(e).__name__}: {e}")
            self._saved_at = time.time()

    async def clear(self, reason: str) -> dict:
        """
        Forget the table (someone rearranged it): pending frames are dropped, an update in flight is thrown away,
        the current map goes to the Store's history and the map restarts empty. Returns the empty map
        (cleared_ts, cleared_objects).
        """
        self._gen += 1
        self._newest = None
        for _, f in self._keep:
            if not f.done():
                f.cancel()
        self._keep.clear()
        if self.map is not None:  # the in-memory map is up to persist_s newer than the saved one: that one goes to history
            await asyncio.to_thread(self.store.put_table_map, self.map, False)
        self.map = await asyncio.to_thread(self.store.clear_table_map, reason)
        self.swept_ts, self._dirty, self._saved_at = 0.0, False, time.time()
        self.epoch = float(self.map.get("cleared_ts") or time.time())  # the fold starts again here; the log is kept
        return self.map

    async def mark_swept(self) -> None:
        """A whole-table sweep was just applied: the map now covers the whole table. Saved at once, with history."""
        self.swept_ts = time.time()
        if self.map is not None:
            self.map = {**self.map, "swept_ts": self.swept_ts}
            await asyncio.to_thread(self.store.put_table_map, self.map, True)
            self._saved_at = time.time()

    # ---- reading --------------------------------------------------------------------------------------

    def seen_ago(self, obj: dict, now: Optional[float] = None) -> float:
        """Seconds since a map object was last seen (its seen_ts, else the map's last update)."""
        now = now or time.time()
        return now - float(obj.get("seen_ts") or (self.map or {}).get("ts") or 0.0)

    def light(self) -> dict:
        """The map without per-object embeddings, for the dashboard."""
        m = dict(self.map or {})
        m["objects"] = [{k: v for k, v in o.items() if k != "emb"} for o in m.get("objects") or []]
        return m

    def summary(self) -> dict:
        """Status: size, freshness, update cost and drops."""
        now = time.time()
        m = self.map or {}
        return {"live": True, "objects": len(m.get("objects") or []),
                "updated_s_ago": round(now - m["ts"], 1) if m.get("ts") else None,
                "cleared_s_ago": round(now - m["cleared_ts"], 1) if m.get("cleared_ts") else None,
                "epoch": self.epoch,
                "swept_s_ago": round(now - self.swept_ts, 1) if self.swept_ts else None,
                "pending": len(self._keep) + (self._newest is not None), **self.stats}

    async def close(self) -> None:
        """Stop the worker and save what is unsaved."""
        for t in (self._task, self._persist_task):
            if t is not None and not t.done():
                t.cancel()
        for _, f in self._keep:  # a drain() still waiting must not hang on frames that will never be applied
            if not f.done():
                f.cancel()
        self._keep.clear()
        if self._dirty and self.map is not None:
            try:
                await asyncio.to_thread(self.store.put_table_map, self.map, False)
            except Exception as e:
                LOGGER.warning(f"live map: final save failed: {e}")
