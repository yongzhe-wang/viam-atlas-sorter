"""
The library learns while it runs (SORTER_TEACH=1): {"teach": "m&m bag"} and every delivered grasp add views to it.

WHY: on the arm an object that was flipped or seen from another height was not recognised (the back of the KIND bar
came back as granola bar, tape was not known from high up) {VIAM CENTERAL 2026-10-30 "物体一翻面或换高度就认不出"}.
A view is a CLIP embedding plus a small JPEG in Atlas `views`, so $vectorSearch finds it on the next look.

teach(): the chosen object from where the arm stands, two look poses beside it at z 400 and one at z 580. The arm stays
at z >= 400 and touches nothing; EStop stops it. learn_from_grasp(): the crop that found a delivered object.
The library data goes through Store.add_learned_view (duplicates, cap per label).
"""
import asyncio
import io
import math
import os
import time
from typing import Optional

import numpy as np
from PIL import Image

from sorter.store import LEARN_MAX_VIEWS

SIDE_MM = 80.0  # a look pose this far beside the one straight above the object
MATCH_MM = 60.0  # in a later frame, the object within this of where the first frame saw it is the same object
POINT_PX = 80.0  # a click this close to an object's box centre selects it
MIN_RADIUS_MM = 150.0  # as executor.sweep_table: no look pose nearer the arm's base
SEARCHABLE_WAIT_S = 10.0  # Atlas indexes a new view within seconds; a teach reports how long it took
VIEW_JPEG_WIDTH = 256  # the 512 MB free tier: a view is an embedding and a small picture
AMBIGUOUS_MARGIN = 0.02  # as clip_embed.AMBIGUOUS_MARGIN: a closer pick is not sure enough to name a new label
HIGH_Z = 580.0  # as executor.SWEEP_Z: the whole table in view, objects small: the view recognition missed from high up


def enabled() -> bool:
    return os.environ.get("SORTER_TEACH", "0") == "1"


def view_jpeg(crop: np.ndarray) -> bytes:
    """An object's crop as the small JPEG a view keeps."""
    im = Image.fromarray(crop).convert("RGB")
    if im.width > VIEW_JPEG_WIDTH:
        im = im.resize((VIEW_JPEG_WIDTH, max(1, round(im.height * VIEW_JPEG_WIDTH / im.width))))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=75)
    return out.getvalue()


def resolve_label(rows: list, name: str, alias_of: Optional[str] = None) -> tuple:
    """
    (label, aliases, error) for what is being taught, from Store.list_labels() rows.

    A name the library knows adds views to that object, whether it is a label or an alias. alias_of makes the name
    another word for a known object; every other name becomes a new label.
    """
    owner = {n.lower(): row[0] for row in rows for n in row}
    if alias_of:
        label = owner.get(alias_of.strip().lower())
        if label is None:
            return None, [], f"{alias_of!r} is not in the library, so {name!r} cannot be another name for it"
        return label, ([] if name.lower() == label.lower() else [name]), None
    return owner.get(name.lower(), name), [], None


def choose(recs: list, point: Optional[tuple], rank) -> tuple:
    """
    (the record of the object to teach, why). point: the pixel the person clicked; without one, the first of `rank`
    (a callable giving [(Obj, score)] best first). None if the click hit nothing or nothing is in view.
    """
    if not recs:
        return None, "no whole object in view from here"
    if point is not None:
        def distance(r) -> float:
            u0, v0, u1, v1 = r["obj"].bbox
            inside = u0 <= point[0] <= u1 and v0 <= point[1] <= v1
            return 0.0 if inside else math.hypot((u0 + u1) / 2 - point[0], (v0 + v1) / 2 - point[1])
        rec = min(recs, key=distance)
        return (rec, "clicked") if distance(rec) <= POINT_PX else (None, "no object where you clicked")
    if len(recs) == 1:
        return recs[0], "the only object in view"
    first = rank()[0][0]
    return next(r for r in recs if r["obj"] is first), "best match for the name"


def object_xy(obj, pose: dict, intr: dict, perception) -> tuple:
    """Table xy of an object's top in a frame."""
    u, v = obj.top_centroid_px or obj.centroid_px
    return perception.world_xy(pose, intr, u, v, obj.top_depth_mm)


def look_poses(xy: tuple, intr: dict, ppa, perception) -> list:
    """
    (x, y, z) flange poses to see the object at (x, y) from: two beside it at ALIGN_Z, then one at z 580.

    A pose is planned with the object at the image centre, since the camera is not on the jaw axis. A side the arm
    cannot reach gives way to the next one; high up, reach is shorter, so that pose moves toward the base, and the wide
    view there still has the object.
    """
    def over(z: float) -> tuple:
        ox, oy = perception.world_xy({"x": 0.0, "y": 0.0, "theta": 0.0}, intr, intr["cx"], intr["cy"],
                                     ppa.TABLE_DEPTH_MM + z - ppa.ALIGN_Z)
        return xy[0] - ox, xy[1] - oy

    def fits(x: float, y: float, z: float) -> bool:
        return ppa.reachable(x, y, z) and math.hypot(x, y) >= MIN_RADIUS_MM

    x, y = over(ppa.ALIGN_Z)
    sides = [(x + dx, y + dy) for dx, dy in ((0.0, SIDE_MM), (0.0, -SIDE_MM), (SIDE_MM, 0.0), (-SIDE_MM, 0.0))]
    poses = [(px, py, ppa.ALIGN_Z) for px, py in sides if fits(px, py, ppa.ALIGN_Z)][:2]
    x, y = over(HIGH_Z)
    r = math.hypot(x, y) or 1.0
    for pull in (0.0, 50.0, 100.0, 150.0):
        px, py = x * (1 - pull / r), y * (1 - pull / r)
        if fits(px, py, HIGH_Z):
            poses.append((px, py, HIGH_Z))
            break
    return poses


def _same_object(seg, xy: tuple, pose: dict, intr: dict, perception):
    """The whole object in this frame nearest where the first frame saw ours, or None."""
    near = [(math.dist(object_xy(o, pose, intr, perception), xy), o) for o in seg.objects if not o.touches_edge]
    near = [t for t in near if t[0] <= MATCH_MM]
    return min(near, key=lambda t: t[0])[1] if near else None


def _add_view(store, label: str, emb, obj, pose: dict, aliases: list, ts: float, meta: dict = None,
              dedupe: bool = True) -> dict:
    """One teach view into the library; the person chose this object, so only its own label's views count as repeats."""
    return store.add_learned_view(label, emb, view_jpeg(obj.crop), {
        "source": "teach", "aliases": aliases, "colour": obj.colour, "height_mm": round(float(obj.height_mm), 1),
        "pose": {k: round(float(pose[k]), 1) for k in ("x", "y", "z")}, "taught_ts": ts, **(meta or {})},
        same_label_only=True, dedupe=dedupe)


def _frame_row(name: str, pose: dict, res: dict) -> dict:
    return {"view": name, "z": round(float(pose["z"])), **{k: res[k] for k in ("skipped", "cos") if res.get(k) is not None},
            "added": bool(res.get("id"))}


async def _wait_searchable(store, view_id: str, emb) -> tuple:
    """(seconds until $vectorSearch returned the new view or None, its score): Atlas indexes it a moment after the write."""
    t0 = time.monotonic()
    while time.monotonic() - t0 <= SEARCHABLE_WAIT_S:
        hits = await asyncio.to_thread(store.search_views, emb, 5)
        hit = next((h for h in hits if h.get("_id") == view_id), None)
        if hit is not None:
            return round(time.monotonic() - t0, 1), round(float(hit["score"]), 3)
        await asyncio.sleep(1.0)
    return None, None


async def teach(rig, rt, store, name: str, start: dict, alias_of: Optional[str] = None, progress=None) -> dict:
    """
    Teach the object of `start` ({"rec": clip_embed record, "pose": where that frame was taken}) under `name`.

    Returns {"status", "label", "views_added", "views_skipped", "frames", "searchable_s", ...}. status is "done" even
    when every view was a repeat (the library already had this), "estop" when the emergency stop ended it; a refused
    pose is left out. Always ends at the observe pose (not after an EStop).
    """
    t0, now = time.monotonic(), time.time()
    ppa, perception, executor = rt.ppa, rt.perception, rt.executor
    label, aliases, error = resolve_label(await asyncio.to_thread(store.list_labels), name, alias_of)
    if error:
        return {"status": "failed", "fail_phase": "teach", "fail_reason": error}
    intr = executor.intr(rig)
    obj, pose = start["rec"]["obj"], start["pose"]
    xy = object_xy(obj, pose, intr, perception)
    meta = {"xy": [round(float(c), 1) for c in xy]}
    frames, added = [], []

    async def keep(row: str, o, emb, at: dict) -> None:
        # A new name lives on a view: the first one stays even if the picture is a repeat, or the name would be lost
        res = await asyncio.to_thread(_add_view, store, label, emb, o, at, aliases, now, meta, not (aliases and not frames))
        frames.append(_frame_row(row, at, res))
        if res.get("id"):
            added.append((res["id"], emb))

    await keep("current", obj, start["rec"]["emb"], pose)
    poses = look_poses(xy, intr, ppa, perception)
    status, reason = "done", None
    try:
        for k, (x, y, z) in enumerate(poses, 1):
            if progress:
                progress(k, len(poses))
            if ppa.ESTOP["on"]:
                raise ppa.EStop(ppa.ESTOP_STATUS)
            if z < ppa.ALIGN_Z:  # look_poses never asks for it: the promise is that a teach stays up here
                raise ppa.MoveRefused(f"teach stays at z >= {ppa.ALIGN_Z:.0f}, asked for {z:.0f}")
            row = f"look {k}"
            try:
                await ppa.travel(rig, x, y, theta=0.0, z=z)
            except ppa.EStop:
                raise
            except ppa.MoveRefused as e:
                frames.append({"view": row, "z": round(z), "added": False, "skipped": f"refused: {e}"[:120]})
                continue
            rgb, depth = await rig.frame()
            at = executor.pose_dict(await rig.pose())
            rig.record("teach", rgb, k=k, n=len(poses))
            seg = await asyncio.to_thread(perception.segment_full, rgb, depth)
            if seg.intruders:
                frames.append({"view": row, "z": round(z), "added": False, "skipped": "hand in view"})
                continue
            o = await asyncio.to_thread(_same_object, seg, xy, at, intr, perception)
            if o is None:
                frames.append({"view": row, "z": round(z), "added": False, "skipped": "object not in this view"})
                continue
            emb = (await asyncio.to_thread(rt.clip_embed.embed_images, [o.crop]))[0]
            await keep(row, o, emb, at)
    except ppa.EStop:
        status, reason = "estop", ppa.ESTOP_STATUS
    finally:
        await executor.back_to_observe(rig)
    searchable_s, top = await _wait_searchable(store, *added[-1]) if added else (None, None)
    return {"status": status, "fail_reason": reason, "label": label, "alias_of": label if aliases else None,
            "colour": obj.colour, "views_added": len(added), "views_skipped": len(frames) - len(added), "frames": frames,
            "searchable_s": searchable_s, "top_score": top, "duration_s": round(time.monotonic() - t0, 1)}


def label_for_grasp(rows: list, text: str, recognised: Optional[str], margin: Optional[float], clip_embed) -> tuple:
    """
    (label, None) a delivered object's new view goes under, or (None, why not).

    Delivered only says it reached the square, not that it was what was asked for, so a view must not teach the library
    a mistake. Recognised as what the words name, or the words mean nothing to the library: the recognised name.
    Not recognised: the name the person used, if the pick was clear. Recognised as something else than the words name:
    nothing, the {"teach"} command is there to settle it.
    """
    q = clip_embed.clean_query(text or "")
    named = {n.lower(): row[0] for row in rows for n in row}.get(q)
    if named and recognised and named != recognised:
        return None, f"asked for {named!r}, recognised {recognised!r}"
    if recognised or named:
        return recognised or named, None
    words = q.split()
    if not words or all(w in clip_embed.COLOURS or w in clip_embed.FILLER for w in words):
        return None, f"{q!r} does not name an object"
    if margin is None or margin < AMBIGUOUS_MARGIN:
        return None, f"{q!r} is new and the pick was not clear"
    return q, None


def learn_from_grasp(store, clip_embed, doc: dict, obj, text: str) -> Optional[dict]:
    """
    A delivered, checked grasp -> the crop that found the object as a new view, with the primitive and grip that worked.
    Returns what happened ({"label", "view"} or {"skipped"}); None when there is nothing to learn from. Never raises:
    the grasp already succeeded.
    """
    if obj is None or doc.get("status") != "done" or not doc.get("success"):
        return None
    try:
        label, why = label_for_grasp(store.list_labels(), text, getattr(obj, "recognised", None), doc.get("margin"), clip_embed)
        if label is None:
            return {"skipped": why}
        emb = doc.get("embedding")
        emb = clip_embed.embed_images([obj.crop])[0] if emb is None else emb
        res = store.add_learned_view(label, emb, view_jpeg(obj.crop), {
            "source": "grasp", "colour": obj.colour, "height_mm": round(float(obj.height_mm), 1), "attempt": doc.get("_id"),
            "primitive": (doc.get("grip") or {}).get("primitive"), "grip": doc.get("grip")}, cap=LEARN_MAX_VIEWS)
        return {"label": label, "view": res["id"]} if res.get("id") else {"label": label, "skipped": res["skipped"]}
    except Exception as e:
        return {"skipped": f"{type(e).__name__}: {e}"[:200]}
