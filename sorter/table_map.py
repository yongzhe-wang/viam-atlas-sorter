"""
Whole-table object map from a fast sweep: every frame recognised, sightings merged by table position.

WHY: the robot should find a named object anywhere on the table, not only in the observe view or where it was
last remembered. AGENT1 sweeps 4-6 wide frames; this module turns them into one list of objects with table xy,
and answers a query against that list with the same rules as clip_embed.best().

Flow: frames {id, rgb, depth, pose, intrinsics?} -> recognise_all per frame -> table xy per object (pose + pixel
+ depth) -> merge across frames (same label within MERGE_XY_MM is one object) -> map.

Map doc (Store.save_map / latest_map, AGENT 2), latest wins:
  {ts, frames: [{id, pose}], objects: [{id, label, names, xy, height_mm, score, colour, colours, size_mm, frame_ids, emb}],
   square?: drop_square entry {xy, size_mm, yaw_deg, corners, partial, confidence, ...}}   (drop zone: white
   interior of the blue tape, estimated whole even from border-cut views; partial=True: not all four tape edges seen
   yet, so look at it fully before placing)
label/names are None/[] for objects the library does not know; emb (512 floats) lets open-vocabulary queries run
on the map without re-embedding crops.
"""
import base64
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sorter import bucket, camera_model, clip_embed, drop_square, perception

MERGE_XY_MM = 40.0  # two sightings this close on the table can be the same object
MERGE_DH_MM = 15.0  # ... if their heights agree
MERGE_COS = 0.70  # ... and, when not both labelled, their crops look alike (same as label_scan grouping)


def _detections(frames: list, store) -> tuple:
    """Every target of every frame with table xy, size in mm and embedding; drop-square sightings; ms per frame."""
    out, squares, ms = [], [], []
    for f in frames:
        t = time.perf_counter()
        intr = {**perception.INTRINSICS, **(f.get("intrinsics") or {})}
        sq = perception.find_drop_square(f["rgb"], f["depth"])
        if sq:
            squares.append({**drop_square.observe(sq, f["pose"], intr), "frame": f["id"]})
        for r in clip_embed.recognise_all(f["rgb"], f["depth"], store):
            o = r["obj"]
            xy = camera_model.world_xy_full(f["pose"], intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
            mm_px = o.top_depth_mm / intr["fx"]
            out.append({"label": r["label"], "names": r["names"], "xy": xy, "height_mm": o.height_mm,
                        "score": r["score"], "colour": o.colour, "colours": dict(getattr(o, "colours", {}) or {}),
                        "size_mm": (o.length_px * mm_px, o.width_px * mm_px),
                        "emb": r["emb"], "frame": f["id"]})
        ms.append((time.perf_counter() - t) * 1000)
    return out, squares, ms


def _same(a: dict, m: dict) -> bool:
    """Is detection `a` the same physical object as map object `m`?"""
    if math.dist(a["xy"], m["xy"]) > MERGE_XY_MM or abs(a["height_mm"] - m["height_mm"]) > MERGE_DH_MM:
        return False
    if a["label"] and m["label"]:
        return a["label"] == m["label"]
    # One side unrecognised (or both): require colour and appearance to agree, as for scan grouping
    if not (clip_embed.colour_ok({a["colour"]}, m["colour"]) or clip_embed.colour_ok({m["colour"]}, a["colour"])):
        return False
    if a.get("emb") is None or m.get("emb") is None:
        return False  # no appearance to compare (entry from a map saved without embeddings)
    return float(a["emb"] @ m["emb"]) >= MERGE_COS


def _label_fits_colours(m: dict) -> bool:
    """The colour words in the object's label/aliases are present in its colour shares (no colour word: True)."""
    words = {w for n in [m["label"], *m["names"]] for w in clip_embed.clean_query(n).split()} & clip_embed.COLOURS
    obj = SimpleNamespace(colour=m["colour"], colours=m["colours"])
    return not words or clip_embed.colour_present(words, obj)


def _resolve_conflicts(objs: list) -> list:
    """
    Two differently labelled objects at the same place and height are one thing recognised two ways.

    WHY: on the 32-frame sweep the red screwdriver was recognised as "red screwdriver" in 2 frames and as
    "blue screwdriver" in 2 others (both handles are dominantly black), leaving a phantom blue screwdriver on the
    map. Keep the label whose colour words fit the object's colour shares; if both or neither fit, the one seen
    in more frames. The loser's sightings join the winner.
    """
    keep = []
    for m in sorted(objs, key=lambda m: -len(m["frame_ids"])):
        twin = next((k for k in keep if k["label"] and m["label"] and k["label"] != m["label"]
                     and math.dist(np.mean(k["_xy"], 0), np.mean(m["_xy"], 0)) <= MERGE_XY_MM
                     and abs(np.median(k["_h"]) - np.median(m["_h"])) <= MERGE_DH_MM), None)
        if twin is None:
            keep.append(m)
            continue
        for c, f in m["colours"].items():
            twin["colours"][c] = max(twin["colours"].get(c, 0.0), f)
        if _label_fits_colours(m) and not _label_fits_colours(twin):
            twin["label"], twin["names"], twin["score"] = m["label"], m["names"], m["score"]
        for key in ("frame_ids", "_xy", "_w", "_h", "_e"):
            twin[key] += m[key]
    return keep


def map_from_frames(frames: list, store=None) -> dict:
    """
    One map of the table from a sweep. Recognised detections seed objects first (highest confidence first), so an
    unrecognised view of a known object joins it instead of becoming a second object.
    Returns the map doc plus "timing_ms" (per frame) for the caller's logs.
    """
    dets, squares, ms = _detections(frames, store)
    dets.sort(key=lambda d: (d["label"] is None, -d["score"]))
    objs = []
    for d in dets:
        m = next((m for m in objs if _same(d, m)), None)
        if m is None:
            objs.append({**d, "colours": dict(d["colours"]), "frame_ids": [d["frame"]], "_xy": [d["xy"]], "_w": [d["score"] + 0.1],
                         "_h": [d["height_mm"]], "_e": [d["emb"]], "names": list(d["names"])})
            continue
        m["frame_ids"].append(d["frame"])
        m["_xy"].append(d["xy"]); m["_w"].append(d["score"] + 0.1); m["_h"].append(d["height_mm"]); m["_e"].append(d["emb"])
        m["score"] = max(m["score"], d["score"])
        for c, f in d["colours"].items():  # strongest share seen in any view
            m["colours"][c] = max(m["colours"].get(c, 0.0), f)
        if m["label"] is None and d["label"]:
            m["label"], m["names"] = d["label"], list(d["names"])
    objs = _resolve_conflicts(objs)
    out = []
    for i, m in enumerate(objs):
        w = np.array(m["_w"])
        xy = (np.array(m["_xy"]) * w[:, None]).sum(0) / w.sum()
        e = np.mean(m["_e"], axis=0)
        out.append({"id": i, "label": m["label"], "names": m["names"], "xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)],
                    "height_mm": round(float(np.median(m["_h"])), 1), "score": round(float(m["score"]), 3),
                    "colour": m["colour"], "colours": m["colours"], "size_mm": [round(float(x), 1) for x in m["size_mm"]],
                    "frame_ids": sorted(set(m["frame_ids"])),
                    "emb": [round(float(x), 4) for x in e / np.linalg.norm(e)]})  # 4 decimals: ~3x smaller doc
    doc = {"ts": time.time(), "frames": [{"id": f["id"], "pose": f["pose"]} for f in frames], "objects": out,
           "timing_ms": [round(x) for x in ms]}
    sq = None
    for q in squares:  # every view of the square, whole or border-cut, folded as the live map does
        sq = drop_square.update(sq, q, doc["ts"])
    if sq is not None:
        doc["square"] = {**sq, "frame_ids": [q["frame"] for q in squares]}
    _mark_in_square(doc)
    return doc


def _mark_in_square(tmap: dict) -> None:
    """
    Flag every object whose xy lies in the drop square's estimated full interior (in_square=True), partial or not.

    WHY: the module filters objects already in the square out of a single frame, but a query against the map would
    otherwise pick the yellow block out of the square again after it was delivered; with the square only partly in
    view the visible part is too small, so the estimated rectangle (drop_square) is used.
    {VIAM CENTERAL 2026-10-29 ""OBJECTS IN THE SQUARE" USES THE ESTIMATED FULL INTERIOR EVEN WHEN PARTIAL"}
    """
    sq, bk = tmap.get("square"), tmap.get("bucket")
    for e in tmap["objects"]:
        # The drop target is the bucket when one is on the map (SORTER_TARGET=bucket): delivered = inside its mouth
        e["in_square"] = bucket.contains(bk, e["xy"]) if bk and bucket.TARGET == "bucket" else drop_square.contains(sq, e["xy"])


# ---- dynamic map: every frame the arm takes updates it ------------------------------------------------------
SKIP_S = 0.3  # a frame this soon after the last one at the same pose adds nothing: skip (saves the CLIP batch);
# under the 0.5 s look period, so 2 Hz frames go into the map
SAME_POSE_MM, SAME_POSE_DEG = 5.0, 2.0
MOVE_CONF = 0.5  # a confident sighting of a label seen elsewhere moves that object (one of each on the table)
MISSES_TO_DROP = 2  # expected in view, not detected, this many frames in a row -> removed from the map
FOOT_MARGIN_PX = 60  # an entry projected closer than this to the image border is not expected to be detected
XY_ALPHA = 0.5  # weight of a new sighting in the entry's xy (objects that move should follow quickly)


EMB_DIM = 512  # open_clip ViT-B-32 image embedding size (clip_embed); entries without one get zeros in queries


def empty_map() -> dict:
    """A new map with every field update_map / apply_observation / best_on_map use: start (or clear) with this."""
    return {"ts": time.time(), "objects": [], "next_id": 0, "last_update": None}


def _ensure_map(tmap: dict) -> dict:
    """
    Fill in, in place, whatever the fold needs that a map made elsewhere may lack; returns the map.

    WHY: AGENT 2's module started from a hand-made {"objects": []} (startup, clear_map, a saved map read back), and
    update_map raised KeyError: 'next_id' on every frame on sfhack4 (265 times). Any of {}, a cleared map, a
    map_from_frames() map or a stored one now works: next_id continues after the highest id, entries get their
    bookkeeping fields (n_seen from frame_ids), an entry without an embedding gets None (it then merges by label only).
    {VIAM CENTERAL 2026-10-29 "UPDATE_MAP FAILS ON EVERY FRAME WITH KEYERROR: 'NEXT_ID' (265 TIMES)"}
    [CONFIDENCE: CONFIRMED 90% — tests/test_table_map.py feeds {}, empty_map(), cleared and survey-style maps]
    """
    objs = [e for e in tmap.get("objects") or [] if isinstance(e, dict) and e.get("xy") is not None]
    next_id = max([int(e["id"]) + 1 for e in objs if e.get("id") is not None] + [int(tmap.get("next_id") or 0)])
    for e in objs:
        if e.get("id") is None:
            e["id"], next_id = next_id, next_id + 1
        e["xy"] = [float(e["xy"][0]), float(e["xy"][1])]
        for k, v in (("label", None), ("height_mm", 0.0), ("score", 0.0), ("colour", None), ("partial", False),
                     ("misses", 0), ("emb", None)):
            e.setdefault(k, v)
        for k in ("names", "frame_ids"):
            e[k] = list(e.get(k) or [])
        e["colours"] = dict(e.get("colours") or {})
        e["size_mm"] = list(e.get("size_mm") or [0.0, 0.0])
        e.setdefault("n_seen", len(e["frame_ids"]))
    tmap["objects"], tmap["next_id"] = objs, next_id
    if tmap.get("square") is not None:
        sq = drop_square.from_legacy(tmap["square"])  # a square saved before drop_square: rebuilt as a model entry
        if sq is None:
            tmap.pop("square")
        else:
            tmap["square"] = sq
    tmap.setdefault("last_update", None)
    tmap.setdefault("ts", 0.0)
    return tmap


def _same_pose(a: dict, b: dict) -> bool:
    return (math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"])) <= SAME_POSE_MM
            and abs(float(a.get("theta", 0)) - float(b.get("theta", 0))) <= SAME_POSE_DEG)


DEPTH_CELL_PX = 16  # negative evidence reads depth validity per 16x16 px cell, kept in every observation
# Fields apply_observation() reads from a record (observe_frame() writes all of them)
# ("square" may be absent: a None field is dropped when a record comes back through a Viam Struct, AGENT 2 c1e51c2)
REC_KEYS = ("ts", "source", "pose", "intr", "shape", "table_depth_mm", "detections", "intruders", "depth_valid")


def _valid_grid(depth: np.ndarray) -> dict:
    """Depth validity per DEPTH_CELL_PX cell (at least half the pixels have depth), bit-packed for the record."""
    c = DEPTH_CELL_PX
    rows, cols = depth.shape[0] // c, depth.shape[1] // c
    g = (depth[:rows * c, :cols * c] > 0).reshape(rows, c, cols, c).mean(axis=(1, 3)) >= 0.5
    return {"cell": c, "rows": rows, "cols": cols, "bits": base64.b64encode(np.packbits(g).tobytes()).decode()}


def _grid(rec: dict) -> np.ndarray:
    """The observation's depth-validity grid back as a bool array (rows x cols)."""
    g = rec["depth_valid"]
    rows, cols = int(g["rows"]), int(g["cols"])  # ints come back as floats through a Viam Struct (DoCommand export)
    bits = np.unpackbits(np.frombuffer(base64.b64decode(g["bits"]), np.uint8))
    return bits[:rows * cols].reshape(rows, cols).astype(bool)


def _expected_visible(e: dict, rec: dict, grid: np.ndarray, xy=None) -> bool:
    """
    Should this map entry be clearly detected in this observation? Its whole extent projects inside the view (minus
    a margin), with valid depth at its centre (not the blind range, not the held object), not under the fingertips.

    The whole extent, not just the centre: the 17 cm KIND bar and the 19x14 cm square had their centres inside the
    view while their bodies were cut by the edge, were not detected as whole, and got wrongly dropped.
    """
    H, W = rec["shape"]
    pose, intr = rec["pose"], rec["intr"]
    z = max(rec["table_depth_mm"] - e.get("height_mm", 0.0), 50.0)
    cx, cy = xy or e["xy"]
    r = 0.5 * max(e.get("size_mm") or [0.0, 0.0])
    m = FOOT_MARGIN_PX
    tilted = camera_model.is_tilted(pose)

    def px_of(x, y):
        # Straight down: the camera distance of the top is the table's minus the height. Tilted: project the 3-D point.
        if tilted:
            return camera_model.project(pose, intr, (x, y, e.get("height_mm", 0.0)))[:2]
        return perception.project_xy(pose, intr, (x, y), z)

    for px, py in ((cx, cy), (cx - r, cy - r), (cx - r, cy + r), (cx + r, cy - r), (cx + r, cy + r)):
        pu, pv = px_of(px, py)
        if not (m <= pu < W - m and m <= pv < H - m):
            return False
    u, v = px_of(cx, cy)
    if any(u0 <= u < u1 and v0 <= v < v1 for u0, v0, u1, v1 in perception.FINGER_BOXES):
        return False
    row, col = int(v) // DEPTH_CELL_PX, int(u) // DEPTH_CELL_PX
    return bool(0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col])


def observe_frame(rgb: np.ndarray, depth: np.ndarray, pose: dict, intr: dict = None, store=None, recognised: list = None,
                  seg=None, source: str = "frame", now: float = None) -> dict:
    """
    One camera frame -> observation record: everything the map takes from it, JSON-safe (AGENT 2 logs these).

    WHY: every frame is kept and the map is a fold over the log; the live update_map() and a replay
    (fold_observations) run the same apply_observation() on the same record, so they build the same map, and a
    replay needs no images and no CLIP. Detections carry their CLIP embedding as float16 (base64, ~1.4 kB) because
    unknown objects merge by appearance; depth survives only as the per-cell validity grid negative evidence reads.
    {VIAM CENTERAL 2026-10-29 "MAKE THE MAP A DETERMINISTIC FOLD OVER IT … USING THE STORED DETECTIONS (NO RE-RECOGNITION)"}
    [CONFIDENCE: CONFIRMED 90% — live vs replay compared on the 89-frame scan, see fold_observations]
    recognised / seg as for update_map(). Record: {ts, source, pose, intr, shape, table_depth_mm, detections:
    [{label, names, score, xy, height_mm, bbox, colour, colours, size_mm, partial, emb16}], square: drop_square.observe()
    {hull_xy, real, cut, partial, xy, size_mm} | None, intruders: [bbox], depth_valid}.
    """
    intr = {**perception.INTRINSICS, **(intr or {})}
    seg = seg if seg is not None else perception.segment_full(rgb, depth, pose=pose, intr=intr)
    recs = recognised if recognised is not None else clip_embed.recognise_all(rgb, depth, store, seg=seg,
                                                                               include_partial=True)
    dets = []
    for r in recs:
        o = r["obj"]
        xy = camera_model.world_xy_full(pose, intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
        mm_px = o.top_depth_mm / intr["fx"]
        dets.append({"label": r["label"], "names": list(r["names"]), "score": round(float(r["score"]), 4),
                     "xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)], "height_mm": round(float(o.height_mm), 2),
                     "bbox": [int(c) for c in o.bbox], "colour": o.colour,
                     "colours": {c: round(float(f), 4) for c, f in (o.colours or {}).items()},
                     "size_mm": [round(float(o.length_px * mm_px), 2), round(float(o.width_px * mm_px), 2)],
                     "partial": bool(r.get("partial")),
                     "emb16": base64.b64encode(np.asarray(r["emb"], np.float16).tobytes()).decode()})
    sq = perception.find_drop_square(rgb, depth, plane=seg.plane)
    square = drop_square.observe(sq, pose, intr) if sq else None
    bk = None
    if seg.bucket is not None and seg.bucket.get("center_px") is not None:
        a, b, c = seg.plane
        cu, cv = seg.bucket["center_px"]
        bk = bucket.locate(seg.bucket, depth.astype(float), float(a * cu + b * cv + c), pose, intr)
    return {"ts": float(time.time() if now is None else now), "source": source,
            "pose": {k: float(pose[k]) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta") if k in pose},  # o_*: tilted wrist
            "intr": {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")},
            "shape": [int(depth.shape[0]), int(depth.shape[1])], "table_depth_mm": round(float(seg.table_depth_mm), 2),
            "detections": dets, "square": square, "bucket": bk, "intruders": [[int(c) for c in i["bbox"]] for i in seg.intruders],
            "depth_valid": _valid_grid(depth)}


def apply_observation(tmap: dict, rec: dict) -> dict:
    """
    Fold one observation record into the map (in place; returns it). Pure: reads only the record and the map.

      1. each whole detection updates the entry it matches (same label nearby, or same look for unknowns), or moves
         the one entry of that label if seen confidently elsewhere (one of each object on the table), or adds one
      2. partly visible (edge) detections reset misses without moving the entry (see below)
      3. negative evidence: an entry that should be clearly visible here but was not detected gets a miss;
         MISSES_TO_DROP misses in a row remove it; an observation with a hand in view gives no negative evidence
      4. the drop square is refined from the tape edges this view sees (drop_square.update), or gets a miss
    Per object, "seen_ts" is the ts of the last observation that saw it; the map's "ts" the last update.
    """
    t0 = time.perf_counter()
    _ensure_map(tmap)
    now, source, pose = rec["ts"], rec["source"], rec["pose"]
    grid = _grid(rec)
    hand = bool(rec["intruders"])
    objs = tmap["objects"]
    touched = set()

    def det(d):
        # Defaults for fields a Viam Struct round trip drops when None/empty (an unrecognised object has no label)
        return {"label": None, "names": [], "colours": {}, "partial": False, **d,
                "emb": np.frombuffer(base64.b64decode(d["emb16"]), np.float16).astype(np.float32)}

    def new_entry(d, partial):
        e = {"id": tmap["next_id"], "label": d["label"], "names": list(d["names"]), "xy": list(d["xy"]), "n_seen": 0,
             "height_mm": d["height_mm"], "score": 0.0, "colour": d["colour"], "colours": {}, "frame_ids": [],
             "size_mm": [0.0, 0.0], "emb": d["emb"].tolist(), "partial": partial}
        tmap["next_id"] += 1
        objs.append(e)
        return e

    def seen(e, d):
        e["score"] = round(max(e["score"] * 0.95, d["score"]), 3)
        e.update(seen_ts=now, misses=0, source=source, n_seen=e["n_seen"] + 1)
        e["frame_ids"] = (e["frame_ids"] + [source + "@" + str(round(now, 1))])[-20:]
        touched.add(e["id"])

    for d in (det(d) for d in rec["detections"] if not d.get("partial")):
        e = next((e for e in objs if e["id"] not in touched
                  and _same(d, {**e, "emb": None if e["emb"] is None else np.asarray(e["emb"])})), None)
        if e is None and d["label"] and d["score"] >= MOVE_CONF:
            e = next((e for e in objs if e["label"] == d["label"] and e["id"] not in touched), None)  # it moved
        if e is None:
            e = new_entry(d, partial=False)
        else:
            far = math.dist(e["xy"], d["xy"]) > MERGE_XY_MM  # moved: jump, do not average two places
            a = 1.0 if far else XY_ALPHA
            e["xy"] = [(1 - a) * e["xy"][0] + a * d["xy"][0], (1 - a) * e["xy"][1] + a * d["xy"][1]]
            e["height_mm"] = (1 - XY_ALPHA) * e["height_mm"] + XY_ALPHA * d["height_mm"]
            emb = d["emb"] if e["emb"] is None else 0.8 * np.asarray(e["emb"]) + 0.2 * d["emb"]
            e["emb"] = (emb / np.linalg.norm(emb)).tolist()
        if d["label"] and not e["label"]:
            e["label"], e["names"] = d["label"], list(d["names"])
        e["xy"] = [round(e["xy"][0], 1), round(e["xy"][1], 1)]
        e["height_mm"] = round(float(e["height_mm"]), 1)
        e["score"] = round(max(e["score"] * 0.95, d["score"]), 3)
        e["colour"] = d["colour"]
        for c, f in d["colours"].items():
            e["colours"][c] = round(max(e["colours"].get(c, 0.0), f), 3)
        e["size_mm"] = [round(float(x), 1) for x in d["size_mm"]]
        e["partial"] = False  # seen whole: geometry is now reliable
        seen(e, d)

    # Partly visible (edge-cut) objects. Their xy is biased, so they never move a whole entry that is fine:
    # - an entry nearby: still there (reset misses, no xy update)
    # - recognised, and the label has no entry, or its entry should be visible here but is not: the object moved
    #   to this edge spot; keep it as a "partial" entry (approximate xy) so the executor knows where to look
    for d in (det(d) for d in rec["detections"] if d.get("partial")):
        e = next((e for e in objs if e["id"] not in touched and math.dist(e["xy"], d["xy"]) <= 1.5 * MERGE_XY_MM
                  and abs(e["height_mm"] - d["height_mm"]) <= MERGE_DH_MM
                  and (not d["label"] or not e["label"] or e["label"] == d["label"])), None)
        if e is not None:
            e["misses"] = 0
            touched.add(e["id"])
            continue
        if not d["label"] or d["score"] < MOVE_CONF:
            continue
        e = next((e for e in objs if e["label"] == d["label"] and e["id"] not in touched), None)
        if e is None:
            e = new_entry(d, partial=True)
        elif e.get("partial") or _expected_visible(e, rec, grid):
            e.update(xy=[round(float(d["xy"][0]), 1), round(float(d["xy"][1]), 1)], partial=True)
        else:
            continue  # its whole entry is out of view here and may well be right: do not move it on a partial view
        seen(e, d)

    dropped = []
    if not hand:
        for e in objs:
            if e["id"] not in touched and _expected_visible(e, rec, grid):
                e["misses"] = e.get("misses", 0) + 1
        dropped = [e for e in objs if e.get("misses", 0) >= MISSES_TO_DROP]
        tmap["objects"] = [e for e in objs if e.get("misses", 0) < MISSES_TO_DROP]

    obs, sq = rec.get("square"), tmap.get("square")
    if obs and "hull_xy" in obs:
        # Every view of the square, whole or border-cut, refines one rectangle (drop_square: sides, yaw, size)
        new = drop_square.update(sq, obs, now)
        if new is not None:
            tmap["square"] = new
    elif not obs and sq and not hand and _expected_visible(
            {"height_mm": 0.0, "size_mm": [2 * drop_square.radius(sq)] * 2}, rec, grid, xy=sq["xy"]):
        # Its whole estimated interior should be in view and it was not found: gone (MISSES_TO_DROP in a row)
        sq["misses"] = sq.get("misses", 0) + 1
        if sq["misses"] >= MISSES_TO_DROP:
            tmap.pop("square")

    if rec.get("bucket"):  # each view of the bucket moves the map's entry (it is carried around the table)
        tmap["bucket"] = bucket.update(tmap.get("bucket"), rec["bucket"], now)
    _mark_in_square(tmap)
    tmap["ts"] = now
    # detections without their embeddings, for the dashboard (AGENT 2); the full record is the observation itself
    light = [{k: v for k, v in {"label": None, "names": [], "colours": {}, "partial": False, **d}.items() if k != "emb16"}
             for d in rec["detections"]]  # same defaults as det(): a round-tripped record shows the same last_update
    tmap["last_update"] = {"ts": now, "pose": dict(pose), "source": source, "skipped": False,
                           "ms": round((time.perf_counter() - t0) * 1000), "detected": len(rec["detections"]),
                           "dropped": [e["label"] or f"unknown#{e['id']}" for e in dropped],
                           "hand_in_view": hand, "detections": light, "square": rec.get("square"),
                           "intruders": rec["intruders"]}
    return tmap


def update_map(tmap: dict, rgb: np.ndarray, depth: np.ndarray, pose: dict, intr: dict = None, store=None,
               source: str = "frame", recognised: list = None, seg=None, now: float = None) -> dict:
    """
    Fold one camera frame into the dynamic table map (in place; returns it): observe_frame() + apply_observation().

    WHY: the user wants every picture the arm takes to keep the map current, so a pick request can be answered
    without a fresh sweep and moved or removed objects disappear. A frame < SKIP_S after the last one at the same
    pose is skipped (last_update["skipped"]; nothing to log). Otherwise last_update["observation"] is the record to
    log: fold_observations() over the logged records rebuilds this same map.

    tmap None starts an empty map. recognised / seg: the caller's recognise_all(..., include_partial=True) output and
    the segment_full() it came from, so the same frame is not segmented or embedded twice (recognised without seg:
    the frame is segmented again but CLIP is skipped; without include_partial there are just no partial sightings).
    """
    tmap = _ensure_map(tmap if tmap is not None else empty_map())
    now = time.time() if now is None else now
    last = tmap.get("last_update")
    if last and not last.get("skipped") and now - last["ts"] < SKIP_S and _same_pose(last["pose"], pose):
        tmap["last_update"] = {**last, "skipped": True, "skip_ts": now}
        return tmap
    t0 = time.perf_counter()
    rec = observe_frame(rgb, depth, pose, intr, store, recognised, seg, source, now)
    apply_observation(tmap, rec)
    tmap["last_update"].update(ms=round((time.perf_counter() - t0) * 1000), observation=rec)
    return tmap


def fold_observations(observations, tmap: dict = None) -> dict:
    """
    Replay logged observation records (update_map's last_update["observation"]) into a map, in ts order.

    Deterministic and image-free: the same records give the same map as the live updates that produced them
    (only last_update.ms differs). The skip rule is not applied: a logged record was never a skipped frame.
    tmap: continue an existing map (None: start empty). Extra fields in a record (_id, paths, epoch) are ignored.
    """
    tmap = _ensure_map(tmap if tmap is not None else empty_map())
    for rec in sorted(observations, key=lambda r: r["ts"]):
        # A record rebuilt from last_update's light fields lacks the embeddings and the depth grid: refuse it
        # rather than fold a different map (no appearance merging, no negative evidence) without saying so
        missing = [k for k in REC_KEYS if k not in rec] + (["detections[].emb16"] if any(
            "emb16" not in d for d in rec.get("detections") or []) else [])
        if missing:
            raise ValueError(f"observation at ts {rec.get('ts')} lacks {missing}: log update_map's "
                             "last_update['observation'] record as is (AGENT3)")
        apply_observation(tmap, rec)
    return tmap


def rebuild_map(observations, since_ts: float = None) -> dict:
    """fold_observations() over the records at or after since_ts (None: all), from an empty map."""
    return fold_observations([r for r in observations if since_ts is None or r["ts"] >= since_ts])


def _current(objs: list) -> list:
    """
    The map's entries minus stale duplicates: of the dynamic-map entries (with seen_ts) sharing a label, only the
    most recently seen one. One of each object is on the table (the MOVE rule), so the other is where it used to be.

    WHY: the silver knife delivered into the square left its old entry behind (its old spot was not looked at
    again), so "silver box cutter" picked that empty spot instead of answering "already in the square".
    {LIVE TABLE_MAP 2026-10-29 "UTILITY KNIFE XY=[270.4, 257.6] … N=6.0" AND "UTILITY KNIFE XY=[385.3, 74.4] … IN_SQ=TRUE"}
    [CONFIDENCE: CONFIRMED 85% — tests/test_table_map.py::test_stale_duplicate; survey maps (no seen_ts) untouched]
    """
    latest = {}
    for m in objs:
        if m.get("label") and m.get("seen_ts") is not None and m.get("xy") is not None:
            key = (float(m["seen_ts"]), int(m.get("n_seen") or 0))
            if m["label"] not in latest or key > latest[m["label"]][0]:
                latest[m["label"]] = (key, id(m))
    return [m for m in objs if m.get("xy") is not None
            and (m.get("label") not in latest or m.get("seen_ts") is None or latest[m["label"]][1] == id(m))]


def best_on_map(table_map: dict, text: str, store=None) -> dict:
    """
    clip_embed.best() on the map's objects; adds "xy" and "map_obj" so the executor can go there.

    On a fresh whole-table map, searchable=True means "known to the library but not on the table".
    Objects already in the drop square are never picked; if the named object is only found there, the answer is
    a refusal with reason "... is already in the square" and already_in_square=True.
    """
    # Read-only and tolerant of maps made elsewhere (missing fields get _ensure_map's defaults, the map is not changed)
    live = _current(table_map.get("objects") or [])
    cands = [m for m in live if not m.get("in_square")]
    recs = []
    for k, m in enumerate(cands):
        size = m.get("size_mm") or [0.0, 0.0]
        # id = list position: dynamic-map ids are stable and stop matching positions once entries are dropped
        ns = SimpleNamespace(id=k, colour=m.get("colour"), colours=m.get("colours") or {},
                             length_px=size[0], width_px=size[1], height_mm=m.get("height_mm", 0.0), crop=None,
                             recognised=m.get("label"), recognised_names=m.get("names") or [])
        emb = m.get("emb")
        recs.append({"obj": ns, "label": m.get("label"), "names": m.get("names") or [], "score": m.get("score", 0.0),
                     "emb": np.zeros(EMB_DIM, np.float32) if emb is None else np.asarray(emb, dtype=np.float32)})
    b = clip_embed.best(None, text, store, recognised=recs)
    pick = cands[b["obj"].id] if b["obj"] is not None else None
    if pick is None:
        # best() refused (low far score, a tie): an entry named so (label or alias), the latest seen, is still the place to
        # look first; the close look there decides strictly {USER 2026-10-30 "WHEN YOU DO PLANNING SHOULD FIRST CHECK THE
        # LOCATION ON THE MAP FOR THAT OBJECT IF NOT FOUND THEN MOVE"}
        named = [m for m in cands if m.get("label")
                 and clip_embed.name_matches(text, [m["label"], *(m.get("names") or [])])]
        if named:
            pick = max(named, key=lambda m: float(m.get("seen_ts") or 0.0))
            b = {**b, "ambiguous": False, "score": float(pick.get("score") or 0.0)}
    done = None
    if pick is None:
        done = next((m for m in live if m.get("in_square") and m.get("label")
                     and clip_embed.name_matches(text, [m["label"], *(m.get("names") or [])])), None)
        if done is not None:
            b = {**b, "reason": f"{done['label']!r} is already in the square", "searchable": False}
    # look_first: the entry comes from an edge-cut sighting (approximate xy): go there, re-observe, then grasp
    return {**b, "obj": None, "map_obj": pick, "xy": tuple(pick["xy"]) if pick else None,
            "look_first": bool(pick and pick.get("partial")), "already_in_square": done is not None}


def load_scan_frames(ids: list, scan_dir: Path = None) -> list:
    """Frames saved by AGENT1 (data/scan/<id>.jpg, _depth.npy, .json) in map_from_frames() form."""
    scan_dir = scan_dir or Path(__file__).resolve().parent.parent / "data" / "scan"
    out = []
    for i in ids:
        meta = json.loads((scan_dir / f"{i}.json").read_text())
        out.append({"id": i, "rgb": np.asarray(Image.open(scan_dir / f"{i}.jpg").convert("RGB")),
                    "depth": np.load(scan_dir / f"{i}_depth.npy"), "pose": meta.get("pose", meta),
                    "intrinsics": meta.get("intrinsics")})
    return out
