"""
Whole-table object map from a fast sweep: every frame recognised, sightings merged by table position.

WHY: the robot should find a named object anywhere on the table, not only in the observe view or where it was
last remembered. AGENT1 sweeps 4-6 wide frames; this module turns them into one list of objects with table xy,
and answers a query against that list with the same rules as clip_embed.best().

Flow: frames {id, rgb, depth, pose, intrinsics?} -> recognise_all per frame -> table xy per object (pose + pixel
+ depth) -> merge across frames (same label within MERGE_XY_MM is one object) -> map.

Map doc (Store.save_map / latest_map, AGENT 2), latest wins:
  {ts, frames: [{id, pose}], objects: [{id, label, names, xy, height_mm, score, colour, colours, size_mm, frame_ids, emb}],
   square?: {xy, size_mm: [along x, along y], frame_ids}}   (drop zone: white interior of the blue tape, when fully seen)
label/names are None/[] for objects the library does not know; emb (512 floats) lets open-vocabulary queries run
on the map without re-embedding crops.
"""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sorter import clip_embed, perception

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
            u0, v0, u1, v1 = sq["bbox"]
            mm_px = sq["depth_mm"] / intr["fx"]
            # Image +v is world -x and +u is world -y, so the v extent is the size along x
            squares.append({"xy": perception.world_xy(f["pose"], intr, *sq["centroid_px"], sq["depth_mm"]),
                            "size_mm": ((v1 - v0) * mm_px, (u1 - u0) * mm_px), "frame": f["id"]})
        for r in clip_embed.recognise_all(f["rgb"], f["depth"], store):
            o = r["obj"]
            xy = perception.world_xy(f["pose"], intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
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
    if squares:
        # Median over the frames that saw the whole square; frames agreed within ~1.3 mm on the z=400 survey
        doc["square"] = {"xy": [round(float(np.median([q["xy"][k] for q in squares])), 1) for k in (0, 1)],
                         "size_mm": [round(float(np.median([q["size_mm"][k] for q in squares])), 1) for k in (0, 1)],
                         "frame_ids": [q["frame"] for q in squares]}
    _mark_in_square(doc)
    return doc


SQUARE_MARGIN_MM = 10.0  # an object this close to the square's interior counts as inside it


def _mark_in_square(tmap: dict) -> None:
    """
    Flag every object whose xy lies in the drop square's interior (in_square=True).

    WHY: the module filters objects already in the square out of a single frame, but a query against the map would
    otherwise pick the yellow block out of the square again after it was delivered.
    """
    sq = tmap.get("square")
    for e in tmap["objects"]:
        e["in_square"] = bool(sq) and (abs(e["xy"][0] - sq["xy"][0]) <= sq["size_mm"][0] / 2 + SQUARE_MARGIN_MM
                                       and abs(e["xy"][1] - sq["xy"][1]) <= sq["size_mm"][1] / 2 + SQUARE_MARGIN_MM)


# ---- dynamic map: every frame the arm takes updates it ------------------------------------------------------
SKIP_S = 0.5  # a frame this soon after the last one at the same pose adds nothing: skip (saves the CLIP batch)
SAME_POSE_MM, SAME_POSE_DEG = 5.0, 2.0
MOVE_CONF = 0.5  # a confident sighting of a label seen elsewhere moves that object (one of each on the table)
MISSES_TO_DROP = 2  # expected in view, not detected, this many frames in a row -> removed from the map
FOOT_MARGIN_PX = 60  # an entry projected closer than this to the image border is not expected to be detected
XY_ALPHA = 0.5  # weight of a new sighting in the entry's xy (objects that move should follow quickly)


def empty_map() -> dict:
    return {"ts": time.time(), "objects": [], "next_id": 0, "last_update": None}


def _same_pose(a: dict, b: dict) -> bool:
    return (math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"])) <= SAME_POSE_MM
            and abs(float(a.get("theta", 0)) - float(b.get("theta", 0))) <= SAME_POSE_DEG)


def _expected_visible(e: dict, pose: dict, intr: dict, seg, depth: np.ndarray, xy=None) -> bool:
    """
    Should this map entry be clearly detected in this frame? Its whole extent projects inside the view (minus a
    margin), with valid depth at its centre (not the blind range, not the held object), not under the fingertips.

    The whole extent, not just the centre: the 17 cm KIND bar and the 19x14 cm square had their centres inside the
    view while their bodies were cut by the edge, were not detected as whole, and got wrongly dropped.
    """
    H, W = depth.shape
    z = max(seg.table_depth_mm - e.get("height_mm", 0.0), 50.0)
    cx, cy = xy or e["xy"]
    r = 0.5 * max(e.get("size_mm") or [0.0, 0.0])
    m = FOOT_MARGIN_PX
    for px, py in ((cx, cy), (cx - r, cy - r), (cx - r, cy + r), (cx + r, cy - r), (cx + r, cy + r)):
        pu, pv = perception.project_xy(pose, intr, (px, py), z)
        if not (m <= pu < W - m and m <= pv < H - m):
            return False
    u, v = perception.project_xy(pose, intr, (cx, cy), z)
    if any(u0 <= u < u1 and v0 <= v < v1 for u0, v0, u1, v1 in perception.FINGER_BOXES):
        return False
    win = depth[int(v) - 7:int(v) + 8, int(u) - 7:int(u) + 8]
    return bool(win.size and (win > 0).mean() >= 0.5)


def update_map(tmap: dict, rgb: np.ndarray, depth: np.ndarray, pose: dict, intr: dict = None, store=None,
               source: str = "frame", recognised: list = None, seg=None, now: float = None) -> dict:
    """
    Fold one camera frame into the dynamic table map (in place; returns it).

    WHY: the user wants every picture the arm takes to keep the map current, so a pick request can be answered
    without a fresh sweep and moved or removed objects disappear. Per frame:
      1. detections (recognise_all) -> table xy with the full pose (theta included)
      2. each detection updates the entry it matches (same label nearby, or same look for unknowns), or moves the
         one entry of that label if seen confidently elsewhere (one of each object on the table), or adds one
      3. negative evidence: an entry that should be clearly visible here but was not detected gets a miss;
         MISSES_TO_DROP misses in a row remove it. Partly visible (edge) objects reset misses without moving the
         entry; a frame with a hand in view gives no negative evidence.
      4. the drop square is updated the same way.
    A frame < SKIP_S after the last one at the same pose is skipped. tmap["last_update"] records what happened.

    tmap None starts an empty map. recognised / seg: the caller's recognise_all(..., include_partial=True) output and
    the segment_full() it came from, so the same frame is not segmented or embedded twice (recognised without seg:
    the frame is segmented again but CLIP is skipped; without include_partial there are just no partial sightings).
    Per object, "seen_ts" is the time of the last frame that saw it; the map's "ts" the last update.
    """
    tmap = tmap if tmap is not None else empty_map()
    now = time.time() if now is None else now
    last = tmap.get("last_update")
    if last and not last.get("skipped") and now - last["ts"] < SKIP_S and _same_pose(last["pose"], pose):
        tmap["last_update"] = {**last, "skipped": True, "skip_ts": now}
        return tmap
    t0 = time.perf_counter()
    intr = {**perception.INTRINSICS, **(intr or {})}
    seg = seg if seg is not None else perception.segment_full(rgb, depth)
    recs = recognised if recognised is not None else clip_embed.recognise_all(rgb, depth, store, seg=seg,
                                                                               include_partial=True)
    objs = tmap["objects"]
    touched = set()

    def det_of(o, r):
        xy = perception.world_xy(pose, intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
        mm_px = o.top_depth_mm / intr["fx"]
        return {"label": r["label"], "names": r["names"], "xy": xy, "height_mm": o.height_mm, "score": r["score"],
                "colour": o.colour, "colours": dict(o.colours or {}), "emb": r["emb"],
                "size_mm": (o.length_px * mm_px, o.width_px * mm_px)}

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

    for r in (r for r in recs if not r.get("partial")):
        d = det_of(r["obj"], r)
        e = next((e for e in objs if e["id"] not in touched and _same(d, {**e, "emb": np.asarray(e["emb"])})), None)
        if e is None and d["label"] and d["score"] >= MOVE_CONF:
            e = next((e for e in objs if e["label"] == d["label"] and e["id"] not in touched), None)  # it moved
        if e is None:
            e = new_entry(d, partial=False)
        else:
            far = math.dist(e["xy"], d["xy"]) > MERGE_XY_MM  # moved: jump, do not average two places
            a = 1.0 if far else XY_ALPHA
            e["xy"] = [(1 - a) * e["xy"][0] + a * d["xy"][0], (1 - a) * e["xy"][1] + a * d["xy"][1]]
            e["height_mm"] = (1 - XY_ALPHA) * e["height_mm"] + XY_ALPHA * d["height_mm"]
            emb = 0.8 * np.asarray(e["emb"]) + 0.2 * d["emb"]
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
    for r in (r for r in recs if r.get("partial")):
        d = det_of(r["obj"], r)
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
        elif e.get("partial") or _expected_visible(e, pose, intr, seg, depth):
            e.update(xy=[round(float(d["xy"][0]), 1), round(float(d["xy"][1]), 1)], partial=True)
        else:
            continue  # its whole entry is out of view here and may well be right: do not move it on a partial view
        seen(e, d)

    dropped = []
    if not seg.intruders:
        for e in objs:
            if e["id"] not in touched and _expected_visible(e, pose, intr, seg, depth):
                e["misses"] = e.get("misses", 0) + 1
        dropped = [e for e in objs if e.get("misses", 0) >= MISSES_TO_DROP]
        tmap["objects"] = [e for e in objs if e.get("misses", 0) < MISSES_TO_DROP]

    sq = perception.find_drop_square(rgb, depth, plane=seg.plane)
    if sq:
        u0, v0, u1, v1 = sq["bbox"]
        mm_px = sq["depth_mm"] / intr["fx"]
        xy = perception.world_xy(pose, intr, *sq["centroid_px"], sq["depth_mm"])
        old = tmap.get("square")
        a = 1.0 if not old or math.dist(old["xy"], xy) > MERGE_XY_MM else XY_ALPHA
        oxy = old["xy"] if old else xy
        tmap["square"] = {"xy": [round((1 - a) * oxy[0] + a * xy[0], 1), round((1 - a) * oxy[1] + a * xy[1], 1)],
                          "size_mm": [round((v1 - v0) * mm_px, 1), round((u1 - u0) * mm_px, 1)],
                          "seen_ts": now, "misses": 0}
    elif tmap.get("square") and not seg.intruders and _expected_visible(
            {"height_mm": 0.0, "size_mm": tmap["square"]["size_mm"]}, pose, intr, seg, depth, xy=tmap["square"]["xy"]):
        tmap["square"]["misses"] = tmap["square"].get("misses", 0) + 1
        if tmap["square"]["misses"] >= MISSES_TO_DROP:
            tmap.pop("square")

    _mark_in_square(tmap)
    tmap["ts"] = now
    tmap["last_update"] = {"ts": now, "pose": dict(pose), "source": source, "skipped": False,
                           "ms": round((time.perf_counter() - t0) * 1000), "detected": len(recs),
                           "dropped": [e["label"] or f"unknown#{e['id']}" for e in dropped],
                           "hand_in_view": bool(seg.intruders)}
    return tmap


def best_on_map(table_map: dict, text: str, store=None) -> dict:
    """
    clip_embed.best() on the map's objects; adds "xy" and "map_obj" so the executor can go there.

    On a fresh whole-table map, searchable=True means "known to the library but not on the table".
    Objects already in the drop square are never picked; if the named object is only found there, the answer is
    a refusal with reason "... is already in the square" and already_in_square=True.
    """
    cands = [m for m in table_map["objects"] if not m.get("in_square")]
    recs = []
    for k, m in enumerate(cands):
        # id = list position: dynamic-map ids are stable and stop matching positions once entries are dropped
        ns = SimpleNamespace(id=k, colour=m["colour"], colours=m.get("colours", {}),
                             length_px=m["size_mm"][0], width_px=m["size_mm"][1],
                             height_mm=m["height_mm"], crop=None, recognised=m["label"], recognised_names=m["names"])
        recs.append({"obj": ns, "label": m["label"], "names": m["names"] or [], "score": m["score"],
                     "emb": np.asarray(m["emb"], dtype=np.float32)})
    b = clip_embed.best(None, text, store, recognised=recs)
    pick = cands[b["obj"].id] if b["obj"] is not None else None
    done = None
    if pick is None:
        done = next((m for m in table_map["objects"] if m.get("in_square") and m["label"]
                     and clip_embed.name_matches(text, [m["label"], *(m["names"] or [])])), None)
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
