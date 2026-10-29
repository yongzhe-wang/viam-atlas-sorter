"""
Whole-table object map from a fast sweep: every frame recognised, sightings merged by table position.

WHY: the robot should find a named object anywhere on the table, not only in the observe view or where it was
last remembered. AGENT1 sweeps 4-6 wide frames; this module turns them into one list of objects with table xy,
and answers a query against that list with the same rules as clip_embed.best().

Flow: frames {id, rgb, depth, pose, intrinsics?} -> recognise_all per frame -> table xy per object (pose + pixel
+ depth) -> merge across frames (same label within MERGE_XY_MM is one object) -> map.

Map doc (Store.save_map / latest_map, AGENT 2), latest wins:
  {ts, frames: [{id, pose}], objects: [{id, label, names, xy, height_mm, score, colour, colours, size_mm, frame_ids, emb}]}
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
    """Every target of every frame with table xy, size in mm and embedding; plus ms per frame."""
    out, ms = [], []
    for f in frames:
        t = time.perf_counter()
        intr = {**perception.INTRINSICS, **(f.get("intrinsics") or {})}
        for r in clip_embed.recognise_all(f["rgb"], f["depth"], store):
            o = r["obj"]
            xy = perception.world_xy(f["pose"], intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
            mm_px = o.top_depth_mm / intr["fx"]
            out.append({"label": r["label"], "names": r["names"], "xy": xy, "height_mm": o.height_mm,
                        "score": r["score"], "colour": o.colour, "colours": dict(getattr(o, "colours", {}) or {}),
                        "size_mm": (o.length_px * mm_px, o.width_px * mm_px),
                        "emb": r["emb"], "frame": f["id"]})
        ms.append((time.perf_counter() - t) * 1000)
    return out, ms


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
    dets, ms = _detections(frames, store)
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
    return {"ts": time.time(), "frames": [{"id": f["id"], "pose": f["pose"]} for f in frames], "objects": out,
            "timing_ms": [round(x) for x in ms]}


def best_on_map(table_map: dict, text: str, store=None) -> dict:
    """
    clip_embed.best() on the map's objects; adds "xy" and "map_obj" so the executor can go there.

    On a fresh whole-table map, searchable=True means "known to the library but not on the table".
    """
    recs = []
    for m in table_map["objects"]:
        ns = SimpleNamespace(id=m["id"], colour=m["colour"], colours=m.get("colours", {}),
                             length_px=m["size_mm"][0], width_px=m["size_mm"][1],
                             height_mm=m["height_mm"], crop=None, recognised=m["label"], recognised_names=m["names"])
        recs.append({"obj": ns, "label": m["label"], "names": m["names"] or [], "score": m["score"],
                     "emb": np.asarray(m["emb"], dtype=np.float32)})
    b = clip_embed.best(None, text, store, recognised=recs)
    pick = table_map["objects"][b["obj"].id] if b["obj"] is not None else None
    return {**b, "obj": None, "map_obj": pick, "xy": tuple(pick["xy"]) if pick else None}


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
