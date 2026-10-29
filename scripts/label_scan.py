"""
Turn AGENT1's arm survey (data/scan/) into labelled object views, with one human labelling pass per object.

WHY: the object library should be built without a person turning objects. The arm visits several poses;
every fully visible object is seen several times. Grouping those sightings by where they are on the
table means a person labels each physical object once (from a contact sheet), and every crop of it
becomes a labelled view for retrieval.

Flow:
  label_scan.py            segment every frame -> world xy per object -> group -> docs/scan/contact.jpg
                           + data/scan/groups.json (+ crops, embeddings). Writes no views.
  label_scan.py --apply labels.json     {"3": ["green square", "green block"], ...}: first name = label,
                           rest = aliases -> Store.add_view for every crop in each labelled group
                           (source="arm_scan"). Unlabelled groups (e.g. transparent cups) are skipped.

Frames: data/scan/<n>.jpg + <n>_depth.npy (or <n>_depth.png, uint16 mm) + <n>.json (arm pose x, y, z,
theta; optionally "intrinsics"). Depth must be aligned to colour.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sorter import clip_embed, perception  # noqa: E402

SCAN = ROOT / "data" / "scan"
SHEET = ROOT / "docs" / "scan" / "contact.jpg"
# Measured with get_properties() on sfhack4 2026-10-29; a frame's own "intrinsics" overrides these
INTRINSICS = {"fx": 909.93, "fy": 909.36, "cx": 645.48, "cy": 365.37}
# Jaw (flange) axis relative to the camera axis, camera coordinates, from the hand-measured fingertip pixel
# (864, 685) at 137 mm: ((864-645.5)*137/909.9, (685-365.4)*137/909.4). Config cam offset is 3-4 cm off.
# {PICK_PLACE_AUTO.PY "JAW_TIP_PX = (864.0, 685.0)" "TIP_BELOW_CAM = 155.0 - CAM_T[2]"}
# [CONFIDENCE: CONFIRMED 80% — fingertip pixel measured by AGENT1; constant errors cancel when grouping]
JAW_IN_CAM_MM = (32.9, 48.1)
# Image +u is world -y and image +v is world -x at theta 0 (arm +y moved the scene +u, arm +x moved it +v).
# Wrist rotation direction is not verified on hardware: keep theta constant across a survey if possible.
THETA_SIGN = -1.0
LINK_XY_MM = 40.0  # two sightings closer than this on the table can be the same object
LINK_DH_MM = 15.0  # ... if their heights agree within this
LINK_COS = 0.70  # ... and their CLIP crops are at least this similar
CROPS_PER_ROW = 6


def load_frames():
    """[(name, rgb, depth, pose dict, intrinsics dict)] for every <n>.json with matching images."""
    out = []
    for pj in sorted(SCAN.glob("*.json")):
        if pj.name in ("groups.json", "applied.json"):
            continue
        meta = json.loads(pj.read_text())
        pose = meta.get("pose", meta)
        stem = pj.with_suffix("")
        img = next((p for p in (stem.with_suffix(".jpg"), stem.with_suffix(".png")) if p.exists()), None)
        dnp, dpng = Path(f"{stem}_depth.npy"), Path(f"{stem}_depth.png")
        if img is None or not (dnp.exists() or dpng.exists()):
            continue
        rgb = np.asarray(Image.open(img).convert("RGB"))
        depth = np.load(dnp) if dnp.exists() else np.asarray(Image.open(dpng)).astype(np.uint16)
        out.append((stem.name, rgb, depth, pose, {**INTRINSICS, **meta.get("intrinsics", {})}))
    return out


def world_xy(pose: dict, intr: dict, u: float, v: float, depth_mm: float) -> tuple:
    """
    Table xy (arm base frame, mm) of the point seen at pixel (u, v) at camera distance depth_mm.

    Pinhole back-projection to camera coordinates, then the offset from the jaw axis rotated into the
    world; a point exactly on the jaw axis maps to the flange xy.
    """
    xc = (u - intr["cx"]) * depth_mm / intr["fx"]
    yc = (v - intr["cy"]) * depth_mm / intr["fy"]
    dx, dy = JAW_IN_CAM_MM[1] - yc, JAW_IN_CAM_MM[0] - xc  # world offset at theta 0
    t = math.radians(THETA_SIGN * float(pose.get("theta", 0.0)))
    return (pose["x"] + dx * math.cos(t) - dy * math.sin(t), pose["y"] + dx * math.sin(t) + dy * math.cos(t))


def sightings(frames: list) -> tuple:
    """Every fully visible object in every frame, with world xy and a CLIP embedding."""
    items, crops = [], []
    for name, rgb, depth, pose, intr in frames:
        seg = perception.segment_full(rgb, depth, align=perception.IDENTITY)
        for o in seg.objects:
            # Top-face centroid sits above the object's axis from any view; the full mask drifts with its sides
            xy = world_xy(pose, intr, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
            items.append({"frame": name, "obj": o.id, "bbox": list(o.bbox), "xy": [round(xy[0], 1), round(xy[1], 1)],
                          "colour": o.colour, "height_mm": round(o.height_mm, 1), "rect_fill": round(o.rect_fill, 3)})
            crops.append(o.crop)
    emb = clip_embed.embed_images(crops) if crops else np.zeros((0, 512), np.float32)
    return items, crops, emb


def group(items: list, emb: np.ndarray) -> list:
    """
    Union-find over sightings: same place on the table, same height, compatible colour, similar crop.

    Each test alone fails somewhere (two touching objects share a place; two yellow cylinders share a
    look), so a link needs all of them.
    """
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["frame"] == b["frame"]:
                continue  # two objects in one frame are two objects
            if math.dist(a["xy"], b["xy"]) > LINK_XY_MM or abs(a["height_mm"] - b["height_mm"]) > LINK_DH_MM:
                continue
            if not clip_embed.colour_ok({a["colour"]}, b["colour"]) or float(emb[i] @ emb[j]) < LINK_COS:
                continue
            parent[find(i)] = find(j)
    members = {}
    for i in range(len(items)):
        members.setdefault(find(i), []).append(i)
    groups = sorted(members.values(), key=lambda m: -len(m))
    return groups


def write_outputs(items: list, crops: list, emb: np.ndarray, groups: list) -> dict:
    """Crops, embeddings, groups.json and the contact sheet; returns the groups dict."""
    (SCAN / "crops").mkdir(parents=True, exist_ok=True)
    np.save(SCAN / "embeddings.npy", emb)
    for k, (it, c) in enumerate(zip(items, crops)):
        it["crop"] = f"crops/{it['frame']}_{it['obj']}.jpg"
        it["index"] = k
        Image.fromarray(c).save(SCAN / it["crop"], quality=90)
    out = {}
    for gid, m in enumerate(groups):
        its = [items[i] for i in m]
        hs = [it["height_mm"] for it in its]
        cols = sorted({it["colour"] for it in its})
        out[str(gid)] = {"n": len(its), "colours": cols, "height_mm": round(float(np.median(hs)), 1),
                         "xy": [round(float(np.median([it["xy"][0] for it in its])), 1),
                                round(float(np.median([it["xy"][1] for it in its])), 1)],
                         # Height spread or mixed colours hints at fragments (transparent things) or a bad merge
                         "unstable": bool(np.std(hs) > 10 or len(cols) > 2), "items": its}
    (SCAN / "groups.json").write_text(json.dumps(out, indent=1))
    contact_sheet(out, crops)
    return out


def contact_sheet(groups: dict, crops: list) -> None:
    """One row per group: id + summary, then up to CROPS_PER_ROW crops scaled to the same height."""
    row_h, label_w, cell_w = 150, 300, 200
    sheet = Image.new("RGB", (label_w + cell_w * CROPS_PER_ROW, row_h * max(len(groups), 1)), "white")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    for r, (gid, g) in enumerate(groups.items()):
        y0 = r * row_h
        dr.rectangle((0, y0, sheet.width - 1, y0 + row_h - 1), outline=(200, 200, 200))
        dr.text((10, y0 + 10), f"G{gid}", fill=(200, 0, 120), font=font)
        dr.text((10, y0 + 45), f"n={g['n']} {'/'.join(g['colours'])}", fill=(0, 0, 0), font=small)
        dr.text((10, y0 + 68), f"h={g['height_mm']:.0f} mm", fill=(0, 0, 0), font=small)
        dr.text((10, y0 + 91), f"xy=({g['xy'][0]:.0f},{g['xy'][1]:.0f})", fill=(0, 0, 0), font=small)
        if g["unstable"]:
            dr.text((10, y0 + 114), "UNSTABLE", fill=(220, 0, 0), font=small)
        for c, it in enumerate(g["items"][:CROPS_PER_ROW]):
            im = Image.fromarray(crops[it["index"]])
            im.thumbnail((cell_w - 10, row_h - 10))
            sheet.paste(im, (label_w + c * cell_w + 5, y0 + 5))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET, quality=88)


def apply(labels_path: str, force: bool) -> None:
    """Write one Store view per crop of every labelled group (source="arm_scan")."""
    from sorter.store import Store

    marker = SCAN / "applied.json"
    if marker.exists() and not force:
        sys.exit(f"labels already applied ({marker}); use --force to add them again")
    labels = json.loads(Path(labels_path).read_text())
    groups = json.loads((SCAN / "groups.json").read_text())
    emb = np.load(SCAN / "embeddings.npy")
    store = Store()
    written = {}
    for gid, names in labels.items():
        if gid not in groups or not names:
            print(f"G{gid}: not in groups.json or no name, skipped")
            continue
        label, aliases = names[0], list(names[1:])
        ids = []
        for it in groups[gid]["items"]:
            jpeg = (SCAN / it["crop"]).read_bytes()
            ids.append(store.add_view(label, emb[it["index"]], jpeg, {
                "aliases": aliases, "source": "arm_scan", "group": int(gid), "frame": it["frame"],
                "colour": it["colour"], "height_mm": it["height_mm"], "rect_fill": it["rect_fill"], "xy": it["xy"]}))
        written[gid] = {"label": label, "views": ids}
        print(f"G{gid}: {len(ids)} views as {label!r} aliases={aliases}")
    marker.write_text(json.dumps(written, indent=1))
    print(f"store online={store.online}; skipped groups: {sorted(set(groups) - set(labels), key=int)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", help="labels.json: {group id: [label, alias, ...]}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scan-dir", help="read frames from here instead of data/scan (tests)")
    a = ap.parse_args()
    if a.scan_dir:
        SCAN = Path(a.scan_dir)
        SHEET = SCAN / "contact.jpg"
    if a.apply:
        apply(a.apply, a.force)
    else:
        fr = load_frames()
        its, cr, em = sightings(fr)
        gs = write_outputs(its, cr, em, group(its, em))
        print(f"{len(fr)} frames, {len(its)} sightings, {len(gs)} groups -> {SHEET}")
        for gid, g in gs.items():
            print(f"  G{gid}: n={g['n']} {g['colours']} h={g['height_mm']} xy={g['xy']}{' UNSTABLE' if g['unstable'] else ''}")
