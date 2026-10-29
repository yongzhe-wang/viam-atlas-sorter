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
  label_scan.py --phone DIR             phone close-ups: crops + CLIP-proposed labels -> docs/scan/phone_contact.jpg
  label_scan.py --phone DIR --apply labels.json     {"P2": ["green cube", "green square"]} -> source="phone"

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
        # AGENT1 also writes objects.json / free_spots.json (lists) next to the frames
        if not isinstance(meta, dict):
            continue
        pose = meta.get("pose", meta)
        if not isinstance(pose, dict) or "x" not in pose:
            continue
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


def _sheet(rows: list, path: Path) -> None:
    """Contact sheet: rows of (title, [text lines], [PIL images]); title in magenta, images scaled to fit."""
    row_h, label_w, cell_w = 150, 300, 200
    sheet = Image.new("RGB", (label_w + cell_w * CROPS_PER_ROW, row_h * max(len(rows), 1)), "white")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    for r, (title, lines, ims) in enumerate(rows):
        y0 = r * row_h
        dr.rectangle((0, y0, sheet.width - 1, y0 + row_h - 1), outline=(200, 200, 200))
        dr.text((10, y0 + 10), title, fill=(200, 0, 120), font=font)
        for k, line in enumerate(lines):
            dr.text((10, y0 + 45 + 23 * k), line, fill=(220, 0, 0) if line == "UNSTABLE" else (0, 0, 0), font=small)
        for c, im in enumerate(ims[:CROPS_PER_ROW]):
            im = im.copy()
            im.thumbnail((cell_w - 10, row_h - 10))
            sheet.paste(im, (label_w + c * cell_w + 5, y0 + 5))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def contact_sheet(groups: dict, crops: list) -> None:
    """Arm-scan sheet: one row per group with id, colour, height, table xy and up to 6 crops."""
    rows = []
    for gid, g in groups.items():
        lines = [f"n={g['n']} {'/'.join(g['colours'])}", f"h={g['height_mm']:.0f} mm",
                 f"xy=({g['xy'][0]:.0f},{g['xy'][1]:.0f})"] + (["UNSTABLE"] if g["unstable"] else [])
        rows.append((f"G{gid}", lines, [Image.fromarray(crops[it["index"]]) for it in g["items"]]))
    _sheet(rows, SHEET)


# ---- phone photos (low-weight extra views) ------------------------------------------------------------
# Objects per photo, from the user's list relayed by VIAM CENTERAL 2026-10-29. Overviews (7383-7385, 7409)
# and the measuring cups are skipped. A proposal is restricted to the photo's own objects.
PHONE_MAP = {
    **{n: ["black tape roll"] for n in (7386, 7387, 7388)},
    **{n: ["granola bar", "KIND bar"] for n in (7389, 7390, 7391)},
    **{n: ["KIND bar"] for n in (7392, 7393)},
    **{n: ["yellow block"] for n in (7394, 7395, 7396, 7397)},
    **{n: ["green cylinder", "green cube"] for n in (7398, 7399, 7400, 7401, 7402)},
    **{n: ["blue cylinder", "yellow cylinder", "blue block", "blue-handle screwdriver", "green cylinder"]
       for n in (7403, 7404, 7405, 7406)},
    **{n: ["red screwdriver"] for n in (7407, 7408)},
}
# Zero-shot prompts: descriptive, so the two snack bars and the two screwdrivers separate; labels stay short
PHONE_PROMPTS = {
    "black tape roll": "a roll of black tape", "granola bar": "a granola bar in a black and green wrapper",
    "KIND bar": "a KIND bar in a teal wrapper", "yellow block": "a yellow rectangular wooden block",
    "green cylinder": "a green wooden cylinder", "green cube": "a small green wooden cube",
    "blue cylinder": "a blue wooden cylinder", "yellow cylinder": "a yellow wooden cylinder",
    "blue block": "a blue rectangular block", "blue-handle screwdriver": "a screwdriver with a blue and black handle",
    "red screwdriver": "a screwdriver with a red and black handle"}
# A crop whose best global match is one of these (or another photo's object) is a neighbour/background: dropped
PHONE_DISTRACTORS = ["blue tape on white paper", "a sheet of white paper", "a wooden table top",
                     "a clear plastic measuring cup", "the metal base of a robot arm", "a phone in an orange case"]
PHONE_MAX_PER_GROUP = 12
PHONE_SAT, PHONE_VAL = 0.45, 0.25  # saturated-colour blob thresholds (HSV 0-1)
PHONE_AREA = (0.004, 0.12)  # blob area as a fraction of the photo; larger = the blue paper/tape


def phone_crops(path: Path) -> list:
    """Saturated-colour blobs (cubes, cylinders, handles, wrappers) plus a centre crop for black objects."""
    from scipy import ndimage

    im = Image.open(path).convert("RGB")
    im.thumbnail((800, 800))
    a = np.asarray(im)
    hsv = np.asarray(im.convert("HSV")).astype(float) / 255.0
    lab, n = ndimage.label(ndimage.binary_opening((hsv[..., 1] > PHONE_SAT) & (hsv[..., 2] > PHONE_VAL), iterations=2))
    H, W = lab.shape
    out = []
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        frac = len(us) / (H * W)
        borders = (us.min() <= 1) + (vs.min() <= 1) + (us.max() >= W - 2) + (vs.max() >= H - 2)
        if not PHONE_AREA[0] <= frac <= PHONE_AREA[1] or borders >= 2:
            continue
        mu, mv = int((np.ptp(us) + 1) * 0.15) + 4, int((np.ptp(vs) + 1) * 0.15) + 4
        out.append(a[max(0, vs.min() - mv):vs.max() + mv, max(0, us.min() - mu):us.max() + mu].copy())
    out.append(a[H // 5:H - H // 5, W // 5:W - W // 5].copy())  # centre crop
    return out


def phone_propose(photo_dir: Path) -> None:
    """Crop every mapped photo, propose a label per crop by CLIP zero-shot, write the phone contact sheet."""
    out_dir = SCAN / "phone"
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    items, crops = [], []
    for p in sorted(photo_dir.glob("IMG_*.jpg")):
        try:
            num = int(p.stem.split("_")[1])
        except ValueError:
            continue
        if num not in PHONE_MAP or p.stem.count("_") > 1:
            continue
        for k, c in enumerate(phone_crops(p)):
            items.append({"photo": p.name, "allowed": PHONE_MAP[num], "crop": f"phone/crops/{p.stem}_{k}.jpg"})
            crops.append(c)
    emb = clip_embed.embed_images(crops)
    names = sorted(PHONE_PROMPTS)
    # Prompts are embedded raw (they already read "a ..."), so embed_text's "a photo of a" prefix is not doubled
    text = np.stack([clip_embed.embed_text(PHONE_PROMPTS[n]) for n in names] +
                    [clip_embed.embed_text(d) for d in PHONE_DISTRACTORS])
    groups, dropped = {}, 0
    for k, (it, c) in enumerate(zip(items, crops)):
        sims = text @ emb[k]
        top = int(np.argmax(sims))
        # Keep only crops whose best match over every name and distractor is one of this photo's objects
        if top >= len(names) or names[top] not in it["allowed"]:
            dropped += 1
            continue
        Image.fromarray(c).save(SCAN / it["crop"], quality=90)
        it.update(index=k, proposed=names[top], sim=round(float(sims[top]), 3))
        groups.setdefault(it["proposed"], []).append(it)
    np.save(out_dir / "embeddings.npy", emb)
    out = {f"P{i}": {"proposed": lab, "n": min(len(its), PHONE_MAX_PER_GROUP),
                     "items": sorted(its, key=lambda t: -t["sim"])[:PHONE_MAX_PER_GROUP]}
           for i, (lab, its) in enumerate(sorted(groups.items()))}
    print(f"dropped {dropped} neighbour/background crops")
    (out_dir / "groups.json").write_text(json.dumps(out, indent=1))
    rows = [(gid, [g["proposed"], f"n={g['n']}"], [Image.fromarray(crops[it["index"]]) for it in g["items"]])
            for gid, g in out.items()]
    _sheet(rows, SHEET.parent / "phone_contact.jpg")
    print(f"{len(items)} phone crops -> {SHEET.parent / 'phone_contact.jpg'}")
    for gid, g in out.items():
        print(f"  {gid}: {g['proposed']} n={g['n']}")


CARRY_XY_MM = 50.0  # a new group this close to a labelled one (same height/colour) is that object
CARRY_COS = 0.88  # appearance-only carry (the object moved): mean-crop cosine at least this


def _applied_items(groups: dict, applied: dict) -> dict:
    """{(frame, obj): label} for every sighting already written to the Store."""
    if "items" in applied:  # format written by carry()
        return {tuple(k.split("|")): v for k, v in applied["items"].items()}
    out = {}
    for gid, a in applied.items():  # first apply() format: {gid: {"label", "views"}}
        for it in groups.get(gid, {}).get("items", []):
            out[(it["frame"], str(it["obj"]))] = a["label"]
    return out


def carry(labels_path: str) -> None:
    """
    Regroup all frames after a re-shoot and carry the existing labels over; sheet only for what changed.

    Per new group: (1) sightings already labelled keep their label (old frames segment identically);
    (2) else a labelled group at the same table xy, height and colour; (3) else appearance + colour + height
    only, flagged "moved" (the yellow block is the trial object and moves). Writes data/scan/labels_carried.json
    and docs/scan/contact_changed.jpg; apply with --apply labels_carried.json --only-new.
    """
    old_groups = json.loads((SCAN / "groups.json").read_text())
    applied = json.loads((SCAN / "applied.json").read_text()) if (SCAN / "applied.json").exists() else {}
    names = json.loads(Path(labels_path).read_text())  # the labels.json that was applied: {gid: [label, ...]}
    by_label = {v[0]: v for v in names.values()}
    done = _applied_items(old_groups, applied)
    old_emb = np.load(SCAN / "embeddings.npy")
    # Reference per labelled old group: median xy, height, colours, mean embedding
    refs = []
    for gid, v in names.items():
        g = old_groups.get(gid)
        if not g:
            continue
        e = old_emb[[it["index"] for it in g["items"]]].mean(0)
        refs.append({"label": v[0], "xy": g["xy"], "h": g["height_mm"], "colours": set(g["colours"]),
                     "emb": e / np.linalg.norm(e)})
    old_frames = {it["frame"] for g in old_groups.values() for it in g["items"]}
    (SCAN / "applied_prev.json").write_text(json.dumps({"items": {"|".join(k): v for k, v in done.items()}}, indent=1))

    fr = load_frames()
    its, cr, em = sightings(fr)
    new = write_outputs(its, cr, em, group(its, em))
    carried, rows = {}, []
    for gid, g in new.items():
        keys = [(it["frame"], str(it["obj"])) for it in g["items"]]
        prior = [done[k] for k in keys if k in done]
        e = em[[it["index"] for it in g["items"]]].mean(0)
        e = e / np.linalg.norm(e)
        ok = [r for r in refs if abs(r["h"] - g["height_mm"]) <= LINK_DH_MM
              and any(clip_embed.colour_ok({c}, gc) for c in r["colours"] for gc in g["colours"])]
        how = label = None
        if prior:
            label, how = max(set(prior), key=prior.count), "same sightings"
        else:
            near = [r for r in ok if math.dist(r["xy"], g["xy"]) <= CARRY_XY_MM]
            if near:
                label, how = min(near, key=lambda r: math.dist(r["xy"], g["xy"]))["label"], "same place"
            else:
                look = [r for r in ok if float(r["emb"] @ e) >= CARRY_COS]
                if len(look) == 1:
                    label, how = look[0]["label"], "moved (appearance)"
        g["carried"] = {"label": label, "how": how}
        if label:
            carried[gid] = by_label[label]
        has_new = any(it["frame"] not in old_frames for it in g["items"])
        if has_new or (how and how != "same sightings"):
            rows.append((f"G{gid}", [f"-> {label or '?'}", how or "unmatched", f"n={g['n']} h={g['height_mm']:.0f}"],
                         [Image.fromarray(cr[it["index"]]) for it in g["items"]]))
    (SCAN / "groups.json").write_text(json.dumps(new, indent=1))
    (SCAN / "labels_carried.json").write_text(json.dumps(carried, indent=1))
    _sheet(rows, SHEET.parent / "contact_changed.jpg")
    print(f"{len(new)} groups, {len(carried)} labelled by carry-over, {len(rows)} new/changed -> "
          f"{SHEET.parent / 'contact_changed.jpg'}")
    for title, lines, _ in rows:
        print(f"  {title}: {' | '.join(lines)}")


def apply(labels_path: str, force: bool, phone: bool = False, only_new: bool = False) -> None:
    """
    Write one Store view per crop of every labelled group.

    Arm scan: source="arm_scan". Phone photos (--phone): source="phone"; rank() gives these a lower vote
    weight because the oblique phone view differs from the top-down wrist camera.
    """
    from sorter.store import Store

    base = SCAN / "phone" if phone else SCAN
    marker = base / "applied.json"
    done = {}
    if only_new:
        # After carry(): skip sightings already in the Store (recorded in applied_prev.json)
        prev = base / "applied_prev.json"
        done = json.loads(prev.read_text())["items"] if prev.exists() else {}
    elif marker.exists() and not force:
        sys.exit(f"labels already applied ({marker}); use --force, or --only-new after --carry")
    labels = json.loads(Path(labels_path).read_text())
    groups = json.loads((base / "groups.json").read_text())
    emb = np.load(base / "embeddings.npy")
    store = Store()
    written = {}
    items_done = dict(done)
    for gid, names in labels.items():
        if gid not in groups or not names:
            print(f"{gid}: not in groups.json or no name, skipped")
            continue
        label, aliases = names[0], list(names[1:])
        ids = []
        for it in groups[gid]["items"]:
            key = f"{it.get('frame', it.get('photo'))}|{it.get('obj', it['index'])}"
            if key in done:
                continue
            items_done[key] = label
            jpeg = (SCAN / it["crop"]).read_bytes()
            if phone:
                meta = {"aliases": aliases, "source": "phone", "group": gid, "photo": it["photo"]}
            else:
                meta = {"aliases": aliases, "source": "arm_scan", "group": int(gid), "frame": it["frame"],
                        "colour": it["colour"], "height_mm": it["height_mm"], "rect_fill": it["rect_fill"], "xy": it["xy"]}
            ids.append(store.add_view(label, emb[it["index"]], jpeg, meta))
        written[gid] = {"label": label, "views": ids}
        print(f"{gid}: {len(ids)} views as {label!r} aliases={aliases}")
    marker.write_text(json.dumps({"items": items_done, "groups": written}, indent=1))
    print(f"store online={store.online}; skipped groups: {sorted(set(groups) - set(labels))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", help="labels.json: {group id: [label, alias, ...]}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scan-dir", help="read frames from here instead of data/scan (tests)")
    ap.add_argument("--phone", help="photo directory: propose labels for phone close-ups (with --apply: apply them)")
    ap.add_argument("--carry", help="after a re-shoot: the labels.json applied before; regroup and carry labels over")
    ap.add_argument("--only-new", action="store_true", help="with --apply after --carry: add only new sightings")
    a = ap.parse_args()
    if a.scan_dir:
        SCAN = Path(a.scan_dir)
        SHEET = SCAN / "contact.jpg"
    if a.carry:
        carry(a.carry)
    elif a.apply:
        apply(a.apply, a.force, phone=bool(a.phone), only_new=a.only_new)
    elif a.phone:
        phone_propose(Path(a.phone))
    else:
        fr = load_frames()
        its, cr, em = sightings(fr)
        gs = write_outputs(its, cr, em, group(its, em))
        print(f"{len(fr)} frames, {len(its)} sightings, {len(gs)} groups -> {SHEET}")
        for gid, g in gs.items():
            print(f"  G{gid}: n={g['n']} {g['colours']} h={g['height_mm']} xy={g['xy']}{' UNSTABLE' if g['unstable'] else ''}")
