"""
Add library views of one object from frames the module already took, without new arm moves.

WHY: the library only knew the cylinders standing (60 mm tall); lying down (30 mm) the green one was filtered out
by height and called "green cube" or nothing on sfhack4. Views of the new pose come from what the robot saw:
  --observations FILE   records of the module's observation log (JSON list, AGENT 2's export): each detection
                        already carries the robot's CLIP embedding of its full-resolution crop (emb16), height,
                        colour and table xy; an optional "jpeg_b64" per record gives the view a thumbnail
  --frames DIR          module idle frames (data/camera/module_frame.jpg + .json copies): the crop is cut from the
                        <= 640 px jpeg by its box and embedded here; height and colour come from the .json
Detections are picked by table xy (--near X,Y within --radius; frames: by --box-colour only), colour and a height
range; aliases are copied from the label's existing views. A detection already added is skipped (data/scan/
added_views.json), so re-running never duplicates. --dry-run lists what would be added.

  python scripts/add_views.py --label "green cylinder" --observations obs.json --near 214,56 --colour green --h 20,40
"""
import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import clip_embed, perception  # noqa: E402
from sorter.store import Store  # noqa: E402

DONE = Path(__file__).resolve().parent.parent / "data" / "scan" / "added_views.json"


def _crop(rgb: np.ndarray, bbox) -> np.ndarray:
    """bbox grown by perception.CROP_MARGIN on each side, as segment_full cuts library crops."""
    u0, v0, u1, v1 = (int(round(c)) for c in bbox)
    mu, mv = int(perception.CROP_MARGIN * (u1 - u0)), int(perception.CROP_MARGIN * (v1 - v0))
    H, W = rgb.shape[:2]
    return rgb[max(0, v0 - mv):min(H, v1 + mv + 1), max(0, u0 - mu):min(W, u1 + mu + 1)]


def _jpeg(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def from_observations(path: Path, a) -> list:
    """[(key, embedding, jpeg|None, meta)] for the matching whole detections of logged observation records."""
    out = []
    for rec in json.loads(path.read_text()):
        rgb, scale = None, 1.0
        if rec.get("jpeg_b64"):
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(rec["jpeg_b64"]))).convert("RGB"))
            scale = rgb.shape[1] / rec["shape"][1] if rec.get("shape") else 1.0
        for k, d in enumerate(rec.get("detections") or []):
            if d.get("partial") or "emb16" not in d or not _wanted(d.get("xy"), d.get("colour"), d.get("height_mm"), a):
                continue
            emb = np.frombuffer(base64.b64decode(d["emb16"]), np.float16).astype(np.float32)
            jpeg = _jpeg(_crop(rgb, [c * scale for c in d["bbox"]])) if rgb is not None and d.get("bbox") else None
            out.append((f"obs:{rec['ts']:.3f}:{k}", emb, jpeg,
                        {"source": "observation", "obs_ts": rec["ts"], "pose": rec.get("pose"), "colour": d["colour"],
                         "height_mm": d["height_mm"], "xy": d["xy"], "size_mm": d.get("size_mm")}))
    return out


def from_frames(folder: Path, a) -> list:
    """[(key, embedding, jpeg, meta)] for the matching boxes of saved module idle frames (embedded here)."""
    items = []
    for pj in sorted(folder.glob("*.json")):
        meta = json.loads(pj.read_text())
        img = pj.with_suffix(".jpg")
        if not img.exists():
            continue
        for k, o in enumerate(meta.get("objects") or []):
            if not _wanted(None, o.get("colour"), o.get("height_mm"), a):
                continue
            rgb = np.asarray(Image.open(img).convert("RGB"))
            items.append((f"frame:{meta['ts']:.3f}:{k}", _crop(rgb, o["bbox"]),
                          {"source": "module_frame", "obs_ts": meta["ts"], "colour": o["colour"], "height_mm": o["height_mm"]}))
    embs = clip_embed.embed_images([c for _, c, _ in items]) if items else []
    return [(key, e, _jpeg(c), m) for (key, c, m), e in zip(items, embs)]


def _wanted(xy, colour, h, a) -> bool:
    if a.colour and not clip_embed.colour_ok({a.colour}, colour):
        return False
    if a.h and not (a.h[0] <= float(h or 0.0) <= a.h[1]):
        return False
    return not (a.near and xy is not None and math.dist(xy, a.near) > a.radius)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="library label, e.g. 'green cylinder' (aliases copied from it)")
    ap.add_argument("--observations", type=Path)
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--near", type=lambda s: tuple(float(x) for x in s.split(",")))
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--colour")
    ap.add_argument("--h", type=lambda s: tuple(float(x) for x in s.split(",")), help="height range mm, e.g. 20,40")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    found = (from_observations(a.observations, a) if a.observations else []) + (from_frames(a.frames, a) if a.frames else [])
    done = json.loads(DONE.read_text()) if DONE.exists() else {}
    new = [f for f in found if f[0] not in done]
    store = Store()
    aliases = next((v.get("aliases") for v in store.search_views(np.ones(512, np.float32), k=10000, label=a.label)
                    if v.get("aliases")), [])
    print(f"{len(found)} matching detections, {len(new)} new; label {a.label!r} aliases {aliases}")
    for key, emb, jpeg, meta in new:
        print(f"  {key} h={meta['height_mm']} colour={meta['colour']} xy={meta.get('xy')}")
        if not a.dry_run:
            store.add_view(a.label, emb, jpeg, {**meta, "aliases": aliases})
            done[key] = a.label
    if not a.dry_run:
        DONE.write_text(json.dumps(done, indent=1))
        print(f"added {len(new)} views; store online={store.online}")
