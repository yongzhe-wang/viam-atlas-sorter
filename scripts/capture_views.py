"""
Build the object library: a person places an object, we capture several views, they type its label.

WHY: rank() gets much more reliable when each object has a few labelled views in the store (the store
vote separates look-alikes such as the yellow block vs the yellow cylinder, which CLIP alone ranks
within 0.01). The arm stays still: the person rotates or moves the object between views, so this
script only reads the camera and never needs the motion owner.

Flow per object: place it near the image centre -> Enter -> capture -> pick the object (nearest the
centre, not cut by the border) -> repeat for N views -> type label (+ aliases) -> Store.add_view per view.

Usage:
  capture_views.py                       live camera, 4 views per object, until an empty label
  capture_views.py --views 6
  capture_views.py --files a.jpg b.jpg --label "yellow block"   offline: frames saved with *_depth.npy
  --data-dir DIR                         write the JSONL store somewhere else (tests)
Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY; MONGODB_URI optional (Store falls back to JSONL).
"""
import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import clip_embed, perception  # noqa: E402
from sorter.store import Store  # noqa: E402


def pick_centre_object(objs: list, shape: tuple):
    """The object nearest the image centre that is not cut by the border, or None."""
    h, w = shape[:2]
    cand = [o for o in objs if not o.touches_edge]
    if not cand:
        return None
    return min(cand, key=lambda o: (o.centroid_px[0] - w / 2) ** 2 + (o.centroid_px[1] - h / 2) ** 2)


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def save_views(store: Store, label: str, aliases: list, views: list) -> list:
    """Embed every captured crop and write one view doc each; returns the new ids."""
    embs = clip_embed.embed_images([o.crop for o, _ in views])
    ids = []
    for (o, meta), e in zip(views, embs):
        ids.append(store.add_view(label, e, jpeg_bytes(o.crop), {
            **meta, "aliases": aliases, "source": "scan", "colour": o.colour, "height_mm": o.height_mm,
            "length_px": o.length_px, "width_px": o.width_px}))
    return ids


async def live(store: Store, n_views: int) -> None:
    """Interactive capture loop on the live camera (read-only)."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera
    from viam.robot.client import RobotClient

    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    robot = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    arm, cam = Arm.from_robot(robot, "arm"), Camera.from_robot(robot, "cam")
    try:
        while True:
            views = []
            while len(views) < n_views:
                input(f"view {len(views) + 1}/{n_views}: place/rotate ONE object near the centre, then Enter ")
                if await arm.is_moving():
                    print("  arm is moving, wait for it to stop")
                    continue
                p = await arm.get_end_position()
                res = await cam.get_images()
                imgs = res[0] if isinstance(res, tuple) else res
                rgb = depth = None
                for im in imgs:
                    if im.mime_type == "image/jpeg":
                        rgb = np.asarray(Image.open(io.BytesIO(im.data)).convert("RGB"))
                    elif im.data[:8] == b"DEPTHMAP":
                        w = int.from_bytes(im.data[8:16], "big")
                        h = int.from_bytes(im.data[16:24], "big")
                        depth = np.frombuffer(im.data[24:], dtype=">u2").reshape(h, w)
                if rgb is None or depth is None:
                    print("  no colour+depth frame, try again")
                    continue
                o = pick_centre_object(perception.segment_full(rgb, depth).objects, rgb.shape)
                if o is None:
                    print("  no object found near the centre (is it > 8 mm tall and fully in view?)")
                    continue
                print(f"  got it: {o.colour}, {o.height_mm:.0f} mm tall, {o.length_px:.0f}x{o.width_px:.0f} px")
                views.append((o, {"view": len(views), "pose": {"x": p.x, "y": p.y, "z": p.z, "theta": p.theta}}))
            label = input("label (empty = finish): ").strip()
            if not label:
                return
            aliases = [a.strip() for a in input("aliases, comma separated (optional): ").split(",") if a.strip()]
            print(f"  saved {len(save_views(store, label, aliases, views))} views of {label!r} (online={store.online})")
    finally:
        await robot.close()


def offline(store: Store, files: list, label: str) -> None:
    """Same as live, from saved frames (each needs a matching *_depth.npy next to it)."""
    views = []
    for f in files:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        depth = np.load(str(f).rsplit(".", 1)[0] + "_depth.npy")
        o = pick_centre_object(perception.segment_full(rgb, depth).objects, rgb.shape)
        if o is None:
            print(f"{f}: no centre object")
            continue
        views.append((o, {"view": len(views), "file": str(f)}))
    print(f"saved {len(save_views(store, label, [], views))} views of {label!r} (online={store.online})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--label")
    ap.add_argument("--data-dir")
    a = ap.parse_args()
    st = Store(data_dir=a.data_dir)
    if a.files:
        offline(st, a.files, a.label or sys.exit("--label is required with --files"))
    else:
        asyncio.run(live(st, a.views))
