"""
Measure perception on real frames: segmentation recall and text-rank top-1.

WHY: "a scanned object placed anywhere is recognised" needs a number, not a demo that worked once.
Frames come from the observe pose (where AGENT1 parks the arm), objects are labelled by hand, and the
report compares CLIP alone with CLIP + colour/shape cues so each part's effect is measured.

Usage (data in data/eval/, gitignored):
  eval_perception.py capture NAME            read-only; refuses unless the arm is idle at the observe pose
  eval_perception.py annotate NAME           writes NAME_ann.png with object ids for labelling
  eval_perception.py label NAME 0="green square" 2=cup missed=1
  eval_perception.py report [--store] [--absent "green square" ...]
Env for capture: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import clip_embed, perception  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval"
OBSERVE = (250.0, 0.0, 400.0)  # AGENT1's park/observe pose (flange, mm)
OBSERVE_TOL_MM = 15.0


async def capture(name: str) -> None:
    """Save colour + depth + pose. Only camera/pose reads: this script never commands motion."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera
    from viam.robot.client import RobotClient

    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    robot = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    try:
        arm = Arm.from_robot(robot, "arm")
        p = await arm.get_end_position()
        off = max(abs(p.x - OBSERVE[0]), abs(p.y - OBSERVE[1]), abs(p.z - OBSERVE[2]))
        if await arm.is_moving() or off > OBSERVE_TOL_MM:
            sys.exit(f"arm not idle at the observe pose (at {p.x:.0f},{p.y:.0f},{p.z:.0f}); ask AGENT1 to park it")
        res = await Camera.from_robot(robot, "cam").get_images()
        imgs = res[0] if isinstance(res, tuple) else res
        EVAL.mkdir(parents=True, exist_ok=True)
        for im in imgs:
            if im.mime_type == "image/jpeg":
                (EVAL / f"{name}.jpg").write_bytes(im.data)
            elif im.data[:8] == b"DEPTHMAP":
                w = int.from_bytes(im.data[8:16], "big")
                h = int.from_bytes(im.data[16:24], "big")
                np.save(EVAL / f"{name}_depth.npy", np.frombuffer(im.data[24:], dtype=">u2").reshape(h, w).astype(np.uint16))
        (EVAL / f"{name}_pose.json").write_text(json.dumps({"x": p.x, "y": p.y, "z": p.z, "theta": p.theta}))
        print("captured", name)
    finally:
        await robot.close()


def load(name: str):
    rgb = np.asarray(Image.open(EVAL / f"{name}.jpg").convert("RGB"))
    return rgb, np.load(EVAL / f"{name}_depth.npy")


def annotate(name: str) -> None:
    """Draw each object's mask outline, id, colour and height so a person can label it."""
    rgb, depth = load(name)
    seg = perception.segment_full(rgb, depth)
    img = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(img)
    for o in seg.objects:
        dr.rectangle(o.bbox, outline=(255, 0, 255), width=3)
        dr.text((o.bbox[0] + 4, o.bbox[1] + 4), f"{o.id} {o.colour} {o.height_mm:.0f}mm", fill=(255, 0, 255))
    for it in seg.intruders:
        dr.rectangle(it["bbox"], outline=(255, 0, 0), width=3)
    img.save(EVAL / f"{name}_ann.png")
    print(f"{name}: {len(seg.objects)} objects, {len(seg.intruders)} intruders, align={seg.align}")
    for o in seg.objects:
        print(f"  {o.id}: bbox={o.bbox} colour={o.colour} h={o.height_mm:.0f}mm edge={o.touches_edge}")


def label(name: str, args: list) -> None:
    """Store {object id: label} and how many real objects segmentation missed."""
    labels, missed = {}, 0
    for a in args:
        k, v = a.split("=", 1)
        if k == "missed":
            missed = int(v)
        else:
            labels[int(k)] = v.strip('"')
    (EVAL / f"{name}_labels.json").write_text(json.dumps({"labels": labels, "missed": missed}))
    print("labelled", name, labels, "missed", missed)


def report(use_store: bool, absent: list) -> None:
    """
    Segmentation recall, and top-1 for each labelled object queried by its own label.

    Modes: CLIP only, CLIP + colour/shape cues, and with the store best() as the worker uses it (labelled
    views vote; a refusal counts as a miss). `absent`: queries for objects NOT on the table (e.g. "green
    square"); best() should refuse them, and a pick there is counted as a false pick.
    """
    store = None
    if use_store:
        from sorter.store import Store

        store = Store()
        print(f"store: online={store.online}")
    found = missed = 0
    modes = ["clip only", "clip+cues"] + (["best()+store"] if store is not None else [])
    hits = {m: 0 for m in modes}
    n_q = 0
    for lf in sorted(EVAL.glob("*_labels.json")):
        name = lf.name[: -len("_labels.json")]
        lab = json.loads(lf.read_text())
        rgb, depth = load(name)
        objs = perception.segment_full(rgb, depth).objects
        labels = {int(k): v for k, v in lab["labels"].items()}
        found += len(labels)
        missed += lab["missed"]
        crops = clip_embed.embed_images([o.crop for o in objs])
        for oid, text in labels.items():
            n_q += 1
            sims = crops @ clip_embed.embed_text(clip_embed.clean_query(text))
            got = {"clip only": objs[int(np.argmax(sims))].id, "clip+cues": clip_embed.rank(objs, text)[0][0].id}
            if store is not None:
                b = clip_embed.best(objs, text, store)
                got["best()+store"] = b["obj"].id if b["obj"] else None
            for m in modes:
                hits[m] += got[m] == oid
            print(f"  {name}: {text!r} want {oid} -> " + ", ".join(f"{m}={got[m]}" for m in modes))
        for text in absent:
            b = clip_embed.best(objs, text, store)
            print(f"  {name}: absent {text!r} -> {'REFUSED: ' + b['reason'] if b['obj'] is None else 'FALSE PICK ' + str(b['obj'].id)}")
    total = found + missed
    print(f"segmentation recall: {found}/{total} = {found / max(total, 1):.0%}")
    for m in modes:
        print(f"top-1 ({m}): {hits[m]}/{n_q} = {hits[m] / max(n_q, 1):.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        asyncio.run(capture(sys.argv[2]))
    elif cmd == "annotate":
        annotate(sys.argv[2])
    elif cmd == "label":
        label(sys.argv[2], sys.argv[3:])
    elif cmd == "report":
        # report [--store] [--absent "green square" ...]
        rest = sys.argv[2:]
        absent = rest[rest.index("--absent") + 1:] if "--absent" in rest else []
        report("--store" in rest, absent)
