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
  eval_perception.py demo                    rehearsal queries (docs/demo_queries.md) on arm-scan frames
  eval_perception.py sweep                   AGENT1's high sweep frames (kind sweep_test): timing, map, queries
  eval_perception.py square                  drop square per frame (whole or border-cut) vs full-view truth
  eval_perception.py recog                   library recognition, leave-one-frame-out, per label + confusions
  eval_perception.py openvocab               labels held out of the library, plus absent things; prints the gate
Env for capture: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY.
"""
import asyncio
import json
import os
import sys
import time
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


# Rehearsal queries (docs/demo_queries.md): (query, label the pick must be recognised as or None = must refuse,
# arm-scan frames where the target is fully visible at z=400)
DEMO = [("grasp green square", "green cube", ["033", "034", "035"]),
        # 006/007 dropped: the block is cut by the bottom edge there (an old fingertip mask hid that)
        ("pick up the yellow block", "yellow block", ["036", "038", "066", "073"]),
        ("trick or treat: candy", "kind bar", ["014", "015", "035"]),
        ("the blue cylinder", "blue cylinder", ["010", "011", "012"]),
        ("red ball", None, ["032", "035", "036"]),
        # Knives (frames 077-088); 088 (z 580) has both, so "utility knife" / "box cutter" must not pick the red one
        ("grab the utility knife", "utility knife", ["081", "083", "088"]),
        ("box cutter", "utility knife", ["082", "088"]),
        ("pick up the red knife", "red knife", ["085", "087", "088"]),
        # The user's own wording in the live test; "red box cutter" must not drift to the silver one's "box cutter"
        ("silver box cutter", "utility knife", ["081", "083", "088"]),
        ("red box cutter", "red knife", ["085", "087", "088"])]


def demo() -> None:
    """
    Check every rehearsal query with best() on saved arm-scan frames, leave-one-frame-out.

    WHY: the library was built from these frames, so each test frame's own views are removed first; what is
    left is the situation on stage (library from the scan, a new frame from the camera).
    """
    import shutil
    import tempfile

    from sorter.store import Store

    scan = EVAL.parent / "scan"
    rows = [json.loads(line) for line in (EVAL.parent / "views.jsonl").read_text().splitlines() if line.strip()]
    tmp = Path(tempfile.mkdtemp(prefix="demo_lofo_"))
    total = passed = 0
    try:
        for q, want, frames in DEMO:
            res = []
            for fr in frames:
                d = tmp / fr
                if not d.exists():
                    d.mkdir()
                    (d / "views.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("frame") != fr))
                rgb = np.asarray(Image.open(scan / f"{fr}.jpg").convert("RGB"))
                b = clip_embed.best(perception.segment_full(rgb, np.load(scan / f"{fr}_depth.npy")).objects, q,
                                    Store(uri="", data_dir=str(d)))
                ok = b["obj"] is None if want is None else (b["obj"] is not None and b["recognised"] == want)
                total, passed = total + 1, passed + ok
                res.append(f"{fr}:{'OK' if ok else 'FAIL'}(s={b['score']:.2f} m={b['margin']:.2f}{' ' + b['reason'] if b['reason'] else ''})")
            print(f"{q!r:28} -> {want or 'REFUSE':14} " + "  ".join(res))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"demo queries: {passed}/{total} passed")


def sweep() -> None:
    """
    Evaluate AGENT1's high sweep frames (json "kind": "sweep_test"): per-frame timing and recognition per height,
    then one table map from the z=500 frames and the demo queries on it.

    WHY: the library views were all taken ~378 mm from the table; at z 480-550 objects are smaller, so recognition
    confidence can drop. The sweep frames are not in the library, so this is a fair test.
    """
    from sorter import table_map
    from sorter.store import Store

    scan = EVAL.parent / "scan"
    ids = sorted(p.stem for p in scan.glob("*.json")
                 if isinstance(m := json.loads(p.read_text()), dict) and m.get("kind") == "sweep_test" and not m.get("skipped"))
    if not ids:
        sys.exit("no sweep_test frames in data/scan yet")
    store = Store()
    frames = table_map.load_scan_frames(ids)
    clip_embed.embed_images([np.zeros((50, 50, 3), np.uint8)])  # warm the model so timings are steady-state
    for f in frames:
        t = time.perf_counter()
        recs = clip_embed.recognise_all(f["rgb"], f["depth"], store)
        ms = (time.perf_counter() - t) * 1000
        print(f"{f['id']} z={f['pose']['z']:.0f}: {ms:.0f} ms, " + ", ".join(
            f"{r['label'] or '?'}({r['obj'].height_mm:.0f}mm {r['score']:.2f})" for r in recs))
    z500 = [f for f in frames if abs(f["pose"]["z"] - 500) < 15] or frames
    m = table_map.map_from_frames(z500, store)
    print(f"map from {len(z500)} frames: {len(m['objects'])} objects, {sum(1 for o in m['objects'] if o['label'])} "
          f"recognised, per frame {m['timing_ms']} ms")
    for o in sorted((o for o in m["objects"] if o["label"]), key=lambda o: o["label"]):
        print(f"   {o['label']:18} xy=({o['xy'][0]:.0f},{o['xy'][1]:.0f}) conf={o['score']:.2f} frames={o['frame_ids']}")
    for q, want, _ in DEMO + [("blue screwdriver", "blue screwdriver", []), ("red screwdriver", "red screwdriver", [])]:
        b = table_map.best_on_map(m, q, store)
        got = b["map_obj"]["label"] if b["map_obj"] else None
        print(f"  {q!r:28} -> {got} {'OK' if got == want else 'MISS'} {b['reason']}")


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


def recog() -> None:
    """
    Library recognition, leave-one-frame-out: every view with a frame is recognised (clip_embed.recognise, the real
    colour / height rules) against the views of all other frames. Prints accuracy per label and the confusions.

    WHY: the height filter decides which labels can win at all (a lying cylinder is 30 mm tall, a standing one 60),
    so any change to it is checked here on every labelled sighting, not only on the demo queries.
    """
    rows = [json.loads(line) for line in (EVAL.parent / "views.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("embedding")]
    E = np.asarray([r["embedding"] for r in rows], float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    class _Others:
        """search_views over the views of other frames only (Store scale: (1 + cos) / 2)."""

        def __init__(self, frame):
            self.keep = np.array([(r.get("frame") or r.get("obs_ts")) != frame for r in rows])

        def search_views(self, emb, k=8, label=None):
            s = (1 + E @ (np.asarray(emb, float) / np.linalg.norm(emb))) / 2
            s[~self.keep] = -1
            return [{**{k2: v for k2, v in rows[i].items() if k2 != "embedding"}, "score": float(s[i])}
                    for i in np.argsort(-s)[:k] if s[i] >= 0]

    per, wrong = {}, {}
    for i, r in enumerate(rows):
        frame = r.get("frame") or r.get("obs_ts")  # views from the observation log have the record's ts instead
        if not frame:
            continue
        lab, _, _ = clip_embed.recognise(E[i], _Others(frame), colour=r.get("colour"), height_mm=r.get("height_mm"))
        ok = per.setdefault(r["label"], [0, 0])
        ok[0] += lab == r["label"]
        ok[1] += 1
        if lab != r["label"]:
            wrong[(r["label"], lab)] = wrong.get((r["label"], lab), 0) + 1
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    for lab, (a, n) in sorted(per.items()):
        print(f"  {lab:18s} {a:3d}/{n:<3d}")
    print("  confusions:", ", ".join(f"{a} -> {b}: {n}" for (a, b), n in sorted(wrong.items(), key=lambda t: -t[1])) or "none")
    print(f"recognition (leave-one-frame-out): {tot[0]}/{tot[1]} = {tot[0] / max(tot[1], 1):.1%}")


# Open-vocabulary eval: a colourless noun per label (several labels may fit it, so its WRONG count is an upper bound)
OPEN_NOUN = {"blue block": "block", "yellow block": "block", "yellow cylinder": "cylinder", "blue cylinder": "cylinder",
             "green cylinder": "cylinder", "kind bar": "snack bar", "granola bar": "snack bar", "red screwdriver": "screwdriver",
             "blue screwdriver": "screwdriver", "black tape": "tape", "green cube": "cube", "computer mouse": "mouse",
             "red knife": "knife", "utility knife": "knife"}
OPEN_ABSENT = ("banana", "red ball", "scissors", "mug")  # must be refused; the gate's thresholds were tuned on these
OPEN_ABSENT_UNSEEN = ("apple", "spoon", "phone", "pencil", "orange", "hammer", "toothbrush")  # not tuned on (a real water bottle joined the table)


def openvocab() -> None:
    """
    Open vocabulary, offline: treat each library label as an object the system has never seen.

    For label L the library loses every view of L and of the test frame, then best() is asked for L on real arm-scan
    frames: full name / colourless noun / no library at all. The right object is the one whose crop matches L's stored
    view in that frame. Wrong picks are the dangerous outcome, refusals are only unhelpful. Then absent things are
    asked for on the same frames, with the full library and with L held out (a really unknown object on the table).
    GATE: full name right >= 55 of 65, WRONG <= 3, the tuned absent queries never picked.
    """
    import collections

    scan = EVAL.parent / "scan"
    rows = [json.loads(line) for line in (EVAL.parent / "views.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("embedding")]
    E = np.asarray([r["embedding"] for r in rows], float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    class Lib:
        """Store stand-in: the library without one label and without one frame's views."""

        def __init__(self, label, frame):
            self.keep = np.array([r["label"] != label and r.get("frame") != frame for r in rows])

        def search_views(self, emb, k=8, label=None):
            s = (1 + E @ (np.asarray(emb, float) / np.linalg.norm(emb))) / 2
            s[~self.keep] = -1
            if label:
                s[[r["label"] != label for r in rows]] = -1
            return [{**{a: b for a, b in rows[i].items() if a != "embedding"}, "score": float(s[i])}
                    for i in np.argsort(-s)[:k] if s[i] >= 0]

        def list_labels(self):
            seen = {}
            for r, k in zip(rows, self.keep):
                if k:
                    seen.setdefault(r["label"], {r["label"]}).update(r.get("aliases") or [])
            return [sorted(v) for v in seen.values()]

    cases, frames = collections.defaultdict(list), {}
    for i, r in enumerate(rows):
        if str(r.get("frame", "")).isdigit() and (scan / f"{r['frame']}_depth.npy").exists():
            cases[r["label"]].append(i)

    def frame(fr):
        if fr not in frames:
            rgb = np.asarray(Image.open(scan / f"{fr}.jpg").convert("RGB"))
            objs = perception.segment_full(rgb, np.load(scan / f"{fr}_depth.npy")).objects
            frames[fr] = (objs, clip_embed.embed_images([o.crop for o in objs]) if objs else [])
        return frames[fr]

    def recs(objs, embs, lib):
        """recognise_all() output for these objects under this library (the crops are embedded once per frame)."""
        out = []
        for o, e in zip(objs, embs):
            lab, names, sc = clip_embed.recognise(e, lib, colour=o.colour, height_mm=o.height_mm)
            out.append({"obj": o, "label": lab, "names": names, "score": sc, "emb": e})
        return out

    tally = {c: collections.Counter() for c in ("full name", "noun only", "no library")}
    per, not_right, right_s, wrong_s, ambiguous = {}, [], [], [], collections.Counter()
    absent = {"tuned": collections.Counter(), "unseen": collections.Counter()}
    picks, asked = collections.Counter(), collections.Counter()
    seen_full = set()
    cand_stats = {"unknown (held out)": collections.Counter(), "known (full library)": collections.Counter()}
    for label, idx in sorted(cases.items()):
        step = max(1, len(idx) // 5)
        for i in idx[::step][:5]:
            fr = rows[i]["frame"]
            objs, embs = frame(fr)
            if not len(objs):
                continue
            match = embs @ E[i]
            if match.max() < 0.97:  # segmentation differs from the scan's: cannot tell which object is the target
                continue
            target = objs[int(match.argmax())]
            held = Lib(label, fr)
            rh = recs(objs, embs, held)
            for cond, q, st in (("full name", label, held), ("noun only", OPEN_NOUN.get(label, label), held), ("no library", label, None)):
                b = clip_embed.best(objs, q, st, recognised=rh)
                out = "refused" if b["obj"] is None else "right" if b["obj"] is target else "WRONG"
                tally[cond][out] += 1
                per.setdefault(label, {}).setdefault(cond, collections.Counter())[out] += 1
                if cond == "full name":
                    if out != "right":
                        not_right.append(f"{label}@{fr}: {out}" + (f" ({b['reason']})" if b["reason"] else f" (picked {b['obj'].colour} h={b['obj'].height_mm:.0f})"))
                    if b["obj"] is not None:
                        (right_s if out == "right" else wrong_s).append(b["score"])
                        ambiguous[(out, bool(b["ambiguous"]))] += 1
            # candidates(): where does the target rank, and did it rescue what best() refused?
            full = Lib(None, fr)
            rf = recs(objs, embs, full)
            for cond, st, rr in (("unknown (held out)", held, rh), ("known (full library)", full, rf)):
                c = clip_embed.candidates(objs, label, st, recognised=rr)
                pos = next(k for k, x in enumerate(c) if x["obj"] is target)
                b = clip_embed.best(objs, label, st, recognised=rr)
                t = cand_stats[cond]
                t["n"] += 1
                t["top1"] += pos == 0
                t["top2"] += pos <= 1
                t["best right"] += b["obj"] is target
                t["rescued"] += b["obj"] is None and pos <= 1
                t["best wrong, cand right"] += b["obj"] is not None and b["obj"] is not target and pos == 0
            libs = [("held out", held, rh)]
            if fr not in seen_full:  # the full-library condition once per frame
                seen_full.add(fr)
                full = Lib(None, fr)
                libs.append(("full", full, recs(objs, embs, full)))
            for name, lib, rr in libs:
                for group, qs in (("tuned", OPEN_ABSENT), ("unseen", OPEN_ABSENT_UNSEEN)):
                    for q in qs:
                        b = clip_embed.best(objs, q, lib, recognised=rr)
                        asked[(group, name)] += 1
                        if b["obj"] is not None:
                            absent[group][name] += 1
                            picks[(q, name, b["obj"].colour, round(b["obj"].height_mm), fr)] += 1
    for c, t in tally.items():
        print(f"{c:11s} right {t['right']:3d}  WRONG {t['WRONG']:3d}  refused {t['refused']:3d}  of {t['right'] + t['WRONG'] + t['refused']}")
    print("per label [right/WRONG/refused] full name | noun only | no library")
    for lab, d in per.items():
        print(f"  {lab:18s}", " | ".join(f"{d[c]['right']}/{d[c]['WRONG']}/{d[c]['refused']}" for c in tally))
    print("full name, not right:", "; ".join(not_right) or "none")
    print("full name picks by (outcome, ambiguous):", dict(ambiguous))
    if right_s:
        print(f"score when right: median {np.median(right_s):.3f} min {min(right_s):.3f}")
    if wrong_s:
        print(f"score when WRONG: median {np.median(wrong_s):.3f} max {max(wrong_s):.3f}")
    for group in absent:
        print(f"absent {group}: picked " + ", ".join(f"{absent[group][n]}/{asked[(group, n)]} ({n})" for n in ("full", "held out")))
    if picks:
        print("absent picks (query, library, picked colour, height, frame):", dict(picks))
    for cond, t in cand_stats.items():
        print(f"candidates, {cond}: target first {t['top1']}/{t['n']}, in top 2 {t['top2']}/{t['n']}; best() right {t['best right']}; "
              f"best() refused but target in top 2 {t['rescued']}; best() wrong but target first {t['best wrong, cand right']}")
    t = tally["full name"]
    ok = t["right"] >= 55 and t["WRONG"] <= 3 and not sum(absent["tuned"].values())
    print(f"GATE {'PASS' if ok else 'FAIL'}: full name right {t['right']} >= 55, WRONG {t['WRONG']} <= 3, tuned absent picks {sum(absent['tuned'].values())} == 0")


def square() -> None:
    """
    The drop square on every arm-scan frame that sees it: each view alone, and all views folded in time order.

    WHY: the user moves the paper freely, so a border-cut view must still place the whole square; this measures how
    far each single view's estimate is from the truth. Truth per placement of the paper ("era": the fold restarted,
    i.e. the paper moved or turned) = median of that era's views that saw all four tape edges.
    """
    import math

    from sorter import drop_square

    scan = EVAL.parent / "scan"
    recs = []
    for pj in sorted(scan.glob("[0-9][0-9][0-9].json")):
        meta = json.loads(pj.read_text())
        if not isinstance(meta, dict) or "pose" not in meta:
            continue
        rgb = np.asarray(Image.open(pj.with_suffix(".jpg")).convert("RGB"))
        hit = perception.find_drop_square(rgb, np.load(scan / f"{pj.stem}_depth.npy"))
        if hit:
            intr = {**perception.INTRINSICS, **meta.get("intrinsics", {})}
            recs.append((meta.get("ts", 0.0), pj.stem, drop_square.observe(hit, meta["pose"], intr), meta["pose"]))
    recs.sort()
    # Fold in time order; an era is one placement of the paper (drop_square restarts on a move/turn: since_ts)
    sq, eras, folded = None, {}, {}
    for ts, n, r, _ in recs:
        sq = drop_square.update(sq, r, ts)
        eras[n], folded[n] = sq["since_ts"], dict(sq)
    single = {n: drop_square.update(None, r, ts) for ts, n, r, _ in recs}
    truth = {}
    for e in set(eras.values()):
        full = [single[n]["xy"] for n in eras if eras[n] == e and not single[n]["partial"]]
        truth[e] = tuple(np.median(full, axis=0)) if full else None
    errs = []
    print(f"{'frame':5} {'pose':>22} {'sides':8} {'conf':>4} {'yaw':>6} {'single-view xy':>16} {'err':>5}  {'folded xy':>16} {'err':>5}")
    for ts, n, r, pose in recs:
        s, f, t = single[n], folded[n], truth[eras[n]]
        e1 = math.dist(s["xy"], t) if t else float("nan")
        e2 = math.dist(f["xy"], t) if t else float("nan")
        if s["partial"] and t:
            errs.append(e1)
        print(f"{n:5} ({pose['x']:4.0f},{pose['y']:5.0f},{pose['z']:3.0f},{pose['theta']:+4.0f}) {''.join(sorted(s['sides'])):8} "
              f"{s['confidence']:4.2f} {s['yaw_deg']:+6.1f} {str(s['xy']):>16} {e1:5.1f}  {str(f['xy']):>16} {e2:5.1f}"
              f"{'' if s['partial'] else '  (full view)'}")
    for e, t in sorted(truth.items()):
        print(f"era since ts {e:.0f}: truth {np.round(t, 1).tolist() if t else None} from the full views")
    if errs:
        print(f"partial single views: {len(errs)}, centre error median {np.median(errs):.1f} mm, max {max(errs):.1f} mm")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        asyncio.run(capture(sys.argv[2]))
    elif cmd == "annotate":
        annotate(sys.argv[2])
    elif cmd == "label":
        label(sys.argv[2], sys.argv[3:])
    elif cmd == "demo":
        demo()
    elif cmd == "sweep":
        sweep()
    elif cmd == "square":
        square()
    elif cmd == "recog":
        recog()
    elif cmd == "openvocab":
        openvocab()
    elif cmd == "report":
        # report [--store] [--absent "green square" ...]
        rest = sys.argv[2:]
        absent = rest[rest.index("--absent") + 1:] if "--absent" in rest else []
        report("--store" in rest, absent)
