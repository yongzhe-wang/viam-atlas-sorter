"""
Pick ANY object on the table (sorter.perception.Obj) and put it in the blue tape square; clear the square.

WHY: the demo is "grasp green square" with whatever is on the table, and the square must be emptied
between runs without a person. The arm mechanics (servo, depth heights, reconnect, intruder pause,
square search) live in scripts/pick_place_auto.py; this module adds object choice and tracking for
arbitrary objects, the in-square check, and the reset.

Flow, execute(text, target=None, target_xy=None): object from a given frame (target), from one look at a
table xy (target_xy, e.g. the table map), or found by looking around -> square from that same frame (else the
last known square, else search while carrying) -> ObjTracker -> ppa.pick -> ppa.place_in_square ->
objects_in_hole check -> Store.log_attempt -> back to the observe pose.
sweep_table(): theta 0 frames over the whole reachable table for sorter.table_map (AGENT3) to build a map.
Flow, clear_square(): find square -> each object in the hole: pick (<= 2 tries) -> place_at random free spot;
failed pick -> wipe (<= 2 passes) -> verify hole empty by depth -> Store.log_attempt(mode="reset").
Contract: docs/EXEC_PLAN.md (Command flow, Reset).
"""
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402  arm library (Rig, pick, place_in_square, place_at, ...)

from sorter import clip_embed, perception  # noqa: E402

MAX_OPEN_MM = 78.0  # jaws open ~84 mm; keep a margin so the fingers clear the sides
COLOUR_TOL = 60.0  # RGB distance that still counts as the object at close range (grey/black/white objects)
HUE_TOL = 9  # PIL hue units (~13 deg) for saturated objects: hue barely changes when the block looks brighter up close
SAT_REF = 90  # PIL saturation above which an object is "coloured" and tracked by hue
# Where to look for objects, flange x/y mm: home first, then the square-search grid
OBSERVE_XY = [ppa.HOME_XY] + ppa.SEARCH_XY
# Random drop area for reset, flange x/y mm (see run_trials.RESET_*); place_at refuses occupied/square spots
RESET_X = (230.0, 380.0)
RESET_Y = (-220.0, 220.0)
WIPE_CLEAR_MM = 10.0  # fingertips this far above the table during a wipe
WIPE_STEP_MM = 20.0  # wipe in short moves = slow, and interruptible between steps
WORKER_ID = "sfhack4-worker"  # heartbeat / frame id; AGENT 2's Viam module uses "module:<name>"
# What the arm is doing right now, for heartbeats: the current attempt's times dict ("at" = phase) + last error
STATE = {"phase": "idle", "times": None, "error": None}
PHASES = {"square": "observe", "find": "observe", "check": "place", "start": "observe"}
# Flange x/y over the square's centre from the latest locate/place, and when; used when a frame shows no square
LAST_SQUARE = {"xy": None, "ts": 0.0}
SQUARE_CACHE_S = 600.0
# Whole-table sweep: theta 0 poses at SWEEP_Z. The first layout (z 500, rows at flange x 280/460) saw x >= 138
# only, so the green cube near the base (166, 118) was cut by the image edge and dropped, and no frame held the
# whole square. Raised to z 580 (user: "raise a little higher"): camera ~562 mm above the table, one frame covers
# ~445 mm in x by ~790 mm in y, centred ~(48, 33) mm from the flange (perception.JAW_IN_CAM_MM). Rows 170/400 and
# columns -180/+180 overlap ~210 / ~430 mm: green cube, KIND bar corner and the whole square (~(370, 84),
# 137 x 188 mm) each lie fully inside at least one of the 4 frames.
SWEEP_Z = 580.0
SWEEP_XY = [(x, y) for x in (170.0, 400.0) for y in (-180.0, 180.0)]
# Every sweep frame is saved here for offline debugging (AGENT3); SORTER_DATA_DIR overrides (module data dir)
SWEEP_DIR = Path(os.environ.get("SORTER_DATA_DIR") or Path(__file__).resolve().parent.parent / "data") / "sweeps"


def phase_now(rig) -> str:
    """Heartbeat phase: link/person states first, then the step in progress, else idle."""
    if rig.status:
        return rig.status
    at = (STATE.get("times") or {}).get("at")
    return PHASES.get(at, at) if at else STATE["phase"]


def publish_frame(store, rgb: np.ndarray, seg, phase: str, scores: dict = None) -> None:
    """Latest frame + boxes for the dashboard: targets, ignored (excluded) objects, and intruders."""
    if not hasattr(store, "put_frame"):
        return
    objs = [{"id": o.id, "bbox": list(o.bbox), "label": o.colour, "height_mm": round(o.height_mm),
             **({"score": round(scores[o.id], 3)} if scores and o.id in scores else {})} for o in seg.objects]
    objs += [{"id": o.id, "bbox": list(o.bbox), "label": why, "ignored": True} for o, why in seg.excluded]
    meta = {"objects": objs, "intruders": [{"bbox": list(i["bbox"]), "reason": i["reason"]} for i in seg.intruders],
            "phase": phase}
    try:
        store.put_frame(WORKER_ID, jpeg(rgb), meta)
    except Exception as e:
        print(f"put_frame failed: {e}")


async def apply_limits(rig) -> None:
    """Obstacle keep-outs (ppa.apply_workspace_limits) applied to OBSERVE_XY and the reset rectangle; logged."""
    await ppa.apply_workspace_limits(rig)
    OBSERVE_XY[:] = [q for q in OBSERVE_XY if ppa.allowed(*q)]
    global RESET_X, RESET_Y
    for x0, x1, y0, y1 in ppa.KEEP_OUT:
        # A keep-out spanning the whole rectangle in one axis cuts the other axis; partial ones are point-checked
        if y0 <= RESET_Y[0] and y1 >= RESET_Y[1] and x1 > RESET_X[0] and x0 < RESET_X[1]:
            RESET_X = (x1, RESET_X[1]) if x0 <= RESET_X[0] else (RESET_X[0], x0)
        if x0 <= RESET_X[0] and x1 >= RESET_X[1] and y1 > RESET_Y[0] and y0 < RESET_Y[1]:
            RESET_Y = (y1, RESET_Y[1]) if y0 <= RESET_Y[0] else (RESET_Y[0], y0)
    print(f"  workspace: {len(OBSERVE_XY)} observe points, reset x[{RESET_X[0]:.0f},{RESET_X[1]:.0f}] "
          f"y[{RESET_Y[0]:.0f},{RESET_Y[1]:.0f}]")


def intr(rig) -> dict:
    return {"fx": rig.fx, "fy": rig.fy, "cx": rig.cx, "cy": rig.cy}


def pose_dict(p) -> dict:
    return {"x": p.x, "y": p.y, "z": p.z, "theta": p.theta}


def start_recording(rig, store, doc: dict) -> None:
    """
    Give the attempt its Store id and record every frame the arm code already grabs into that attempt's clip.

    WHY: the dashboard's Replay plays these frames (AGENT 2's put_clip_frame); no extra photos are taken, so
    recording costs only JPEG encoding. No-op when the Store has no clip support.
    """
    rig.on_frame = None
    if not (hasattr(store, "new_attempt_id") and hasattr(store, "put_clip_frame")):
        return
    aid = store.new_attempt_id()
    doc["_id"] = aid
    t0 = time.monotonic()

    def on_frame(phase: str, rgb: np.ndarray, meta: dict) -> None:
        m = dict(meta, t=round(time.monotonic() - t0, 2))
        if rig.last_pose is not None:
            m["pose"] = {k: round(v, 1) for k, v in pose_dict(rig.last_pose).items()}
        store.put_clip_frame(aid, phase, jpeg(rgb), m)

    rig.on_frame = on_frame


def finish(rig, store, doc: dict) -> None:
    """Log the attempt under its recording id (Store keeps a given _id) and stop recording."""
    rig.on_frame = None
    doc["_id"] = store.log_attempt(dict(doc))


async def back_to_observe(rig) -> None:
    """
    End every task at the observe pose (HOME_XY, ALIGN_Z, theta 0) with whatever it holds already handled.

    WHY: the module's idle loop takes its frames there (within +-15 mm / +-1 deg), so the next command's photo
    is current. Never raises: a failed return is reported by the next command, not by this one.
    """
    try:
        await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
    except Exception as e:
        print(f"  could not return to the observe pose: {e}")


def square_from_frame(rgb: np.ndarray, depth: np.ndarray, pose: dict, intrinsics: dict):
    """
    Square hole centre seen in one theta-0 frame: {"px": [u, v], "xy": [x, y]}, or None if not enclosed.

    xy is the flange x/y that puts the jaws over the hole centre (perception.world_xy at the table depth),
    so a place can go straight there instead of searching. Also refreshes LAST_SQUARE.
    """
    if depth is None or abs(float(pose.get("theta", 0.0))) > 1.0:
        return None
    hit = ppa.find_square(rgb)
    if hit is None:
        return None
    table, _ = ppa.table_and_near(depth)
    x, y = (float(v) for v in perception.world_xy(pose, intrinsics, hit[0], hit[1], table))
    LAST_SQUARE.update(xy=(x, y), ts=time.time())
    return {"px": [int(round(hit[0])), int(round(hit[1]))], "xy": [round(x, 1), round(y, 1)]}


def record_sightings(store, rgb, depth, seg, pose: dict, intrinsics: dict, source: str) -> None:
    """Tell the object memory what this frame saw (AGENT3's where_is follows moved objects); never fails."""
    try:
        recs = clip_embed.recognise_all(rgb, depth, store, seg=seg)
        clip_embed.record_sightings(store, recs, pose, intrinsics, source=source)
    except Exception as e:
        print(f"  sightings not recorded: {e}")


class ObjTracker:
    """
    Finder (rgb -> (u, v, yaw, area) or None) that follows ONE object of any colour, and only that object.

    WHY (2026-10-29, "grasp blue screwdriver" put the BLUE BLOCK in the square): (1) at close range the hue
    tracker matched only the handle's blue parts and centred the jaws on the blue end cap, so the grab missed;
    (2) the re-alignment after the miss started from that close-range pixel, and in the far frame the nearest
    object to it was the blue block; (3) nothing checked identity before closing. So:
    - far and close positions are kept apart: far frames are matched against the last FAR position only;
    - far frames (depth valid): every object incl. edge-cut ones (a turned screwdriver's shaft leaves the frame);
      when more than one is within reach of the last position, best() with the request text decides; with no
      candidate in reach it re-acquires by recognition over the whole frame;
    - close frames (depth blind < ~280 mm): pixels of ANY of the object's dominant colours (a blue+black handle
      is matched as a whole, not just its blue cap), nearest blob to the last position;
    - executor.verify_target() re-checks identity by recognition before every descent (ppa.pick verify=).
    """

    MAX_JUMP_PX = 260.0

    def __init__(self, rig, obj, rgb: np.ndarray, text: str = None, store=None):
        self.rig, self.text, self.store = rig, text, store
        self.far_uv = obj.centroid_px  # last position in a far frame
        self.close_uv = None  # last position in a close frame; cleared when the arm goes back up
        self.close_target = None  # jaw pixel for the close servo, set by ppa.pick before it starts
        self.uv = self.far_uv
        self._learn(rgb, obj.mask)

    def back_to_far(self) -> None:
        """Called by ppa.pick after a lift: the next frame is a far one, compare it with the last FAR sighting."""
        self.close_uv = None
        self.uv = self.far_uv

    def _learn(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        """
        Close-range colour model from the object's latest far view: one hue for its saturated part (hue barely
        changes when the object looks brighter up close) and one RGB median for its unsaturated part (black/grey/
        white), each kept if it covers >= 15% of the object. A blue+black screwdriver gets both.
        """
        hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))[mask].astype(float)
        px = rgb[mask].astype(float)
        sat = (hsv[:, 1] > SAT_REF) & (hsv[:, 2] > 50)
        self.hue = float(np.median(hsv[sat, 0])) if sat.mean() >= 0.15 else None
        self.rgb_ref = np.median(px[~sat], axis=0) if (~sat).mean() >= 0.15 else None
        if self.hue is None and self.rgb_ref is None:
            self.rgb_ref = np.median(px, axis=0)

    def _close_mask(self, rgb: np.ndarray) -> np.ndarray:
        m = np.zeros(rgb.shape[:2], bool)
        if self.hue is not None:
            hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
            dh = np.abs(hsv[..., 0] - int(self.hue))
            dh = np.minimum(dh, 256 - dh)  # hue is circular
            m |= (dh <= HUE_TOL) & (hsv[..., 1] > SAT_REF * 0.6) & (hsv[..., 2] > 50)
        if self.rgb_ref is not None:
            m |= np.linalg.norm(rgb.astype(float) - self.rgb_ref, axis=2) < COLOUR_TOL
        m[ppa.FINGER_ROW:] = False
        return ndimage.binary_closing(ndimage.binary_opening(m, iterations=2), iterations=4)

    def _far(self, rgb: np.ndarray, depth: np.ndarray):
        seg = perception.segment_full(rgb, depth)
        objs = list(seg.objects) + [o for o, _ in seg.excluded]
        if not objs:
            return None, None

        def dist(o):
            return math.hypot(o.centroid_px[0] - self.far_uv[0], o.centroid_px[1] - self.far_uv[1])

        near = [o for o in objs if dist(o) <= self.MAX_JUMP_PX]
        pick = min(near, key=dist) if near else None
        if len(near) > 1 and self.text:
            # Several objects in reach (the screwdriver next to the blue block): identity by name, not distance
            named = self._best(near)
            if named is not None:
                pick = named
        if pick is None:
            pick = self._best(objs)  # lost: re-acquire by recognition over the whole frame
            if pick is not None and dist(pick) > 2 * self.MAX_JUMP_PX:
                pick = None
            if pick is not None:
                print("  re-acquired the target by recognition")
        return pick, objs

    def _best(self, objs: list):
        if not self.text or not objs:
            return None
        try:
            return clip_embed.best(objs, self.text, self.store).get("obj")
        except Exception as e:
            print(f"  recognition failed: {e}")
            return None

    def __call__(self, rgb: np.ndarray):
        depth = self.rig.last_depth
        if depth is not None and (depth > 0).mean() > 0.6:
            o, _ = self._far(rgb, depth)
            if o is None:
                return None
            self.far_uv = self.uv = o.centroid_px
            self.close_uv = None
            if not o.touches_edge:
                self._learn(rgb, o.mask)
            return (o.centroid_px[0], o.centroid_px[1], o.yaw_deg, int(o.mask.sum()))
        ref = self.close_uv or self.far_uv
        lab, n = ndimage.label(self._close_mask(rgb))
        cands, self._labels = [], []
        for k in range(1, n + 1):
            vs, us = np.nonzero(lab == k)
            if len(us) < 1500:
                continue
            self._labels.append(k)
            cu, cv = us.mean(), vs.mean()
            w, vec = np.linalg.eigh(np.cov(np.vstack([us - cu, vs - cv])))
            cands.append((cu, cv, math.degrees(math.atan2(-vec[1, 1], vec[0, 1])) % 180.0, len(us)))
        if not cands:
            return None
        k_best = min(range(len(cands)), key=lambda i: math.hypot(cands[i][0] - ref[0], cands[i][1] - ref[1]))
        best = cands[k_best]
        if math.hypot(best[0] - ref[0], best[1] - ref[1]) > self.MAX_JUMP_PX:
            return None
        # A long object does not fit the close view: the fingertip band (rows >= FINGER_ROW) and the image edges
        # crop it, the visible part's centroid slides toward its far end, and chasing it walked the jaws onto the
        # screwdriver's end cap. When the blob is cut off, trust the far alignment: report the jaw pixel itself.
        vs, us = np.nonzero(lab == self._labels[k_best])
        cut = vs.max() >= ppa.FINGER_ROW - 3 or vs.min() <= 2 or us.min() <= 2 or us.max() >= rgb.shape[1] - 3
        if cut and self.close_target is not None:
            print("  close view cuts the object off: keeping the far alignment")
            return (self.close_target[0], self.close_target[1], best[2], best[3])
        self.close_uv = self.uv = (best[0], best[1])
        return best


async def verify_target(rig, text: str, store) -> tuple:
    """
    (ok, why) just before a descent: is the object under the jaws the one `text` asks for?

    The object under the jaws is the one nearest the jaw-axis pixel at its own top depth. It passes if best()
    over everything in view picks it, or (if best() picks nothing) best() over it alone accepts it. Any other
    winner means the tracker followed the wrong thing: refuse rather than grab it.
    """
    rgb, depth = await rig.frame()
    seg = perception.segment_full(rgb, depth)
    objs = list(seg.objects) + [o for o, _ in seg.excluded]
    if not objs:
        return False, "nothing under the jaws"

    def off_axis(o):
        au, av = ppa.axis_pixel(rig.fx, rig.fy, rig.cx, rig.cy, o.top_depth_mm)
        return math.hypot(o.centroid_px[0] - au, o.centroid_px[1] - av)

    under = min(objs, key=off_axis)
    if off_axis(under) > 80.0:
        return False, "nothing under the jaws"
    found = clip_embed.best(objs, text, store)
    win = found.get("obj")
    if win is not None:
        same = math.hypot(win.centroid_px[0] - under.centroid_px[0], win.centroid_px[1] - under.centroid_px[1]) < 40.0
        if same:
            return True, "target under the jaws"
        return False, f"the object under the jaws is not '{text}' ('{found.get('recognised') or win.colour}' is elsewhere)"
    alone = clip_embed.best([under], text, store)
    if alone.get("obj") is not None:
        return True, "target under the jaws (only candidate)"
    return False, f"the object under the jaws does not look like '{text}'"


def hole_mask(rgb: np.ndarray):
    """Bool mask of the tape hole incl. anything standing in it (even a blue object), or None if not in view."""
    blue = ndimage.binary_closing(ppa.blue_mask(rgb), iterations=3)
    lab, n = ndimage.label(~blue)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    best = None
    for i in range(1, n + 1):
        if i in border:
            continue
        size = int((lab == i).sum())
        if size > 2000 and (best is None or size > best[1]):
            best = (i, size)
    if best is None:
        return None
    # Fill so that an object inside the hole that is itself blue still counts as inside
    return ndimage.binary_fill_holes(lab == best[0])


def objects_in_hole(rgb: np.ndarray, depth: np.ndarray) -> tuple:
    """
    (hole visible?, [Obj inside the hole], segmentation) from a frame taken above the square at z >= 400.

    Inside = >= half the mask. Must be looked at from ALIGN_Z: below ~280 mm depth is blind and a block in
    the square is invisible. Edge-cut objects are counted too (a full square can push one to the border).
    """
    seg = perception.segment_full(rgb, depth) if depth is not None else None
    hole = hole_mask(rgb)
    if hole is None or seg is None:
        return False, [], seg
    objs = list(seg.objects) + [o for o, _ in seg.excluded]
    return True, [o for o in objs if (o.mask & hole).sum() >= 0.5 * o.mask.sum()], seg


def jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def width_mm(obj, fx: float) -> float:
    """Object width across the jaws (short side) in mm, from pixels at its top-surface depth."""
    return obj.width_px * obj.top_depth_mm / fx


async def look_here(rig, text: str, store, source: str = "demo") -> dict:
    """
    One frame from the current pose -> best match for `text` among objects outside the square.

    Returns clip_embed.best()'s dict plus "rgb", "depth", "pose", and "square" (square_from_frame) when the
    hole is in view. Objects cut by the image border or standing in the square are never candidates.
    """
    rgb, depth = await rig.frame()
    pose = pose_dict(await rig.pose())
    rig.record("observe", rgb)
    seg = perception.segment_full(rgb, depth)
    publish_frame(store, rgb, seg, "observe")
    if abs(pose["theta"]) <= 1.0:
        record_sightings(store, rgb, depth, seg, pose, intr(rig), source)
    hole = hole_mask(rgb)
    objs = [o for o in seg.objects if not o.touches_edge and (hole is None or (o.mask & hole).sum() < 0.5 * o.mask.sum())]
    found = clip_embed.best(objs, text, store) if objs else {"obj": None, "reason": "nothing seen", "searchable": True}
    found.update(rgb=rgb, depth=depth, pose=pose, square=square_from_frame(rgb, depth, pose, intr(rig)))
    return found


async def find_target(rig, text: str, store) -> dict:
    """
    Look from the current pose, then OBSERVE_XY, for the object best matching `text` (outside the square).

    Stops early when best() says the thing is not something the library knows ("searchable" False): looking
    elsewhere cannot help. Returns look_here()'s dict plus "path" (current / grid); obj None if no match.
    """
    p = await rig.pose()
    poses = [None] + sorted(OBSERVE_XY, key=lambda q: math.hypot(q[0] - p.x, q[1] - p.y))
    last = {"obj": None, "reason": "nothing seen"}
    for xy in poses:
        if xy is not None:
            await ppa.travel(rig, *xy, theta=0.0)
        found = await look_here(rig, text, store)
        found["path"] = "current" if xy is None else "grid"
        if found.get("obj") is not None:
            return found
        last = found
        if found.get("searchable") is False:
            break
    return last


async def look_at(rig, table_xy, text: str, store) -> dict:
    """One look at a table x/y (e.g. from the table map): move so it is near the image centre, then look_here."""
    fx, fy = perception.look_flange_xy(tuple(table_xy))
    if not ppa.allowed(fx, fy) or math.hypot(fx, fy) < 120.0:
        return {"obj": None, "reason": f"look pose ({fx:.0f},{fy:.0f}) is out of reach"}
    await ppa.travel(rig, fx, fy, theta=0.0)
    return await look_here(rig, text, store, source="look")


def save_frame(out: Path, k: int, fr: dict) -> None:
    """Sweep frame k to out/: <k>.jpg, <k>_depth.npy (uint16 mm), <k>.json (id, pose, intrinsics, square)."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fr["rgb"]).save(out / f"{k:02d}.jpg", quality=92)
        np.save(out / f"{k:02d}_depth.npy", fr["depth"])
        meta = {key: fr[key] for key in ("id", "pose", "intrinsics", "square") if key in fr}
        (out / f"{k:02d}.json").write_text(json.dumps(meta))
    except OSError as e:
        print(f"  could not save sweep frame {k}: {e}")


async def sweep_table(rig, progress=None) -> list:
    """
    Frames over the whole reachable table for sorter.table_map.map_from_frames (AGENT3); ends at observe.

    Each frame: {"id", "rgb", "depth" (uint16 mm, aligned), "pose", "intrinsics"} plus "square" when the hole
    is fully in view. progress(k, n) is called after each pose (dashboard: "scanning table... k/n").
    """
    poses = [q for q in SWEEP_XY if ppa.allowed(*q) and 150.0 <= math.hypot(*q) <= 600.0]
    frames, n = [], len(poses)
    t0 = time.monotonic()
    out = SWEEP_DIR / time.strftime("%Y%m%d-%H%M%S")
    try:
        for k, (x, y) in enumerate(poses, 1):
            await ppa.travel(rig, x, y, theta=0.0, z=SWEEP_Z)
            rgb, depth = await rig.frame()
            pose = pose_dict(await rig.pose())
            rig.record("sweep", rgb, k=k, n=n)
            fr = {"id": f"sweep-{int(time.time() * 1000)}-{k}", "rgb": rgb,
                  "depth": np.clip(depth, 0, 65535).astype(np.uint16), "pose": pose, "intrinsics": intr(rig)}
            sq = square_from_frame(rgb, depth, pose, fr["intrinsics"])
            if sq:
                fr["square"] = sq
            frames.append(fr)
            save_frame(out, k, fr)
            STATE["sweep"] = {"k": k, "n": n}
            if progress:
                progress(k, n)
    finally:
        STATE["sweep"] = None
        await back_to_observe(rig)
    print(f"sweep: {len(frames)} frames in {time.monotonic() - t0:.1f} s -> {out}")
    return frames


async def execute(rig, text: str, store, mode: str = "demo", target=None, target_xy=None) -> dict:
    """
    One command: put the object described by `text` into the blue square. Returns the logged attempt doc.

    target: clip_embed.best() dict + "rgb", "depth", "pose", "ts" from a frame the caller already took at the
      observe pose (theta within +-1 deg): no observe move, pick at once. Ignored if its theta is off.
    target_xy: table x/y (e.g. from the table map): one look there, then pick; "not where the map said" if absent.
    Neither: look around (find_target). The square comes from the same frame when visible, else LAST_SQUARE
    (< SQUARE_CACHE_S old), else is searched while carrying. Always ends at the observe pose.
    Never raises for an ordinary failure: status is done / not_found / ambiguous / too_wide / failed.
    """
    doc = {"ts": time.time(), "mode": mode, "target_text": text, "success": False, "status": "failed",
           "fail_phase": None, "fail_reason": None}
    times = {}
    STATE.update(times=times, error=None)
    start_recording(rig, store, doc)
    t0 = time.monotonic()
    try:
        times["at"] = "find"
        found = None
        if target is not None and target.get("obj") is not None and abs(float((target.get("pose") or {}).get("theta", 99.0))) <= 1.0:
            found, doc["find_path"] = target, "frame"
            tp = target["pose"]
            p = await rig.pose()
            # The tracker seeds at the target's pixel: be where that frame was taken
            if math.hypot(p.x - tp["x"], p.y - tp["y"]) > 15.0 or abs(p.z - tp["z"]) > 15.0 or abs(p.theta) > 1.0:
                await ppa.travel(rig, tp["x"], tp["y"], theta=0.0, z=tp["z"])
            if found.get("square") is None:
                found["square"] = square_from_frame(found["rgb"], found.get("depth"), tp, intr(rig))
        elif target_xy is not None:
            found = await look_at(rig, target_xy, text, store)
            doc["find_path"] = "map"
            if found.get("obj") is None:
                doc.update(status="not_found", fail_phase="find",
                           fail_reason=f"not where the map said ({found.get('reason')})")
                return doc
        else:
            found = await find_target(rig, text, store)
            doc["find_path"] = found.get("path")
        obj = found.get("obj")
        if obj is None:
            doc.update(status="not_found", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        doc.update(label=found.get("recognised") or found.get("label") or obj.colour or "object", colour=obj.colour,
                   score=round(float(found.get("score", 0.0)), 3),
                   margin=round(float(found.get("margin", 0.0)), 3), img_before=jpeg(found["rgb"]))
        if found.get("ambiguous"):
            doc.update(status="ambiguous", fail_phase="find", fail_reason=found.get("reason"))
            return doc
        if width_mm(obj, rig.fx) > MAX_OPEN_MM:
            doc.update(status="too_wide", fail_phase="find", fail_reason=f"{width_mm(obj, rig.fx):.0f} mm across")
            return doc
        # Where to put it: same frame, else the last known square, else search while carrying (offset servo)
        square_xy, doc["square_path"] = None, "search"
        if found.get("square"):
            square_xy, doc["square_path"] = tuple(found["square"]["xy"]), "frame"
        elif LAST_SQUARE["xy"] is not None and time.time() - LAST_SQUARE["ts"] < SQUARE_CACHE_S:
            square_xy, doc["square_path"] = LAST_SQUARE["xy"], "cache"
        tracker = ObjTracker(rig, obj, found["rgb"], text=text, store=store)
        height, grip = await ppa.pick(rig, times, tracker, verify=lambda: verify_target(rig, text, store))
        doc["height_mm"], doc["grasp"] = round(height, 1), {"grip_depth": round(grip, 1), "theta": rig.theta}
        await ppa.place_in_square(rig, height, grip, times, square_xy=square_xy)
        # Released over the hole's centre (place_in_square servos onto it when visible): remember it
        q = await rig.pose()
        LAST_SQUARE.update(xy=(q.x, q.y), ts=time.time())
        times["at"] = "check"
        rgb, depth = await rig.frame()
        rig.record("check", rgb)
        seen, inside, seg = objects_in_hole(rgb, depth)
        if seg is not None:
            publish_frame(store, rgb, seg, "place")
        doc["img_after"] = jpeg(rgb)
        # Only a success if what is in the hole is what was asked for
        right = bool(seen and inside) and clip_embed.best(inside, text, store).get("obj") is not None
        doc["success"] = right
        doc["status"] = "done" if right else "failed"
        if not right:
            why = "hole not visible" if not seen else "nothing seen inside the hole" if not inside else "wrong object in the hole"
            doc.update(fail_phase="check", fail_reason=why)
    except Exception as e:
        doc.update(status="failed", fail_phase=times.get("at", "start"), fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    finally:
        times.pop("at", None)
        doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
        finish(rig, store, doc)
        STATE.update(times=None, error=doc["fail_reason"] if doc["status"] == "failed" else None)
        await back_to_observe(rig)
    return doc


async def recover(rig, doc: dict) -> None:
    """After a failure: lift; if still holding something, put it down right there so nothing stays mid-air."""
    try:
        await rig.lift()
        if await rig.holding():
            await ppa.release_here(rig, doc.get("height_mm", 30.0), (doc.get("grasp") or {}).get("grip_depth", 15.0))
    except Exception as e:
        doc["fail_reason"] = f"{doc.get('fail_reason')} | recovery failed: {e}"


FREE_SPOTS_FILE = Path(__file__).resolve().parent.parent / "data" / "scan" / "free_spots.json"


def candidate_spots(n: int = 8) -> list:
    """
    Drop candidates: free spots measured by the table survey (scripts/survey_scan.py), nearest-agnostic
    shuffle; random points in RESET_X x RESET_Y only if no survey exists. Each is re-checked live anyway.
    """
    import json

    try:
        spots = [tuple(s) for s in json.loads(FREE_SPOTS_FILE.read_text())]
    except (OSError, ValueError):
        spots = []
    random.shuffle(spots)
    if spots:
        return spots[: max(n, 12)]
    return [(random.uniform(*RESET_X), random.uniform(*RESET_Y)) for _ in range(n)]


async def drop_somewhere(rig, height: float, grip: float) -> tuple:
    """Put the held object at a free spot (not near the square, no object within ~90 mm); returns (x, y)."""
    for x, y in candidate_spots():
        if not 150.0 <= math.hypot(x, y) <= 600.0 or not ppa.allowed(x, y):
            continue
        print(f"reset: trying ({x:.0f},{y:.0f})")
        if await ppa.place_at(rig, x, y, height, grip):
            return x, y
    raise RuntimeError("no free spot to drop the object; still holding it")


async def wipe(rig) -> None:
    """
    Push whatever is in the hole out sideways with the closed gripper (fallback when picking fails).

    One slow straight pass along world y (sideways to the arm: never toward the base or away into the room),
    fingertips WIPE_CLEAR_MM above the measured table, toward the side whose depth shows fewer objects.
    Starts with the arm centred over the hole (after the square servo). Not yet run on hardware.
    """
    p = await rig.pose()
    z1, table, _ = await rig.measure()
    rgb, depth = await rig.frame()
    near = (depth > 0) & (depth < table - 8)
    w = near.shape[1]
    # Image +u is world -y (measured on sfhack4), so the image's right half is the -y side
    left_busy, right_busy = int(near[:, : w // 3].sum()), int(near[:, 2 * w // 3:].sum())
    direction = 1.0 if left_busy <= right_busy else -1.0  # +y = toward the image's left
    z_wipe = max(z1 - (table - ppa.TIP_BELOW_CAM - WIPE_CLEAR_MM), rig.floor_z)
    start, end = p.y - direction * 90.0, p.y + direction * 180.0
    print(f"  wipe: y {start:.0f} -> {end:.0f} at z={z_wipe:.0f}")
    await ppa.travel(rig, p.x, start)
    await rig.grab()  # close the jaws into a flat pusher; returns False with nothing inside, as expected
    await rig.wait_clear()
    await rig.descend(z_wipe)
    y = start
    while (end - y) * direction > 0:
        y = y + direction * min(WIPE_STEP_MM, abs(end - y))
        await rig.goto(p.x, y, z_wipe)
    await rig.lift()
    await rig.open()


async def clear_square(rig, store) -> dict:
    """
    Empty the blue square. Returns the logged attempt doc (mode "reset"); success = hole empty by depth.

    Raises nothing for ordinary failures; status "failed" means it parked and the next trial must not start.
    """
    doc = {"ts": time.time(), "mode": "reset", "target_text": "__reset__", "success": False, "status": "failed",
           "removed": 0, "wipes": 0, "fail_phase": None, "fail_reason": None}
    t0 = time.monotonic()
    STATE.update(times={"at": "reset"}, error=None)
    start_recording(rig, store, doc)
    try:
        await ppa.search_square(rig)
        await rig.servo(ppa.find_square, False, "square")
        for _round in range(6):
            rgb, depth = await rig.frame()
            seen, inside, seg = objects_in_hole(rgb, depth)
            if seg is not None:
                publish_frame(store, rgb, seg, "reset")
            if not seen:
                raise RuntimeError("square not visible after servo")
            if not inside:
                q = await rig.pose()
                # Arm is centred over the empty square with an empty gripper: the place target for the next run
                doc.update(success=True, status="done", square_xy=[q.x, q.y])
                LAST_SQUARE.update(xy=(q.x, q.y), ts=time.time())
                break
            obj = inside[0]
            print(f"reset: {len(inside)} object(s) in the square; removing one ({obj.colour}, {obj.height_mm:.0f} mm)")
            picked = False
            if width_mm(obj, rig.fx) <= MAX_OPEN_MM:
                for _try in range(2):
                    try:
                        height, grip = await ppa.pick(rig, None, ObjTracker(rig, obj, rgb))
                        doc.setdefault("dropped_at", []).append(list(await drop_somewhere(rig, height, grip)))
                        picked = True
                        doc["removed"] += 1
                        break
                    except ppa.PickFailed as e:
                        print(f"reset: pick failed ({e})")
                    await rig.lift()
            if not picked:
                if doc["wipes"] >= 2:
                    raise RuntimeError("square still not empty after 2 wipes")
                await ppa.search_square(rig)
                await rig.servo(ppa.find_square, False, "square")
                await wipe(rig)
                doc["wipes"] += 1
            # Back over the square to look again
            await ppa.search_square(rig)
            await rig.servo(ppa.find_square, False, "square")
        if not doc["success"]:
            doc.update(fail_phase="verify", fail_reason="objects still in the square")
        doc["img_after"] = jpeg((await rig.frame())[0])
    except Exception as e:
        doc.update(fail_phase=doc.get("fail_phase") or "reset", fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    doc["duration_s"] = round(time.monotonic() - t0, 1)
    finish(rig, store, doc)
    STATE.update(times=None, error=None if doc["success"] else doc["fail_reason"])
    await back_to_observe(rig)
    return doc


async def resume(rig, store):
    """
    After a restart: if the gripper still holds something, put it in the square first. Returns the doc or None.

    WHY: the SDK can abort the process mid-carry; the supervisor restarts it with the object still held.
    Height/grip come from data/arm_state.json (pick_place_auto.save_state); defaults fit the yellow block.
    """
    st = ppa.load_state()
    if not await rig.holding():
        if st.get("holding"):
            ppa.save_state(False)
        return None
    print("resume: gripper is holding something; placing it in the square first")
    height, grip = st.get("height") or 32.0, st.get("grip") or 15.0
    doc = {"ts": time.time(), "mode": "resume", "target_text": "held after restart", "success": False,
           "status": "failed", "fail_phase": None, "fail_reason": None, "height_mm": height}
    times = {}
    STATE.update(times=times, error=None)
    start_recording(rig, store, doc)
    t0 = time.monotonic()
    try:
        await ppa.place_in_square(rig, height, grip, times)
        times["at"] = "check"
        rgb, depth = await rig.frame()
        seen, inside, _ = objects_in_hole(rgb, depth)
        doc["success"] = bool(seen and inside)
        doc["status"] = "done" if doc["success"] else "failed"
    except Exception as e:
        doc.update(fail_phase=times.get("at", "start"), fail_reason=f"{type(e).__name__}: {e}")
        await recover(rig, doc)
    times.pop("at", None)
    doc["phase_times"], doc["duration_s"] = times, round(time.monotonic() - t0, 1)
    finish(rig, store, doc)
    STATE.update(times=None, error=doc["fail_reason"])
    return doc
