"""
CLIP embeddings and text ranking: which object on the table did the user mean?

WHY: the demo command is a free description ("green square", "the cup"); CLIP puts text and image in one
512-d space, so the description can be compared with every object crop directly, with no training.
CLIP is weak on colour + shape words, so the object's dominant colour is added on top. With a Store,
each object is also *recognised* from its nearest labelled views (the arm scan), and the query is
matched against that label and its aliases. That is what separates look-alikes such as the green cube
("green square") and the green cylinder.

Depth-based shape cues were tried and dropped: on sfhack4 frames the top faces of standing cylinders
(0.82-0.87 hull rectangularity) and of blocks (0.83-0.95) overlap, so a shape rule both refused cubes
and picked cylinders.

Contract: docs/EXEC_PLAN.md (embed_images, embed_text, rank). Model: open_clip ViT-B-32, CPU.
"""
import re
import time

import numpy as np

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
COLOURS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "white", "grey", "black"}
ALIASES = {"gray": "grey", "violet": "purple", "pink": "red"}
# What a person calls a colour vs the hue bin perception picks: a light-blue cylinder was binned "cyan" on
# frame e01, and best("blue cylinder") refused it. A query colour accepts these object colours.
COMPATIBLE = {"blue": {"blue", "cyan", "purple"}, "cyan": {"cyan", "blue", "green"}, "green": {"green", "cyan"},
              "yellow": {"yellow", "orange"}, "orange": {"orange", "yellow", "red"}, "red": {"red", "orange", "purple"},
              "purple": {"purple", "blue", "red"}, "white": {"white", "grey"}, "grey": {"grey", "white", "black"},
              "black": {"black", "grey"}}
# Command verbs/fillers stripped before embedding; the destination is always the blue square
STOP = r"^(please\s+)?(grasp|grab|pick( up)?|get|take|move|put|place)\s+(up\s+)?(the\s+|a\s+|an\s+)?"
FILLER = {"the", "a", "an", "one", "object", "thing"}
COLOUR_BONUS = 0.04  # colour word matches the object's dominant colour (CLIP sims sit around 0.2-0.3)
COLOUR_PENALTY = 0.04  # colour word names a different, saturated colour
SHAPE_BONUS = 0.02  # "long"/"stick" vs footprint aspect; the only shape cue that held up
# Library recognition. View scores are on the Store's 0-1 scale ((1 + cos) / 2). Measured on the 32-frame arm
# scan: same object across frames median 0.93 (p5 0.86); different objects up to 0.95, but only same-shape
# objects of another colour (blue vs green cylinder) or another height (yellow cylinder 60 mm vs yellow block
# 31 mm). So a view must also agree on colour and height to vote.
# [CONFIDENCE: CONFIRMED 75% — 10 clean groups, 100 sightings; see label_scan held-out test]
KNN_MIN = 0.88  # a view at least this close counts as "same object"
VIEW_DH_MM = 15.0  # ... and its height within this of the object's
# Objects that rest in two poses: a cylinder is ~30 mm tall lying (its diameter) and 59-61 mm standing (its length).
# A view of these labels also admits an object at the other pose's height: the green cylinder slid off during a
# carry, lay at 30 mm and every standing view (60-62 mm) was filtered out, leaving only the green cube's (29-33 mm).
# {VIAM CENTERAL 2026-10-29 "CYLINDER 类同时接受直径高度和长度高度（29 和 61 MM 都算）"; LYING NPZ "GREEN H= 30.4 … 61X28MM";
#  STANDING FOOTPRINT 076 "BLUE H=62 FOOTPRINT 31X28 MM"} [CONFIDENCE: CONFIRMED 80% — eval_perception recog/demo]
POSE_HEIGHTS_MM = {"green cylinder": (30.0, 61.0), "yellow cylinder": (30.0, 59.0), "blue cylinder": (30.0, 61.0)}
# Vote weight by view source: phone close-ups are oblique, the wrist camera looks straight down
SOURCE_WEIGHT = {"phone": 0.5}
LABEL_BONUS = 0.15  # object recognised as something the query names
LABEL_PENALTY = 0.10  # object recognised as something else
# Query is exactly one of the object's names, not just contained in one: "utility knife" is the silver knife's
# label but is also contained in the red one's "red utility knife" (same for "box cutter"); without this both get
# LABEL_BONUS and the pick is a coin flip between two look-alike knives. Bare "knife" names neither exactly.
# {USER VIA VIAM CENTERAL "I ADDED THE KNIVES, INCLUDE THOSE TOO"; ALIASES SILVER "UTILITY KNIFE", RED "RED UTILITY KNIFE"}
# [CONFIDENCE: INFERRED 80% — 0.05 > AMBIGUOUS_MARGIN and above the CLIP gap between the two knives]
EXACT_NAME_BONUS = 0.05
# Below this the best crop is not really what was asked for. On frame t01 the correct objects scored
# 0.27-0.35 while "green square" (no green object present) topped out at 0.203 on a laptop key.
# [CONFIDENCE: SINGLE-SRC 60% — one frame, 6 objects; re-tune on the scanned set]
MIN_SCORE = 0.22
AMBIGUOUS_MARGIN = 0.02  # top-1 minus top-2 below this: let the user confirm in the dashboard
# Open vocabulary: a query naming nothing in the library. The library may then only say what an object clearly is:
# recognised as X, it is vetoed only if its crop is more like "a photo of a X" than like the query by OPEN_SET_DELTA
# (a similar known object is often the nearest view, so "recognised as X" alone refused the blue block as "blue
# cylinder"). Absolute CLIP scores do not separate right from wrong (right: median 0.372, wrong: 0.421), so the query
# must also beat every generic prompt by OPEN_NEG_MARGIN. Tuned with each label held out of the library (65 cases):
# true targets have s_X - s_query <= -0.015, other objects mostly >= 0.069; delta 0-0.02 gives the same result.
# {USER VIA VIAM CENTERAL 2026-10-30 "EXPAND THIS TO OPEN VOCAB, SO THAT ANY OBJECT CAN BE GRASPED"}
# {GRID "DELTA 0.01 NM 0.03: RIGHT 58 WRONG 1 REFUSED 6; ABSENT TUNE 0/0; HELD-OUT QUERIES 1/1"}
# [CONFIDENCE: INFERRED 75% — held-out labels of the same 14 objects; no truly new object measured yet]
OPEN_SET_DELTA = 0.01
OPEN_NEG_MARGIN = 0.03
OPEN_NEGATIVES = ("table", "wooden table", "piece of paper", "object", "thing on a table", "cable", "hand")

_model = _preprocess = _tokenizer = None


COLOUR_PRESENT_FRAC = 0.10  # blue screwdriver 0.33-0.38 blue, red one 0.31 red; stray hue on black objects <= 0.06


def colour_ok(query_colours: set, obj_colour: str) -> bool:
    """True if the object's colour is one a person could call by any of the query's colour words."""
    return any(obj_colour in COMPATIBLE.get(c, {c}) for c in query_colours)


def colour_present(query_colours: set, obj) -> bool:
    """
    The query's colour is the object's dominant colour, or makes up >= COLOUR_PRESENT_FRAC of it.

    WHY: both screwdrivers are dominantly "black" (handles); only the secondary colour tells "blue screwdriver"
    from "red screwdriver". Objects without colour shares (older records) fall back to the dominant colour.
    """
    shares = getattr(obj, "colours", None) or {}
    return colour_ok(query_colours, obj.colour) or any(
        f >= COLOUR_PRESENT_FRAC and colour_ok(query_colours, c) for c, f in shares.items())


def _load():
    """Load the model once per process (first call downloads the weights, ~600 MB)."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        import open_clip
        import torch

        torch.set_num_threads(max(1, torch.get_num_threads()))
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(MODEL)
    return _model, _preprocess, _tokenizer


def embed_images(crops: list) -> np.ndarray:
    """(N, 512) float32, L2-normalised, for RGB uint8 crops."""
    import torch
    from PIL import Image

    model, prep, _ = _load()
    if not crops:
        return np.zeros((0, 512), np.float32)
    batch = torch.stack([prep(Image.fromarray(c)) for c in crops])
    with torch.no_grad():
        f = model.encode_image(batch).float()
    f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy().astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """(512,) float32, L2-normalised, for "a photo of a {text}"."""
    import torch

    model, _, tok = _load()
    with torch.no_grad():
        f = model.encode_text(tok([f"a photo of a {text}"])).float()
    f = f / f.norm(dim=-1, keepdim=True)
    return f[0].numpy().astype(np.float32)


def clean_query(text: str) -> str:
    """Lowercase, strip command verbs ("grasp", "pick up the") and normalise colour aliases."""
    q = re.sub(STOP, "", text.strip().lower())
    return " ".join(ALIASES.get(w, w) for w in q.split())


_NAME_CACHE = {}


def _name_emb(text: str) -> np.ndarray:
    """embed_text(clean_query(text)) memoised: the same few queries repeat on every command (~15 ms each)."""
    q = clean_query(text)
    if q not in _NAME_CACHE:
        _NAME_CACHE[q] = embed_text(q)
    return _NAME_CACHE[q]


def _words(text: str) -> set:
    return set(clean_query(text).split()) - FILLER


def name_matches(query: str, names: list) -> bool:
    """
    True if the query names this object: its words are contained in a label/alias or vice versa.

    Word containment, not text-embedding similarity: "green cube" and "green cylinder" embed close
    together, and that closeness is exactly the confusion the library has to remove.
    """
    q = _words(query)
    return bool(q) and any(q <= _words(n) or (_words(n) and _words(n) <= q) for n in names)


def recognise(emb: np.ndarray, store, k: int = 8, colour: str = None, height_mm: float = None) -> tuple:
    """
    (label, names, confidence) from the nearest labelled views, or (None, [], 0.0).

    Views closer than KNN_MIN vote with their score (phone views at reduced weight), but only if their stored
    colour and height (when present) agree with the object's: CLIP alone confuses same-shape objects of a
    different colour or height. A label with two resting poses (POSE_HEIGHTS_MM) accepts either pose's height.
    names = the winning label plus every alias seen.
    """
    votes, names = {}, {}
    for v in store.search_views(emb, k=k):
        if v["score"] < KNN_MIN:
            continue
        if colour and v.get("colour") and not (colour_ok({colour}, v["colour"]) or colour_ok({v["colour"]}, colour)):
            continue
        if height_mm is not None and v.get("height_mm") is not None and not any(
                abs(h - height_mm) <= VIEW_DH_MM for h in (v["height_mm"], *POSE_HEIGHTS_MM.get(v["label"], ()))):
            continue
        votes[v["label"]] = votes.get(v["label"], 0.0) + v["score"] * SOURCE_WEIGHT.get(v.get("source"), 1.0)
        names.setdefault(v["label"], {v["label"]}).update(v.get("aliases") or [])
    if not votes:
        return None, [], 0.0
    lab = max(votes, key=votes.get)
    return lab, sorted(names[lab]), votes[lab] / k


def recognise_all(rgb: np.ndarray, depth: np.ndarray, store=None, seg=None, include_partial: bool = False) -> list:
    """
    Every target in one frame with its library label: [{"obj", "label", "names", "score", "emb", "partial"}, ...].

    WHY: the module's idle loop recognises the table about once a second, and a grasp command should not
    embed the crops again. One CLIP batch covers all crops; pass the result to best(..., recognised=...).
    `seg`: a perception.segment_full() result the caller already has (e.g. to check intruders first).
    `include_partial`: also recognise objects cut by the image edge (partial=True; never grasp candidates, but the
    dynamic map uses them: a moved KIND bar was only ever seen cut at the edge and otherwise fell off the map).
    label/names are None/[] when the object is not recognised (or there is no store); score = vote confidence.
    """
    from sorter import perception

    seg = seg if seg is not None else perception.segment_full(rgb, depth)
    objs = [(o, False) for o in seg.objects] + ([(o, True) for o, _ in seg.excluded] if include_partial else [])
    embs = embed_images([o.crop for o, _ in objs])
    out = []
    for (o, partial), e in zip(objs, embs):
        lab, names, conf = (recognise(e, store, colour=o.colour, height_mm=o.height_mm) if store is not None
                            else (None, [], 0.0))
        o.recognised, o.recognised_names = lab, names
        out.append({"obj": o, "label": lab, "names": names, "score": conf, "emb": e, "partial": partial})
    return out


SIGHTING_MIN_CONF = 0.2  # recognition confidence (vote/k) below this does not move an object's last-seen xy


def record_sightings(store, recs: list, pose: dict, intr: dict = None, source: str = "idle") -> int:
    """
    Update the last-seen table xy of every confidently recognised object in a recognise_all() result.

    WHY: where_is() must follow objects that people or trials move, not only the scan. Called by the idle loop,
    the executor's re-looks and trials, with the arm pose the frame was taken at (arm still, frame not stale).
    Needs Store.log_sighting (AGENT 2); returns the number written, 0 if the Store has no sightings yet.
    """
    from sorter import perception

    if not hasattr(store, "log_sighting"):
        return 0
    n = 0
    for r in recs:
        if r["label"] is None or r["score"] < SIGHTING_MIN_CONF:
            continue
        o = r["obj"]
        xy = perception.world_xy(pose, intr or perception.INTRINSICS, *(o.top_centroid_px or o.centroid_px), o.top_depth_mm)
        store.log_sighting({"label": r["label"], "xy": [round(xy[0], 1), round(xy[1], 1)],
                            "height_mm": round(o.height_mm, 1), "conf": round(r["score"], 3), "source": source,
                            "pose": {k: pose[k] for k in ("x", "y", "z", "theta") if k in pose}, "ts": time.time()})
        n += 1
    return n


def where_is(store, text: str) -> dict:
    """
    Where the object a query names was last seen: {"label", "xy", "source", "look_flange_xy", ...} or None.

    WHY: the observe pose sees only part of the table (frame 076: the green cube, yellow block and KIND bar were
    all outside or cut by the edge, so best() correctly refused). Newest live sighting (record_sightings) first;
    otherwise the arm-scan view from the highest survey frame (frames are numbered in time order).
    look_flange_xy centres that spot in the camera (theta 0, z 400); look there, recognise_all, best() again.
    """
    from sorter import perception

    entry = next((n for n in library_names(store) if name_matches(text, n)), None)
    if entry is None:
        return None
    label = entry[0]
    s = store.last_sighting(label) if hasattr(store, "last_sighting") else None
    if s:
        return {"label": label, "xy": tuple(s["xy"]), "source": s.get("source", "sighting"), "ts": s.get("ts"),
                "look_flange_xy": perception.look_flange_xy(s["xy"])}
    views = [v for v in store.search_views(_name_emb(label), k=1000, label=label) if v.get("xy") and v.get("frame")]
    if not views:
        return None
    v = max(views, key=lambda v: v["frame"])
    return {"label": label, "xy": tuple(v["xy"]), "source": "scan", "frame": v["frame"],
            "look_flange_xy": perception.look_flange_xy(v["xy"])}


def library_names(store) -> list:
    """[[label, *aliases], ...] for every labelled object, so best() knows which queries the library covers."""
    return store.list_labels()


def _known(text: str, store) -> bool:
    """Does the query name something in the library (a label or alias, by word containment)?"""
    return store is not None and any(name_matches(text, n) for n in library_names(store))


def _names_exactly(text: str, store) -> bool:
    """The query is a whole label or alias, or contains one ("green cube", "the green cube please"), not just part of one."""
    q = _words(text)
    return any(_words(name) and _words(name) <= q for names in library_names(store) for name in names)


def _searchable(res: dict) -> dict:
    """A refusal of a query the library partly knows: its own object of that kind may be in another view."""
    return {**res, "searchable": res["obj"] is None}


def rank(objs: list, text: str, store=None, recognised: list = None, known: bool = None) -> list:
    """
    [(Obj, score)] best first for a free-text description.

    score = CLIP(crop, "a photo of a {query}") + colour term + "long" term + library term.
    Library term (store given, query known to the library): +LABEL_BONUS if the object is recognised as something
    the query names, -LABEL_PENALTY if it is confidently recognised as something else. An unknown query has no library
    term: the library has no opinion on an object it never saw (best() applies the open-set gate instead).
    Sets obj.recognised / .recognised_names / .open_gate ({"x_gap", "neg_gap"}, unknown queries only).
    `recognised`: recognise_all() output for these objects; their embeddings and labels are reused.
    known: _known(text, store) if the caller already has it.
    """
    if not objs:
        return []
    known = _known(text, store) if known is None else known
    q = clean_query(text)
    words = set(q.split())
    pre = {id(r["obj"]): r for r in recognised or []}
    missing = [o for o in objs if id(o) not in pre]
    emb_of = {id(o): e for o, e in zip(missing, embed_images([o.crop for o in missing]))} if missing else {}
    emb_of.update({k: r["emb"] for k, r in pre.items()})
    img = np.stack([emb_of[id(o)] for o in objs])
    sims = img @ _name_emb(q)
    qcol = words & COLOURS
    out = []
    for o, emb, s in zip(objs, img, sims):
        score = float(s)
        if qcol:
            if colour_present(qcol, o):
                score += COLOUR_BONUS
            elif o.colour in COLOURS - {"grey", "white", "black"}:
                score -= COLOUR_PENALTY
        if words & {"long", "stick", "pen", "bar"}:
            score += SHAPE_BONUS if o.length_px / max(o.width_px, 1.0) > 2.0 else -SHAPE_BONUS
        o.recognised, o.recognised_names = None, []
        if store is not None:
            if id(o) in pre:
                lab, names = pre[id(o)]["label"], pre[id(o)]["names"]
            else:
                lab, names, _ = recognise(emb, store, colour=o.colour, height_mm=o.height_mm)
            o.recognised, o.recognised_names = lab, names
            if known and lab is not None:
                score += LABEL_BONUS if name_matches(q, names) else -LABEL_PENALTY
                # Exact name beats a name that merely contains the query (see EXACT_NAME_BONUS)
                if _words(q) and any(_words(q) == _words(n) for n in names):
                    score += EXACT_NAME_BONUS
            elif not known:
                o.open_gate = {"x_gap": float(emb @ _name_emb(lab)) - float(s) if lab else None,
                               "neg_gap": float(s) - max(float(emb @ _name_emb(n)) for n in OPEN_NEGATIVES)}
        out.append((o, score))
    return sorted(out, key=lambda t: -t[1])


def best(objs: list, text: str, store=None, recognised: list = None, rgb: np.ndarray = None) -> dict:
    """
    The object to pick for `text`, or a reason not to pick anything.

    WHY: rank() always returns something, so "green square" with no green object on the table would grab a
    laptop key, and with only the green cylinder in view it would grab the cylinder. Refuses when the colour
    is wrong, when the library recognises the best candidate as a different object, or when the match is weak.
    `recognised`: recognise_all() output; then objs may be None and no crop is embedded again.
    `rgb`: the frame the objects were found in. With it, and SORTER_LLM enabled, a vision-language model is
    asked which numbered object was meant (sorter/llm_choose.py): only when this decision refuses or is
    ambiguous (fallback), or every time (always). Its answer replaces the CLIP decision; "decided_by" says
    which one decided ("clip" or "llm"), "llm" holds the model's verdict. Without rgb nothing changes.
    Returns {"obj", "score", "margin", "ambiguous", "reason", "recognised", "searchable"}; obj is None when
    refusing. searchable = refused, but the library knows what was named: it is just not in this view, so the
    caller should look where the library last saw it (where_is). Use the flag, not the reason text: "no green
    object found" and "best candidate is recognised as 'yellow cylinder'" are both searchable refusals.
    """
    if objs is None:
        # Partly visible (edge-cut) objects are never grasp candidates: their geometry is incomplete
        recognised = [r for r in recognised or [] if not r.get("partial")]
        objs = [r["obj"] for r in recognised]
    known = _known(text, store)
    r = rank(objs, text, store, recognised, known=known)
    # The query only overlaps a longer name ("utility knife" in "red utility knife", "cylinder") and nothing in view is
    # recognised as such a name: it may be another object of that kind the library never saw, so the open-set gate decides
    # instead of a flat "not recognised". A refusal stays searchable: the library's own one may be elsewhere.
    # {OPENVOCAB "UTILITY KNIFE@081: REFUSED (NO 'UTILITY KNIFE' RECOGNISED ON THE TABLE)", SILVER KNIFE HELD OUT}
    # [CONFIDENCE: CONFIRMED 80% — openvocab eval; exact names (every demo query) never take this path]
    overlap_only = known and r and not _names_exactly(text, store) and not any(
        o.recognised is not None and name_matches(text, o.recognised_names) for o, _ in r)
    if overlap_only:
        r = rank(objs, text, store, recognised, known=False)
    if not r:
        return {"obj": None, "score": 0.0, "margin": 0.0, "ambiguous": False, "reason": "no objects on the table",
                "recognised": None, "searchable": known}
    (o, s), margin = r[0], (r[0][1] - r[1][1]) if len(r) > 1 else 1.0
    out = {"obj": None, "score": s, "margin": margin, "ambiguous": False, "reason": "", "recognised": o.recognised,
           "searchable": False}
    qcol = set(clean_query(text).split()) & COLOURS
    if store is not None and (not known or overlap_only):
        res = _best_open(r, text, qcol, out)
        return _searchable(res) if known else res
    # The library knows this query ("candy" is an alias of the KIND bar): then only an object recognised as it
    # may be picked. Without this, "candy" with the bar out of view picked an unrecognised cup fragment.
    named = o.recognised is not None and name_matches(text, o.recognised_names)
    # Colour by presence, not dominance: the blue screwdriver is dominantly black but 0.38 blue, and the red one
    # (0.31 red, no blue) once got recognised as "blue screwdriver" on the table map; presence rejects that one
    if qcol and not colour_present(qcol, o):
        out["reason"] = f"no {'/'.join(sorted(qcol))} object found"
    elif o.recognised is not None and not named:
        out["reason"] = f"best candidate is recognised as {o.recognised!r}"
    elif known and o.recognised is None:
        out["reason"] = f"no {clean_query(text)!r} recognised on the table"
    elif s < MIN_SCORE:
        out["reason"] = f"best match only {s:.2f}"
    else:
        out.update(obj=o, ambiguous=margin < AMBIGUOUS_MARGIN)
    out["searchable"] = out["obj"] is None and known
    out["decided_by"] = "clip"
    return _llm_override(out, objs, text, rgb)


def _llm_override(out: dict, objs: list, text: str, rgb) -> dict:
    """Let the LLM decide when enabled and CLIP refused / was unsure (or always). Keeps `out` on any failure."""
    from sorter import llm_choose

    m = llm_choose.mode()
    if rgb is None or not objs or m == "off" or (m == "fallback" and out["obj"] is not None and not out["ambiguous"]):
        return out
    v = llm_choose.choose(rgb, objs, clean_query(text) or text)
    if v is None:  # disabled, no key, offline, timeout or unusable answer: CLIP's decision stands
        return out
    out["llm"] = {"status": v["status"], "reason": v["reason"], "ms": v["ms"], "clip_reason": out["reason"]}
    out["decided_by"] = "llm"
    if v["status"] == "ok":
        o = objs[v["index"]]
        out.update(obj=o, ambiguous=False, reason="", recognised=getattr(o, "recognised", None), searchable=False)
    elif v["status"] == "ambiguous":
        out.update(obj=None, ambiguous=True, reason=v["reason"] or "several objects match", searchable=False)
    else:  # none: refuse, but stay searchable if the library knows the name (it may be out of view)
        out.update(obj=None, ambiguous=False, reason=v["reason"] or "nothing matches")
    return out


def _best_open(ranked: list, text: str, qcol: set, out: dict) -> dict:
    """
    best() for a query the library does not know: any object may be it, unless it is clearly something else.

    Candidates: has the query's colour, not vetoed by the open-set test (OPEN_SET_DELTA), beats the generic prompts
    (OPEN_NEG_MARGIN), at least MIN_SCORE. The margin to the next candidate (not to a vetoed object) sets "ambiguous".
    A refusal is not searchable: nothing tells us the object exists at all.
    """
    cands = []
    for o, s in ranked:
        g = o.open_gate
        vetoed = g["x_gap"] is not None and g["x_gap"] > OPEN_SET_DELTA
        if (not qcol or colour_present(qcol, o)) and not vetoed and g["neg_gap"] >= OPEN_NEG_MARGIN and s >= MIN_SCORE:
            cands.append((o, s))
    if cands:
        margin = cands[0][1] - cands[1][1] if len(cands) > 1 else 1.0
        return {**out, "obj": cands[0][0], "score": cands[0][1], "margin": margin, "ambiguous": margin < AMBIGUOUS_MARGIN,
                "recognised": cands[0][0].recognised}
    # Why not: the reason of the highest-ranked object of the query's colour (or of the highest overall)
    o, s = next(((o, s) for o, s in ranked if qcol and colour_present(qcol, o)), ranked[0])
    if qcol and not colour_present(qcol, o):
        reason = f"no {'/'.join(sorted(qcol))} object found"
    elif o.open_gate["x_gap"] is not None and o.open_gate["x_gap"] > OPEN_SET_DELTA:
        reason = f"best candidate is recognised as {o.recognised!r}"
    elif o.open_gate["neg_gap"] < OPEN_NEG_MARGIN:
        reason = f"nothing here looks like {clean_query(text)!r}"
    else:
        reason = f"best match only {s:.2f}"
    return {**out, "reason": reason, "searchable": False}
