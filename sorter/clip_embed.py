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
# Library recognition. View scores are on the Store's 0-1 scale ((1 + cos) / 2): a crop matched its own
# view at 1.0 and a different object's view at 0.83-0.87 in the offline test.
# [CONFIDENCE: SINGLE-SRC 55% — offline test only; calibrate on the arm scan]
KNN_MIN = 0.90  # a view at least this close counts as "same object"
LABEL_BONUS = 0.15  # object recognised as something the query names
LABEL_PENALTY = 0.10  # object recognised as something else
# Below this the best crop is not really what was asked for. On frame t01 the correct objects scored
# 0.27-0.35 while "green square" (no green object present) topped out at 0.203 on a laptop key.
# [CONFIDENCE: SINGLE-SRC 60% — one frame, 6 objects; re-tune on the scanned set]
MIN_SCORE = 0.22
AMBIGUOUS_MARGIN = 0.02  # top-1 minus top-2 below this: let the user confirm in the dashboard

_model = _preprocess = _tokenizer = None


def colour_ok(query_colours: set, obj_colour: str) -> bool:
    """True if the object's colour is one a person could call by any of the query's colour words."""
    return any(obj_colour in COMPATIBLE.get(c, {c}) for c in query_colours)


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


def recognise(emb: np.ndarray, store, k: int = 5) -> tuple:
    """
    (label, names, confidence) from the nearest labelled views, or (None, [], 0.0).

    Views closer than KNN_MIN vote with their score; names = the winning label plus every alias seen.
    """
    votes, names = {}, {}
    for v in store.search_views(emb, k=k):
        if v["score"] < KNN_MIN:
            continue
        votes[v["label"]] = votes.get(v["label"], 0.0) + v["score"]
        names.setdefault(v["label"], {v["label"]}).update(v.get("aliases") or [])
    if not votes:
        return None, [], 0.0
    lab = max(votes, key=votes.get)
    return lab, sorted(names[lab]), votes[lab] / k


def rank(objs: list, text: str, store=None) -> list:
    """
    [(Obj, score)] best first for a free-text description.

    score = CLIP(crop, "a photo of a {query}") + colour term + "long" term + library term.
    Library term (store given): +LABEL_BONUS if the object is recognised as something the query names,
    -LABEL_PENALTY if it is confidently recognised as something else. Sets obj.recognised / .recognised_names.
    """
    if not objs:
        return []
    q = clean_query(text)
    words = set(q.split())
    img = embed_images([o.crop for o in objs])
    sims = img @ embed_text(q)
    qcol = words & COLOURS
    out = []
    for o, emb, s in zip(objs, img, sims):
        score = float(s)
        if qcol:
            if colour_ok(qcol, o.colour):
                score += COLOUR_BONUS
            elif o.colour in COLOURS - {"grey", "white", "black"}:
                score -= COLOUR_PENALTY
        if words & {"long", "stick", "pen", "bar"}:
            score += SHAPE_BONUS if o.length_px / max(o.width_px, 1.0) > 2.0 else -SHAPE_BONUS
        o.recognised, o.recognised_names = None, []
        if store is not None:
            lab, names, _ = recognise(emb, store)
            o.recognised, o.recognised_names = lab, names
            if lab is not None:
                score += LABEL_BONUS if name_matches(q, names) else -LABEL_PENALTY
        out.append((o, score))
    return sorted(out, key=lambda t: -t[1])


def best(objs: list, text: str, store=None) -> dict:
    """
    The object to pick for `text`, or a reason not to pick anything.

    WHY: rank() always returns something, so "green square" with no green object on the table would grab a
    laptop key, and with only the green cylinder in view it would grab the cylinder. Refuses when the colour
    is wrong, when the library recognises the best candidate as a different object, or when the match is weak.
    Returns {"obj", "score", "margin", "ambiguous", "reason", "recognised"}; obj is None when refusing.
    """
    r = rank(objs, text, store)
    if not r:
        return {"obj": None, "score": 0.0, "margin": 0.0, "ambiguous": False, "reason": "no objects on the table",
                "recognised": None}
    (o, s), margin = r[0], (r[0][1] - r[1][1]) if len(r) > 1 else 1.0
    out = {"obj": None, "score": s, "margin": margin, "ambiguous": False, "reason": "", "recognised": o.recognised}
    qcol = set(clean_query(text).split()) & COLOURS
    if qcol and not colour_ok(qcol, o.colour):
        out["reason"] = f"no {'/'.join(sorted(qcol))} object found"
    elif o.recognised is not None and not name_matches(text, o.recognised_names):
        out["reason"] = f"best candidate is recognised as {o.recognised!r}"
    elif s < MIN_SCORE:
        out["reason"] = f"best match only {s:.2f}"
    else:
        out.update(obj=o, ambiguous=margin < AMBIGUOUS_MARGIN)
    return out
