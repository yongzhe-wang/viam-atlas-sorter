"""
CLIP embeddings and text ranking: which object on the table did the user mean?

WHY: the demo command is a free description ("green square", "the cup"); CLIP puts text and image in one
512-d space, so the description can be compared with every object crop directly, with no training.
CLIP is weak on colour + shape words, so two cheap cues from segmentation are added on top: the
object's dominant colour and its footprint aspect ratio. With a Store, labelled views of scanned objects
also vote, which is how teach-once and the object library improve recognition.

Contract: docs/EXEC_PLAN.md (embed_images, embed_text, rank). Model: open_clip ViT-B-32, CPU.
"""
import re

import numpy as np

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
COLOURS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "white", "grey", "black"}
ALIASES = {"gray": "grey", "violet": "purple", "pink": "red"}
# Command verbs/fillers stripped before embedding; the destination is always the blue square
STOP = r"^(please\s+)?(grasp|grab|pick( up)?|get|take|move|put|place)\s+(up\s+)?(the\s+|a\s+|an\s+)?"
COLOUR_BONUS = 0.04  # colour word matches the object's dominant colour (CLIP sims sit around 0.2-0.3)
COLOUR_PENALTY = 0.04  # colour word names a different, saturated colour
SHAPE_BONUS = 0.02
STORE_WEIGHT = 0.10  # weight of labelled-view votes from the object library

_model = _preprocess = _tokenizer = None


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


def _shape_term(words: set, obj) -> float:
    """Small bonus when shape words agree with the footprint (square ~ aspect 1, long ~ aspect > 2.5)."""
    aspect = obj.length_px / max(obj.width_px, 1.0)
    if words & {"square", "cube", "box"}:
        return SHAPE_BONUS if aspect < 1.4 else -SHAPE_BONUS
    if words & {"long", "stick", "pen", "rectangle", "bar"}:
        return SHAPE_BONUS if aspect > 2.0 else -SHAPE_BONUS
    return 0.0


def rank(objs: list, text: str, store=None) -> list:
    """
    [(Obj, score)] best first for a free-text description.

    score = CLIP(crop, "a photo of a {query}") + colour term + shape term + store vote.
    Store vote: the object's crop retrieves its nearest labelled views; each vote is the view's score
    times how well that view's label matches the query (CLIP text-text cosine).
    """
    if not objs:
        return []
    q = clean_query(text)
    words = set(q.split())
    img = embed_images([o.crop for o in objs])
    sims = img @ embed_text(q)
    qcol = words & COLOURS
    qt = embed_text(q) if store is not None else None
    out = []
    for o, emb, s in zip(objs, img, sims):
        score = float(s)
        if qcol:
            if o.colour in qcol:
                score += COLOUR_BONUS
            elif o.colour in COLOURS - {"grey", "white", "black"}:
                score -= COLOUR_PENALTY
        score += _shape_term(words, o)
        if store is not None:
            views = store.search_views(emb, k=5)
            if views:
                votes = [v["score"] * float(embed_text(v["label"]) @ qt) for v in views]
                score += STORE_WEIGHT * max(votes)
        out.append((o, score))
    return sorted(out, key=lambda t: -t[1])
