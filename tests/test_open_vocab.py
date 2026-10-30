"""
The open-set gate of clip_embed.best(), on hand-made embeddings: no CLIP model, no Store, no frames.

Pins what each kind of query may do: a known name keeps the library's veto; an unknown one is vetoed only when the crop
is clearly more like the recognised label; everything must beat the generic prompts. Run: python tests/test_open_vocab.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import clip_embed as C  # noqa: E402

# One axis per text; a crop is a mix of them, so its similarity to a prompt is exactly the weight given to that axis
AXES = ["mug", "blue cylinder", "red utility knife", "red knife", "green cube", "object"]
DIM = len(AXES) + 1  # the last axis is "nothing we have a prompt for" (keeps the vectors unit length)


def text(name: str) -> np.ndarray:
    v = np.zeros(DIM)
    v[AXES.index(name) if name in AXES else AXES.index("object")] = 1.0  # every generic prompt maps to "object"
    return v


def crop(**weights) -> np.ndarray:
    """A crop whose similarity to prompt k is weights[k] ("blue_cylinder" for "blue cylinder")."""
    v = np.zeros(DIM)
    for k, w in weights.items():
        v[AXES.index(k.replace("_", " "))] = w
    v[-1] = np.sqrt(max(0.0, 1.0 - float(v @ v)))
    return v


class Lib:
    """The library knows these names; recognition is given per object in the tests."""

    def list_labels(self):
        return [["blue cylinder"], ["red knife", "red utility knife"], ["green cube", "green square"]]

    def search_views(self, emb, k=8, label=None):
        return []


def rec(emb, label=None, names=(), colour="blue"):
    obj = NS(colour=colour, colours={colour: 1.0}, length_px=100.0, width_px=60.0, height_mm=30.0, crop=None)
    return {"obj": obj, "label": label, "names": list(names), "score": 0.9 if label else 0.0, "emb": emb}


def best(query, recs):
    return C.best(None, query, Lib(), recognised=recs)


def setup_module(_=None):
    C._name_emb = lambda t: text(C.clean_query(t))  # no model: prompts are the axes above


def test_unknown_query_is_not_vetoed_by_the_nearest_label():
    # Recognised as "blue cylinder" only because that is the nearest thing the library has; the crop prefers the query
    # {VIAM CENTERAL BASELINE "BEST CANDIDATE IS RECOGNISED AS X" REFUSED 29 OF 65}
    r = rec(crop(mug=0.40, blue_cylinder=0.30, object=0.30), "blue cylinder", ["blue cylinder"])
    b = best("mug", [r])
    assert b["obj"] is r["obj"] and b["recognised"] == "blue cylinder", b


def test_unknown_query_vetoed_when_clearly_the_known_object():
    r = rec(crop(mug=0.36, blue_cylinder=0.45, object=0.20), "blue cylinder", ["blue cylinder"])
    b = best("mug", [r])
    assert b["obj"] is None and b["reason"] == "best candidate is recognised as 'blue cylinder'", b
    assert b["searchable"] is False


def test_unknown_query_must_beat_the_generic_prompts():
    r = rec(crop(mug=0.34, object=0.33))  # looks like "an object" as much as like the query (banana on an empty table)
    b = best("mug", [r])
    assert b["obj"] is None and b["reason"] == "nothing here looks like 'mug'", b


def test_vetoed_object_does_not_make_the_right_one_ambiguous():
    known = rec(crop(mug=0.40, blue_cylinder=0.60), "blue cylinder", ["blue cylinder"])
    new = rec(crop(mug=0.39, object=0.25))
    b = best("mug", [known, new])
    assert b["obj"] is new["obj"] and not b["ambiguous"] and b["margin"] == 1.0, b


def test_two_candidates_close_together_are_ambiguous():
    a, c = rec(crop(mug=0.40, object=0.20)), rec(crop(mug=0.39, object=0.20))
    b = best("mug", [a, c])
    assert b["obj"] is a["obj"] and b["ambiguous"], b


def test_colour_word_still_refuses():
    r = rec(crop(mug=0.60, object=0.10), colour="green")
    b = best("red mug", [r])
    assert b["obj"] is None and b["reason"] == "no red object found", b


def test_known_name_keeps_the_library_veto():
    # Every demo query is a whole name: there the library still decides, as before the gate
    r = rec(crop(green_cube=0.60, blue_cylinder=0.10), "blue cylinder", ["blue cylinder"], colour="green")
    b = best("green cube", [r])
    assert b["obj"] is None and b["reason"] == "best candidate is recognised as 'blue cylinder'" and b["searchable"], b
    unrecognised = rec(crop(green_cube=0.60), colour="green")
    b = best("green cube", [unrecognised])
    assert b["obj"] is None and b["reason"] == "no 'green cube' recognised on the table" and b["searchable"], b


def test_overlap_only_query():
    # "knife" is only part of "red knife": the red one in view is the answer, as before ...
    red = rec(crop(red_knife=0.50, object=0.30), "red knife", ["red knife", "red utility knife"], colour="red")  # "knife" -> object axis
    assert best("knife", [red])["obj"] is red["obj"]
    # ... with none in view another knife may be picked by the gate; a refusal stays searchable (it may be elsewhere)
    C._name_emb = lambda t: text("mug" if C.clean_query(t) == "knife" else C.clean_query(t))
    try:
        other = rec(crop(mug=0.40, object=0.20), colour="grey")
        assert best("knife", [other])["obj"] is other["obj"]
        nothing = rec(crop(mug=0.10, object=0.30), colour="grey")
        b = best("knife", [nothing])
        assert b["obj"] is None and b["searchable"], b
    finally:
        setup_module()


if __name__ == "__main__":
    setup_module()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
