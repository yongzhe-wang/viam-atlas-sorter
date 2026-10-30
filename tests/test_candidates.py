"""
clip_embed.candidates() on hand-made embeddings: ranks everything, refuses nothing; best() stays as it was.

No CLIP model, no Store, no frames. Run: python tests/test_candidates.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import clip_embed as C  # noqa: E402

AXES = ["snack", "m&m", "kind bar", "granola bar", "tape", "object"]
DIM = len(AXES) + 1


def text(name):
    v = np.zeros(DIM)
    v[AXES.index(name) if name in AXES else AXES.index("object")] = 1.0
    return v


def crop(**w):
    v = np.zeros(DIM)
    for k, x in w.items():
        v[AXES.index({"mm": "m&m", "kind": "kind bar", "granola": "granola bar"}.get(k, k))] = x
    v[-1] = np.sqrt(max(0.0, 1.0 - float(v @ v)))
    return v


class Lib:
    def list_labels(self):
        return [["kind bar", "snack bar", "candy"], ["granola bar"], ["black tape", "tape"]]

    def search_views(self, emb, k=8, label=None):
        h = {"kind bar": 20.0, "granola bar": 31.0, "black tape": 25.0}
        return [{"label": label, "height_mm": h[label], "score": 0.9}] if label in h else []


def rec(emb, label=None, names=(), h=20.0, width=120.0, length=300.0, edge=False):
    o = NS(colour="black", colours={"black": 1.0}, length_px=length, width_px=width, height_mm=h, crop=None,
           top_depth_mm=350.0, touches_edge=edge)
    return {"obj": o, "label": label, "names": list(names), "score": 0.9 if label else 0.0, "emb": emb}


def cands(q, recs):
    return C.candidates([r["obj"] for r in recs], q, Lib(), recognised=recs)


def setup_module(_=None):
    C._name_emb = lambda t: text(C.clean_query(t))


def test_category_word_prefers_the_graspable_one():
    # {CROPS MULTI1.JPG "SNACK 0.197 / 0.280": THE KIND BAR'S BACK LOSES ON CLIP; THE M&M BAG IS 7 MM HIGH}
    kind = rec(crop(snack=0.20, mm=0.21, object=0.24), "granola bar", ["granola bar"], h=20.0)
    mm = rec(crop(snack=0.28, mm=0.29, object=0.21), h=7.0)
    c = cands("snack", [kind, mm])
    assert c[0]["obj"] is kind["obj"] and c[0]["graspable"] and not c[1]["graspable"], c
    assert "too flat" in c[1]["graspable_why"]


def test_explicit_name_is_not_overruled_by_graspability():
    kind = rec(crop(snack=0.20, mm=0.21, object=0.24), "granola bar", ["granola bar"], h=20.0)
    mm = rec(crop(snack=0.28, mm=0.29, object=0.21), h=7.0)
    assert cands("m&m", [kind, mm])[0]["obj"] is mm["obj"]


def test_known_name_not_recognised_is_still_a_candidate():
    # {VIAM CENTERAL "KIND BAR 翻到背面被认成 GRANOLA BAR，KIND BAR 直接 NOT_FOUND"}: best() refuses, candidates() lists it first
    back = rec(crop(kind=0.25, granola=0.23, object=0.20), "granola bar", ["granola bar"], h=20.0)
    other = rec(crop(kind=0.10, tape=0.40, object=0.20), "black tape", ["black tape", "tape"], h=25.0)
    r = [back, other]
    assert C.best(None, "kind bar", Lib(), recognised=r)["obj"] is None
    c = cands("kind bar", r)
    assert c[0]["obj"] is back["obj"] and c[0]["source"] == "clip" and "height fits" in c[0]["why"], c


def test_named_by_the_library_wins_and_says_so():
    tape = rec(crop(tape=0.25, object=0.20), "black tape", ["black tape", "tape"], h=25.0)
    other = rec(crop(tape=0.30, object=0.20), h=30.0)
    c = cands("tape", [other, tape])
    assert c[0]["obj"] is tape["obj"] and c[0]["source"] == "library" and c[0]["likely"], c


def test_refuses_nothing_and_flags_what_cannot_be_grasped():
    r = [rec(crop(object=0.5)), rec(crop(object=0.5), edge=True), rec(crop(object=0.5), width=300.0), rec(crop(object=0.5), width=10.0)]
    c = cands("banana", r)
    assert len(c) == 4 and not any(x["likely"] for x in c)
    assert sorted(x["graspable_why"].split(" (")[0] for x in c) == ["", "a fragment", "cut by the image border", "too wide"], c


if __name__ == "__main__":
    setup_module()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
