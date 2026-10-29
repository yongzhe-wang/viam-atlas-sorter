"""
Tests for sorter.drop_square on synthetic views: a turned 114.5 x 175.5 mm interior, whole or clipped by the frame.

A view is the convex hull of the visible interior in table mm with each edge marked tape (real) or frame (cut),
which is what drop_square.observe() makes of a find_drop_square() hit. Run: python tests/test_drop_square.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import drop_square as D  # noqa: E402

C, YAW, SIZE = (370.0, 84.0), 8.0, D.SIZE_MM


def rect(c=C, yaw=YAW, size=SIZE):
    """Corners (counter-clockwise) of the interior."""
    t = math.radians(yaw)
    ax, bx = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    c = np.asarray(c, float)
    ha, hb = ax * size[0] / 2, bx * size[1] / 2
    return [c - ha - hb, c + ha - hb, c + ha + hb, c - ha + hb]


def view(poly, clips=()):
    """Clip the polygon by half-planes n.p <= d (the frame); edges made by a clip are not tape."""
    pts, real = [np.asarray(p, float) for p in poly], [True] * len(poly)
    for n, d in clips:
        n = np.asarray(n, float)
        out, out_real = [], []
        for k in range(len(pts)):
            p, q, r = pts[k], pts[(k + 1) % len(pts)], real[k]
            ip, iq = n @ p <= d, n @ q <= d
            if ip:
                out.append(p)
                out_real.append(r)
            if ip != iq:
                x = p + (q - p) * (d - n @ p) / (n @ (q - p))
                out.append(x)
                # entering: x -> q is the rest of this tape edge; leaving: x -> next point runs along the frame
                out_real.append(r if iq else False)
        # the edge from the exit point to the entry point lies on the frame
        pts, real = out, out_real
        for k in range(len(pts)):
            q = pts[(k + 1) % len(pts)]
            if abs(n @ pts[k] - d) < 1e-6 and abs(n @ q - d) < 1e-6:
                real[k] = False
    return {"hull_xy": [p.tolist() for p in pts], "real": real, "cut": [], "partial": not all(real)}


def close(a, b, tol):
    return math.dist(a, b) <= tol


def test_full_view():
    s = D.update(None, view(rect()), 1.0)
    assert close(s["xy"], C, 0.5) and abs(s["yaw_deg"] - YAW) < 0.2, s
    assert not s["partial"] and s["confidence"] == 1.0 and s["size_seen"] == [True, True]
    assert abs(s["size_mm"][0] - SIZE[0]) < 0.5 and abs(s["size_mm"][1] - SIZE[1]) < 0.5


def test_cut_by_one_side():
    # The far end (world +x) outside the view, as on 080/082: three tape edges seen
    # {VIAM CENTERAL 2026-10-29 "A BORDER-CUT / PARTIAL SQUARE IN ANY FRAME MUST STILL GIVE A USABLE ESTIMATE"}
    s = D.update(None, view(rect(), [((1, 0), 400.0)]), 1.0)
    assert s["partial"] and sorted(s["sides"]) == ["a0", "b0", "b1"], s["sides"]
    assert close(s["xy"], C, 1.0), s["xy"]  # prior size = true size here
    assert D.contains(s, (C[0] + 50, C[1]))  # inside the full interior, beyond the visible part


def test_corner_cut():
    s = D.update(None, view(rect(), [((1, 0), 400.0), ((0, 1), 120.0)]), 1.0)
    assert s["partial"] and sorted(s["sides"]) == ["a0", "b0"], s["sides"]
    assert close(s["xy"], C, 1.5) and abs(s["yaw_deg"] - YAW) < 0.5


def test_partial_views_combine():
    # Near half and far half from two poses: together all four sides, the size along a measured, not assumed
    big = (130.0, 175.5)
    near = view(rect(size=big), [((1, 0), 390.0)])
    far = view(rect(size=big), [((-1, 0), -350.0)])
    s = D.update(None, near, 1.0)
    assert s["partial"]
    s = D.update(s, far, 2.0)
    assert not s["partial"] and s["confidence"] == 1.0, s
    assert close(s["xy"], C, 1.0) and abs(s["size_mm"][0] - big[0]) < 1.0, (s["xy"], s["size_mm"])


def test_moved_and_turned():
    s = D.update(None, view(rect()), 1.0)
    moved = D.update(s, view(rect(c=(C[0] + 28, C[1] + 2), yaw=-1.2)), 2.0)  # 076 -> 084 on the table
    assert close(moved["xy"], (C[0] + 28, C[1] + 2), 0.5) and abs(moved["yaw_deg"] + 1.2) < 0.2
    shifted = D.update(s, view(rect(c=(C[0], C[1] + 40))), 3.0)  # same yaw, pushed 40 mm: restart, not averaged
    assert close(shifted["xy"], (C[0], C[1] + 40), 0.5)
    turned = D.update(s, view(rect(yaw=YAW + 90), [((1, 0), 400.0)]), 4.0)  # turned 90 deg, cut: sizes swap
    assert np.allclose(turned["size_learned"], s["size_learned"][::-1], atol=0.5), turned["size_learned"]
    assert close(turned["xy"], C, 1.0), turned["xy"]


def test_legacy_entry():
    old = {"xy": [369.9, 84.0], "size_mm": [114.0, 175.0], "partial": False, "seen_ts": 5.0}
    s = D.from_legacy(old)
    assert close(s["xy"], old["xy"], 0.1) and not s["partial"] and D.contains(s, (369.9, 84.0))
    assert D.from_legacy({**old, "partial": True}) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
