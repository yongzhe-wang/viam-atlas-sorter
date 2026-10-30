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


def scene(top: int = 120):
    """1280x720 frame: wood table at 380 mm, blue tape ring (60 px) around white paper, interior from v=top."""
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (150, 110, 70)
    rgb[max(top - 60, 0):top + 500, 300:800] = (0, 64, 116)  # tape as it reads on sfhack4 (PIL hue ~147)
    rgb[max(top, 0):top + 440, 360:740] = (225, 230, 235)
    return rgb, np.full((720, 1280), 380, np.uint16)


def test_objects_on_and_across_the_tape():
    # A blue-ish object lying in the square (the silver knife reads 52-69 % "tape") must not split the interior, and
    # one lying across the tape must not join the interior to the table: depth says both are objects, not tape
    from sorter import perception as P

    rgb, d = scene()
    clean = P.find_drop_square(rgb, d)
    assert clean and not clean.get("partial"), clean
    knife = (rgb.copy(), d.copy())
    knife[0][320:360, 380:720], knife[1][320:360, 380:720] = (0, 64, 116), 362
    hit = P.find_drop_square(*knife)
    assert hit and hit["bbox"] == clean["bbox"], (hit and hit["bbox"], clean["bbox"])
    across = (rgb.copy(), d.copy())
    across[0][520:680, 520:580], across[1][520:680, 520:580] = (230, 200, 30), 350
    hit = P.find_drop_square(*across)
    assert hit and hit["bbox"] == clean["bbox"], (hit and hit["bbox"], clean["bbox"])
    cut = P.find_drop_square(*scene(top=-60))  # interior runs off the top of the frame
    assert cut and cut.get("partial") and cut["cut"] == ["top"], cut and cut.get("cut")


def test_cable_and_fenced_patch():
    # The robot's cable in front of the lens crosses the square: dark, and depth sees the table behind it
    # {SQUARE_CENTERED.NPZ "PX (500,300) V=27 DEPTH=375"}; and a bright patch fenced by dark objects, no tape, is no square
    from sorter import perception as P

    rgb, d = scene()
    for k in range(-30, 31):  # 60 px dark band from the top-left table corner through the interior's corner
        for t in range(0, 700):
            u, v = 250 + t + k, -100 + t
            if 0 <= v < 720 and 0 <= u < 1280:
                rgb[v, u] = (20, 22, 25)
    hit = P.find_drop_square(rgb, d)
    assert hit is not None, "the cable must not hide the square"
    rgb2, d2 = np.zeros((720, 1280, 3), np.uint8), np.full((720, 1280), 380, np.uint16)
    rgb2[:] = (150, 110, 70)
    rgb2[200:560, 400:880] = (20, 22, 25)  # dark frame (an object) ...
    rgb2[260:500, 460:820] = (225, 230, 235)  # ... around a bright patch
    assert P.find_drop_square(rgb2, d2) is None


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
