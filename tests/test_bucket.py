"""
sorter.bucket on synthetic frames: an orange bucket is found, located at its rim's height, and cut out of segmentation.

No frames and no CLIP needed. Run: python tests/test_bucket.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import bucket as B, perception as P  # noqa: E402

TABLE = 556.0  # the z 580 view
POSE = {"x": 250.0, "y": 0.0, "z": 580.0, "theta": 0.0}
K = dict(P.INTRINSICS)


def scene(cu=680, cv=280, r=230, block=True):
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (190, 150, 110)
    d = np.full((720, 1280), TABLE)
    vv, uu = np.ogrid[:720, :1280]
    disc = (uu - cu) ** 2 + (vv - cv) ** 2 <= r ** 2
    rim = disc & ((uu - cu) ** 2 + (vv - cv) ** 2 >= (r - 20) ** 2)
    rgb[disc] = (240, 110, 30)  # pumpkin orange
    d[disc] = TABLE - 20.0  # its inside, seen from above: the bottom
    d[rim] = TABLE - 175.0  # the rim
    if block:  # an object on the table beside it
        rgb[500:580, 100:200] = (30, 120, 220)
        d[500:580, 100:200] = TABLE - 30.0
    return rgb, d


def test_found_and_located_at_the_rim():
    # {VIAM CENTERAL 2026-10-30 "XY 要用桶沿的深度算（透视！桶沿不在桌面平面上）"}
    rgb, d = scene()
    b = B.find_bucket(rgb, d, POSE, K)
    assert b is not None and not b["partial"], b
    rim_depth = TABLE - b["rim_height_mm"]
    want = P.world_xy(POSE, K, 680, 280, rim_depth)
    assert abs(b["xy"][0] - want[0]) < 3 and abs(b["xy"][1] - want[1]) < 3, (b["xy"], want)
    assert abs(b["rim_height_mm"] - 175.0) < 8 and abs(b["radius_mm"] - 230 * rim_depth / K["fx"]) < 4, b


def test_cut_out_of_segmentation():
    rgb, d = scene()
    saved = B.TARGET
    try:
        B.TARGET = "square"
        off = P.segment_full(rgb, d, align=P.IDENTITY)
        B.TARGET = "bucket"
        on = P.segment_full(rgb, d, align=P.IDENTITY)
    finally:
        B.TARGET = saved
    # without the cut the bucket is a tall blob: never a target either way (no intruders since 2026-10-30)
    assert off.bucket is None and not off.intruders and not on.intruders and on.bucket is not None
    assert [o.colour for o in on.objects] == ["blue"], [(o.colour, o.bbox) for o in on.objects]


def test_partial_when_cut_by_the_frame():
    rgb, d = scene(cu=640, cv=-120, r=300, block=False)
    b = B.find_bucket(rgb, d, POSE, K)
    assert b is not None and b["partial"], b


def test_small_orange_thing_is_no_bucket():
    rgb, d = scene(block=False)
    rgb[:] = (190, 150, 110)
    d[:] = TABLE
    rgb[300:360, 300:380] = (240, 110, 30)  # an orange block, 60x80 px
    d[300:360, 300:380] = TABLE - 30
    assert B.find_bucket(rgb, d, POSE, K) is None


def test_oblique_bucket_is_cut_but_not_located():
    # An ellipse, not a circle: find_mask still returns the region to cut out, find_bucket returns None (no crash)
    rgb, d = scene(block=False)
    rgb[:] = (190, 150, 110)
    d[:] = TABLE
    vv, uu = np.ogrid[:720, :1280]
    ell = ((uu - 640) / 380.0) ** 2 + ((vv - 360) / 150.0) ** 2 <= 1.0
    rgb[ell], d[ell] = (240, 110, 30), TABLE - 175.0
    hit = B.find_mask(rgb)
    assert hit is not None and hit["center_px"] is None and hit["mask"][360, 640]
    assert B.find_bucket(rgb, d, POSE, K) is None


def test_map_contains():
    e = {"xy": [335.0, 18.0], "radius_mm": 98.0}
    assert B.contains(e, (340.0, 20.0)) and not B.contains(e, (450.0, 18.0)) and not B.contains(None, (0, 0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
