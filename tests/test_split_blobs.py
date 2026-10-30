"""
perception.SPLIT_BLOBS on synthetic frames: a cable joined to an object is cut off, thin parts of objects are not.

No frames and no CLIP needed. Run: python tests/test_split_blobs.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import perception as P  # noqa: E402

TABLE = 376


def frame():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (150, 110, 70)
    return rgb, np.full((720, 1280), float(TABLE))


def box(rgb, d, v0, v1, u0, u1, h, colour=(20, 20, 20)):
    d[v0:v1, u0:u1] = TABLE - h
    rgb[v0:v1, u0:u1] = colour


def segment(rgb, d, on: bool):
    saved = P.SPLIT_BLOBS
    P.SPLIT_BLOBS = on
    try:
        return P.segment_full(rgb, d)
    finally:
        P.SPLIT_BLOBS = saved


def mm(o):
    return o.width_px * o.top_depth_mm / P.INTRINSICS["fx"]


def test_cable_to_the_frame_is_cut_off():
    # {VIAM CENTERAL 2026-10-30 "螺丝刀 + 线缆变成螺丝刀单独一块，宽度 < 60 MM"}
    rgb, d = frame()
    box(rgb, d, 300, 370, 500, 800, 30)  # a handle, ~30 x 120 mm ...
    box(rgb, d, 370, 719, 640, 664, 15)  # ... with a 10 mm cable leaving it end-on, down to the image border
    off = segment(rgb, d, False)
    assert len(off.objects) == 0 and off.excluded[0][0].touches_edge, "one blob cut by the frame without the flag"
    on = segment(rgb, d, True)
    assert len(on.objects) == 1 and len(on.cables) == 1, (on.objects, on.cables)
    o = on.objects[0]
    assert o.bbox[3] < 400 and mm(o) < 60 and not o.touches_edge, o.bbox


def test_cable_between_two_objects_makes_two():
    rgb, d = frame()
    box(rgb, d, 300, 370, 200, 420, 30)
    box(rgb, d, 300, 370, 860, 1080, 30)
    box(rgb, d, 325, 349, 420, 860, 15)  # 180 mm long, 10 mm wide: no part of an object is like that
    assert len(segment(rgb, d, False).objects) == 1
    on = segment(rgb, d, True)
    assert len(on.objects) == 2 and len(on.cables) == 1, ([o.bbox for o in on.objects], on.cables)


def test_screwdriver_shaft_stays():
    # Thin and long, but it ends in the open: part of the object
    rgb, d = frame()
    box(rgb, d, 300, 370, 300, 560, 30)
    box(rgb, d, 328, 342, 560, 800, 12)  # ~100 mm shaft
    on = segment(rgb, d, True)
    assert len(on.objects) == 1 and not on.cables and on.objects[0].bbox[2] >= 795, (on.cables, [o.bbox for o in on.objects])


def test_thin_sliver_at_the_frame_stays_a_partial_sighting():
    rgb, d = frame()
    box(rgb, d, 700, 720, 500, 700, 15)  # the end of a bar, the rest out of view
    on = segment(rgb, d, True)
    assert not on.cables and len(on.excluded) == 1, (on.cables, on.excluded)


def test_flat_table_unchanged():
    rgb, d = frame()
    box(rgb, d, 300, 400, 500, 600, 30)
    a, b = segment(rgb, d, False), segment(rgb, d, True)
    assert [o.bbox for o in a.objects] == [o.bbox for o in b.objects] and not b.cables


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
