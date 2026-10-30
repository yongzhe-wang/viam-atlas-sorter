"""
perception.LOW_OBJECTS on synthetic frames: a flat coloured thing on the table is found; tape, paper and shadows are not.

No frames and no CLIP needed. Run: python tests/test_low_objects.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import perception as P  # noqa: E402

TABLE = 376.0
WOOD = (190, 150, 110)


def frame():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = WOOD
    d = np.full((720, 1280), TABLE)
    return rgb, d


def patch(rgb, d, v0, v1, u0, u1, colour, h):
    rgb[v0:v1, u0:u1] = colour
    d[v0:v1, u0:u1] = TABLE - h


def flats(rgb, d, on=True):
    saved = P.LOW_OBJECTS
    P.LOW_OBJECTS = on
    try:
        s = P.segment_full(rgb, d, align=P.IDENTITY)  # the startup alignment check is for real frames
    finally:
        P.LOW_OBJECTS = saved
    return [o for o in s.objects if o.flat], [o for o, _ in s.excluded if o.flat], s


def test_flat_bag_is_found():
    # {VIAM CENTERAL 2026-10-30 "M&M'S 袋 约 100X75 MM, H≈7, BLOCK TOP NOT FOUND IN DEPTH MAP"}
    rgb, d = frame()
    patch(rgb, d, 300, 500, 200, 450, (60, 30, 25), 6.0)
    whole, _, _ = flats(rgb, d)
    assert len(whole) == 1, whole
    o = whole[0]
    assert abs(o.height_mm - 6.0) < 1.0 and abs(o.top_depth_mm - (TABLE - 6.0)) < 1.0, (o.height_mm, o.top_depth_mm)
    assert o.bbox[0] <= 205 and o.bbox[2] >= 445 and o.crop.size > 0 and not o.touches_edge


def test_flag_off_finds_none():
    rgb, d = frame()
    patch(rgb, d, 300, 500, 200, 450, (60, 30, 25), 6.0)
    assert flats(rgb, d, on=False)[0] == []


def test_cut_by_the_frame_is_excluded_not_a_target():
    rgb, d = frame()
    patch(rgb, d, 550, 720, 200, 450, (60, 30, 25), 6.0)
    whole, cut, _ = flats(rgb, d)
    assert not whole and len(cut) == 1


def test_tape_paper_shadow_and_wood_are_not_objects():
    # {SCENE2 HEIGHT MAP: THE PAPER SQUARE CURLS UP SEVERAL MM}
    rgb, d = frame()
    patch(rgb, d, 100, 300, 100, 400, (20, 70, 150), 4.0)  # blue tape, curled up
    patch(rgb, d, 100, 300, 500, 800, (235, 235, 230), 4.0)  # white paper, curled up
    patch(rgb, d, 400, 600, 100, 400, (70, 55, 40), 0.0)  # a dark shadow: colour, no height
    patch(rgb, d, 400, 600, 500, 800, WOOD, 4.0)  # a raised plank: height, no colour
    assert flats(rgb, d)[0] == [] and flats(rgb, d)[1] == []


def test_thin_low_strip_is_no_object():
    rgb, d = frame()
    patch(rgb, d, 300, 322, 100, 700, (30, 30, 30), 5.0)  # a cable lying flat: ~9 mm wide
    assert flats(rgb, d)[0] == []


def test_creases_of_a_bag_are_replaced_by_the_whole_bag():
    rgb, d = frame()
    patch(rgb, d, 300, 500, 200, 450, (60, 30, 25), 6.0)
    patch(rgb, d, 380, 440, 280, 360, (60, 30, 25), 10.0)  # a crease above MIN_HEIGHT_MM: the normal path sees only this
    whole, _, s = flats(rgb, d)
    assert len(s.objects) == 1 and len(whole) == 1 and whole[0].bbox[2] >= 445, [o.bbox for o in s.objects]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
