"""
perception's hand check on synthetic frames: only what is over the table pauses the arm, not a thing on the floor past its edge.

No frames and no CLIP needed. Run: python tests/test_intruders.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import perception as P  # noqa: E402

TABLE = 376
FLOOR = 1126  # ~750 mm below the table top
SKIN = (210, 150, 120)  # in the YCrCb skin range (and light wood is too: why the table around it matters)


def frame(edge_u: int = None):
    """The table, and past edge_u (if given) the floor."""
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (150, 110, 70)
    d = np.full((720, 1280), float(TABLE))
    if edge_u is not None:
        d[:, edge_u:] = FLOOR
        rgb[:, edge_u:] = (40, 40, 45)  # dark carpet
    return rgb, d


def segment(rgb, d):
    return P.segment_full(rgb, d, align=P.IDENTITY)


def test_skin_coloured_hole_over_the_table_is_a_hand():
    # A hand inside the depth blind range reads no depth: a hole, skin-coloured, with the table all around it
    rgb, d = frame()
    d[250:420, 600:780] = 0
    rgb[250:420, 600:780] = SKIN
    assert [i["reason"].startswith("blind-range hole") for i in segment(rgb, d).intruders] == [True]


def test_hole_on_the_floor_past_the_table_is_no_hand():
    # {VIAM CENTERAL 2026-10-30 "桌子外面地上一个缠了蓝胶带的物体 … 被框成 'HAND: THE ARM WAITS'"}
    # [CONFIDENCE: INFERRED 65% — synthetic: no real frame of that floor object yet]
    rgb, d = frame(edge_u=900)
    d[250:420, 1000:1180] = 0  # no depth on the carpet and the thing lying on it
    rgb[250:420, 1000:1180] = SKIN
    assert segment(rgb, d).intruders == [], segment(rgb, d).intruders


def test_tall_thing_past_the_table_is_no_hand():
    # A chair back beside the table: tall above the table's plane, but the floor is around it
    rgb, d = frame(edge_u=900)
    d[250:420, 1000:1180] = TABLE - 150
    assert segment(rgb, d).intruders == [], segment(rgb, d).intruders


def test_tall_thing_on_the_table_stays_an_intruder():
    # A standing object taller than any graspable one pauses the arm, as before; also when its rim reads no depth around it
    # {VIAM CENTERAL 2026-10-30 "保持现在的规则（立着的高物体算 INTRUDER）"}
    rgb, d = frame()
    d[250:420, 600:780] = TABLE - 150
    assert [i["reason"].startswith("blob 150 mm tall") for i in segment(rgb, d).intruders] == [True]
    d[230:440, 580:800] = 0  # a clear jug: no depth around the rim
    d[250:420, 600:780] = TABLE - 150
    assert any(i["reason"].startswith("blob 150 mm tall") for i in segment(rgb, d).intruders), segment(rgb, d).intruders


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
