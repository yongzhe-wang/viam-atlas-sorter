"""
perception.MERGE_FRAGMENTS on a synthetic frame: a clear bottle is two opaque pieces inside a region with no depth.

No frames and no CLIP needed. Run: python tests/test_merge_fragments.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import perception as P  # noqa: E402

TABLE = 376


def scene(neighbour: bool = False):
    """Wood table; a depth-less bottle body u 600-780, v 90-630, with a cap and a label that have depth."""
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (150, 110, 70)
    d = np.full((720, 1280), float(TABLE))
    d[90:630, 600:780] = 0.0  # see-through body: no depth
    rgb[90:630, 600:780] = (170, 160, 150)
    for (v0, v1, u0, u1), h in (((100, 150, 650, 730), 45), ((230, 380, 610, 770), 63)):
        d[v0:v1, u0:u1] = TABLE - h
        rgb[v0:v1, u0:u1] = (200, 210, 220)
    if neighbour:  # an opaque block touching the see-through part of the body, below the label
        d[430:590, 500:600] = TABLE - 30
        rgb[430:590, 500:600] = (230, 200, 30)
    return rgb, d


def segment(rgb, d, on: bool):
    saved = P.MERGE_FRAGMENTS
    P.MERGE_FRAGMENTS = on
    try:
        return P.segment_full(rgb, d)
    finally:
        P.MERGE_FRAGMENTS = saved


def test_flag_off_keeps_the_fragments():
    objs = segment(*scene(), False).objects
    assert len(objs) == 2 and all(o.grip_px is None and o.n_fragments == 1 for o in objs), objs


def test_fragments_become_one_object():
    # {VIAM CENTERAL 2026-10-30 "BBOX / FOOTPRINT / 长宽比按完整物体算; HEIGHT_MM 用最高点; GRIP_PX 取不透明 … 最接近合并物体重心的那一块的中心"}
    objs = segment(*scene(), True).objects
    assert len(objs) == 1, [o.bbox for o in objs]
    o = objs[0]
    assert o.n_fragments == 2 and o.bbox[0] <= 600 and o.bbox[1] <= 95 and o.bbox[2] >= 775 and o.bbox[3] >= 625, o.bbox
    assert o.length_px > 2 * o.width_px and o.height_mm > 55
    assert abs(o.grip_px[0] - 690) < 5 and abs(o.grip_px[1] - 305) < 5, o.grip_px  # the label, not the cap
    assert o.grip_depth_mm < TABLE - 55 and o.mask[360, 690] and o.crop.shape[0] > 540


def test_touching_opaque_object_stays_separate():
    # {VIAM CENTERAL "不能把紧挨着的两个不同物体并起来"}
    objs = segment(*scene(neighbour=True), True).objects
    assert len(objs) == 2, [o.bbox for o in objs]
    block = next(o for o in objs if o.n_fragments == 1)
    bottle = next(o for o in objs if o.n_fragments == 2)
    assert block.grip_px is None and block.bbox[2] < 610 and bottle.bbox[0] >= 590, (block.bbox, bottle.bbox)


def test_depthless_region_on_the_border_is_no_body():
    rgb, d = scene()
    d[:, :700] = np.where(d[:, :700] == TABLE, 0.0, d[:, :700])  # floor beyond the table edge, joined to the frame
    assert all(o.grip_px is None for o in segment(rgb, d, True).objects)


def test_sliver_joins_its_larger_neighbour():
    # A cylinder's side segmented apart from its top (076): with an unknown name the jaws went for the sliver
    # {OPENVOCAB "BLUE CYLINDER@076: WRONG (PICKED CYAN H=14)"}
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (150, 110, 70)
    d = np.full((720, 1280), float(TABLE))
    d[300:420, 500:620], rgb[300:420, 500:620] = TABLE - 60, (20, 90, 200)  # the object
    d[330:370, 640:690], rgb[330:370, 640:690] = TABLE - 14, (20, 90, 200)  # its sliver, 20 px away
    d[330:370, 300:350], rgb[330:370, 300:350] = TABLE - 14, (20, 90, 200)  # a small thing far away: stays
    d[440:480, 540:590], rgb[440:480, 540:590] = TABLE - 14, (220, 30, 30)  # a small red thing just below: stays
    off = segment(rgb, d, False).objects
    on = segment(rgb, d, True).objects
    assert len(off) == 4 and len(on) == 3, ([o.bbox for o in off], [o.bbox for o in on])
    big = max(on, key=lambda o: o.mask.sum())
    assert big.n_fragments == 2 and big.bbox[2] >= 689 and big.grip_px is None, big.bbox
    assert [o.id for o in on] == [0, 1, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
