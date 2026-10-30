"""
sorter.camera_model: rays to the table for any wrist pose, and the table cells a frame saw. No frames needed.

Run: python tests/test_camera_model.py
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import camera_model as M, perception as P  # noqa: E402

K = dict(P.INTRINSICS)


def render_table(pose_full, box=None):
    """Depth image of the table plane z = 0 (and a box: (x0, x1, y0, y1, height)) seen from pose_full, by ray casting."""
    vv, uu = np.mgrid[0:720, 0:1280]
    a = M.camera_to_world(pose_full, K, uu, vv, np.full(uu.shape, 1.0))  # the point at 1 mm along each ray
    o = M.camera_to_world(pose_full, K, 640, 360, 0.0)  # the camera centre
    ray = a - o
    t = -o[2] / ray[..., 2]  # depth (mm along the optical axis) where z reaches 0
    depth = np.where(ray[..., 2] < 0, t, 0.0)
    if box is not None:
        x0, x1, y0, y1, h = box
        tb = (h - o[2]) / ray[..., 2]
        hit = o + ray * tb[..., None]
        on = (ray[..., 2] < 0) & (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & (hit[..., 1] >= y0) & (hit[..., 1] <= y1)
        depth = np.where(on, tb, depth)
    return depth


def test_straight_down_equals_world_xy():
    random.seed(0)
    for _ in range(200):
        pose = {"x": random.uniform(100, 400), "y": random.uniform(-300, 300), "z": random.uniform(380, 600), "theta": random.uniform(-180, 180)}
        u, v, d = random.uniform(0, 1280), random.uniform(0, 720), random.uniform(280, 600)
        a = P.world_xy(pose, K, u, v, d)
        b = M.world_xy_full({**pose, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0}, K, u, v, d)
        assert math.dist(a, b) < 1e-6, (pose, a, b)


def test_tilted_table_maps_back_to_z0():
    pose = [300.0, -120.0, 480.0, 0.0, 0.423, -0.906, 0.0]
    d = render_table(pose)
    vv, uu = np.mgrid[0:720:16, 0:1280:16]
    w = M.camera_to_world(pose, K, uu, vv, d[::16, ::16])
    ok = d[::16, ::16] > 0
    assert np.abs(w[..., 2][ok]).max() < 1e-6


def test_visible_cells_straight_down_covers_the_footprint():
    pose = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}
    cells = M.visible_cells(render_table({**pose, "o_x": 0, "o_y": 0, "o_z": -1}), pose, K)
    assert not cells["object"]
    # the view at 376 mm is ~529 x 298 mm: about 26 x 15 cells of 20 mm
    assert 330 <= len(cells["table"]) <= 450, len(cells["table"])


def test_cells_behind_a_near_object_are_not_seen():
    # {VIAM CENTERAL "被近处物体挡住的格子不算"}: tilted view, a 120 mm tall box hides the table behind it
    pose = [200.0, 0.0, 480.0, 0.422, 0.0, -0.907, 0.0]
    box = (380.0, 440.0, -40.0, 40.0, 120.0)
    free = M.visible_cells(render_table(pose), pose, K)
    hidden = M.visible_cells(render_table(pose, box), pose, K)
    lost = {tuple(c) for c in free["table"]} - {tuple(c) for c in hidden["table"]}
    assert hidden["object"], "the box top is seen"
    xs = [c[0] * 20 + 10 for c in lost]
    assert lost and min(xs) >= 380, sorted(lost)  # what is lost lies at and beyond the box, seen from the -x side


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
