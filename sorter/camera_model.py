"""
Camera rays to table coordinates for any wrist pose, not only straight down; and which table cells a frame really saw.

WHY: the arm now also looks across the table with the wrist tilted (~25 deg). perception.world_xy() assumes the camera
looks straight down, so a tilted frame's objects landed on the wrong table cells, and AGENT1's coverage map needs to know
which cells a frame has actually seen (not the ones hidden behind a near object).
{VIAM CENTERAL 2026-10-30 "VISIBLE_CELLS(DEPTH, POSE_FULL, INTR, TABLE_PLANE)：返回这一帧真正看到桌面（或物体顶面）的世界坐标格子；
被近处物体挡住的格子不算"} [CONFIDENCE: CONFIRMED 80% — identical to world_xy straight down (tested); tilted 25 deg: the bucket's rim centre from three
tilted frames (330.7, 17.0) (334.8, 15.5) (329.9, 20.3) against (335.4, 17.8) straight down; table points at z 2 mm]

Model: the wrist turns about pose - TOOL_AXIS_MM (as perception.world_xy has it), the arm's orientation vector is Viam's
(R = Rz(lon) Ry(lat) Rz(theta), lat = acos(o_z), lon = atan2(o_y, o_x)), and the camera's pose in the tool frame is what
makes this model equal world_xy for o = (0, 0, -1). The camera sits CAM_BELOW_POSE_MM below the pose point when straight down
(table at 376 mm from the camera at z 400, 556 mm at z 580).
pose_full: {x, y, z, o_x, o_y, o_z, theta (degrees)} or the 7-vector [x, y, z, o_x, o_y, o_z, theta]; a pose without o_* is
straight down.
"""
import math

import numpy as np

from sorter import perception

CAM_BELOW_POSE_MM = 24.0  # {OBSERVE Z 400 "TABLE 376" / HIGH Z 580 "TABLE 557": THE TABLE TOP IS AT WORLD Z ~0}
# Camera axes in the tool frame (columns: image +u, image +v, viewing direction): straight down at theta 0, image +u is world
# -y and +v is world -x (perception), and Viam's straight-down tool frame is Ry(180): x and z flipped
_R_CT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _pose(p) -> dict:
    if not isinstance(p, dict):
        p = dict(zip(("x", "y", "z", "o_x", "o_y", "o_z", "theta"), [float(v) for v in p]))
    return {"x": p["x"], "y": p["y"], "z": p.get("z", 0.0), "o_x": p.get("o_x", 0.0), "o_y": p.get("o_y", 0.0),
            "o_z": p.get("o_z", -1.0), "theta": p.get("theta", 0.0)}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tool_rotation(p: dict) -> np.ndarray:
    """World-from-tool rotation for Viam's orientation vector (o_x, o_y, o_z, theta degrees)."""
    o = np.array([p["o_x"], p["o_y"], p["o_z"]], float)
    o /= np.linalg.norm(o)
    lat = math.acos(max(-1.0, min(1.0, o[2])))
    lon = math.atan2(o[1], o[0]) if 1.0 - abs(o[2]) > 1e-9 else 0.0
    return _rz(lon) @ _ry(lat) @ _rz(math.radians(p["theta"]))


# The camera's position in the tool frame, relative to the rotation centre: straight down at theta 0 the camera centre
# is at the jaw axis + JAW_IN_CAM (world (+48.1, +32.9)), the jaw axis is TOOL_AXIS_MM from the rotation centre, and the
# camera is CAM_BELOW_POSE_MM below the pose point. Straight-down tool frame = Ry(180): x and z flip.
_T_CAM = np.array([-(perception.JAW_IN_CAM_MM[1] + perception.TOOL_AXIS_MM[0]),
                   perception.JAW_IN_CAM_MM[0] + perception.TOOL_AXIS_MM[1], CAM_BELOW_POSE_MM])


def camera_to_world(pose_full, intr: dict, u, v, depth) -> np.ndarray:
    """World (x, y, z) mm of pixels (u, v) at camera distance depth (arrays broadcast; last axis of the result is xyz)."""
    p = _pose(pose_full)
    u, v, d = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(depth, float))
    cam = np.stack([(u - intr["cx"]) * d / intr["fx"], (v - intr["cy"]) * d / intr["fy"], d], axis=-1)
    R = tool_rotation(p)
    centre = np.array([p["x"] - perception.TOOL_AXIS_MM[0], p["y"] - perception.TOOL_AXIS_MM[1], p["z"]])
    return centre + (_T_CAM + cam @ _R_CT.T) @ R.T


def world_xy_full(pose_full, intr: dict, u: float, v: float, depth_mm: float) -> tuple:
    """perception.world_xy for any wrist pose: table (x, y) of the point seen at pixel (u, v) at camera distance depth_mm."""
    p = camera_to_world(pose_full, intr, u, v, depth_mm)
    return float(p[0]), float(p[1])


def visible_cells(depth: np.ndarray, pose_full, intr: dict, table_plane=None, cell_mm: float = 20.0, step: int = 4,
                  min_hits: int = 3) -> dict:
    """
    {"cell_mm", "table": [[ix, iy], ...], "object": [[ix, iy], ...]}: table cells (x // cell_mm, y // cell_mm) this frame saw.

    Every valid depth pixel (every `step` px) is a ray that ends on the first surface it meets: its world point marks that
    cell as seen, as table (within MIN_HEIGHT_MM of the table) or as an object top. A table cell behind a near object gets no
    ray, so it is not seen; nor is a cell whose depth is invalid (glare, the blind range). `table_plane` is unused: heights come
    from world z (the table is world z ~0 on sfhack4); kept for the caller's signature. A cell needs `min_hits` rays.
    """
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    d = depth[::step, ::step].astype(float)
    ok = (d > 0) & (d < 2000)
    pts = camera_to_world(pose_full, intr, uu[ok], vv[ok], d[ok])
    ix = np.floor(pts[:, 0] / cell_mm).astype(int)
    iy = np.floor(pts[:, 1] / cell_mm).astype(int)
    top = pts[:, 2] > perception.MIN_HEIGHT_MM
    out = {"cell_mm": cell_mm}
    for name, sel in (("table", ~top), ("object", top)):
        cells, counts = np.unique(np.stack([ix[sel], iy[sel]], axis=1), axis=0, return_counts=True) if sel.any() else (np.zeros((0, 2), int), np.zeros(0))
        out[name] = [[int(a), int(b)] for (a, b), n in zip(cells, counts) if n >= min_hits]
    return out
