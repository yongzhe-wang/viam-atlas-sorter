"""
Where the search has looked: the reachable table as 40 mm cells, each seen or not, and the next view chosen by how many unseen
cells it shows per second of moving there.

WHY: a fixed list of sweep poses missed objects wherever the layout put them out of those views or behind the bucket.
{USER 2026-10-30 "NO MATTER WHAT ARE THE LAYOUT OF THE OBJECTS IT WILL SEARCH FOR IT"}
{VIAM CENTERAL 2026-10-30 "每个候选按「新看到的未覆盖格子数 ÷ 移动时间」打分，贪心选下一个"}
[CONFIDENCE: SINGLE-SRC 70% — offline on recorded frames; straight-down views only: the camera model has no tilt]
A cell is seen when the frame shows the table or an object at it; behind the bucket, without depth, or out of the frame it is not.
Flow: executor.search_and_find (SEARCH_V2) -> Coverage.update() per frame -> next_view() -> the move -> look_here.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pick_place_auto as ppa  # noqa: E402

from sorter import perception  # noqa: E402

CELL_MM = 40.0
TABLE_X, TABLE_Y = (150.0, 600.0), (-440.0, 440.0)  # where the table is: the survey's free spots spanned x 240..535, y -342..430
CELL_R_MM = (205.0, 640.0)  # object positions that matter: from the base's own footprint out past the grasp reach
MARGIN_PX = 25.0  # a cell this close to the frame's border or the fingers is not counted as in view
SEEN_FRAC = 0.4  # of a cell's sampled pixels with table or object depth
VIEW_STEP_MM = {ppa.ALIGN_Z: 90.0, 580.0: 130.0}
MIN_GAIN = 3  # a view showing fewer new cells is not worth the move
MOVE_MM_S, MOVE_FIXED_S = 300.0, 0.7  # the time model: plan + settle, then speed (coordinator's measurements at 120 deg/s)


class Coverage:
    """The cells of the table an object could stand on, and which of them a frame has shown."""

    def __init__(self, k: dict):
        self.k = k
        xs = np.arange(TABLE_X[0] + CELL_MM / 2, TABLE_X[1], CELL_MM)
        ys = np.arange(TABLE_Y[0] + CELL_MM / 2, TABLE_Y[1], CELL_MM)
        cells = np.array([(x, y) for x in xs for y in ys])
        r = np.hypot(cells[:, 0], cells[:, 1])
        self.cells = cells[(r >= CELL_R_MM[0]) & (r <= CELL_R_MM[1])]
        self.seen = np.zeros(len(self.cells), bool)
        self.blocked = np.zeros(len(self.cells), bool)  # in view once but hidden (bucket, no depth)

    def pixels(self, pose: dict) -> tuple:
        """(u, v, in view) of every cell in a frame taken at `pose`, at the table's depth there."""
        d = ppa.TABLE_DEPTH_MM + (float(pose["z"]) - ppa.ALIGN_Z)
        uv = np.array([perception.project_xy(pose, self.k, tuple(c), d) for c in self.cells])
        u, v = uv[:, 0], uv[:, 1]
        inside = (u >= MARGIN_PX) & (u <= 1280 - MARGIN_PX) & (v >= MARGIN_PX) & (v <= ppa.FINGER_ROW - MARGIN_PX)
        return u, v, inside

    def update(self, depth: np.ndarray, pose: dict, hidden: np.ndarray = None) -> int:
        """Mark what this frame shows; `hidden` (bool image) is what hides the table (the bucket). Returns the newly seen count."""
        if depth is None:
            return 0
        u, v, inside = self.pixels(pose)
        a, b, c = perception.fit_table(depth.astype(float))
        H, W = depth.shape
        r = max(2, int(CELL_MM / 4 * self.k["fx"] / (ppa.TABLE_DEPTH_MM + float(pose["z"]) - ppa.ALIGN_Z)))
        new = 0
        for i in np.nonzero(inside)[0]:
            cu, cv = int(u[i]), int(v[i])
            v0, v1, u0, u1 = max(0, cv - r), min(H, cv + r + 1), max(0, cu - r), min(W, cu + r + 1)
            patch = depth[v0:v1:2, u0:u1:2].astype(float)
            vv, uu = np.mgrid[v0:v1:2, u0:u1:2]
            plane = a * uu + b * vv + c
            h = plane - patch
            ok = (patch > 0) & (h > -15.0) & (h < perception.MAX_OBJ_HEIGHT_MM)
            if hidden is not None:
                ok &= ~hidden[v0:v1:2, u0:u1:2]
            if ok.mean() >= SEEN_FRAC:
                new += int(not self.seen[i])
                self.seen[i] = True
            else:
                self.blocked[i] = True
        return new

    def gain(self, pose: dict) -> int:
        """Unseen cells a straight-down frame at `pose` would show."""
        return int((self.pixels(pose)[2] & ~self.seen).sum())

    def fraction(self) -> float:
        return float(self.seen.mean()) if len(self.seen) else 1.0

    def hidden_cells(self) -> int:
        """Cells that were in view but never seen: behind the bucket or without depth."""
        return int((self.blocked & ~self.seen).sum())


def views() -> list:
    """Candidate straight-down views (x, y, z): grids at ALIGN_Z and z 580 over what the arm reaches there."""
    out = []
    for z, step in VIEW_STEP_MM.items():
        for x in np.arange(150.0, 600.0 + 0.1, step):
            for y in np.arange(-450.0, 450.0 + 0.1, step):
                if ppa.reachable(x, y, z) and math.hypot(x, y) >= 150.0:
                    out.append((float(x), float(y), float(z)))
    return out


def move_s(a: tuple, b: tuple) -> float:
    """Rough seconds from flange (x, y, z) a to b."""
    return MOVE_FIXED_S + math.dist(a, b) / MOVE_MM_S


def next_view(cov: Coverage, at: tuple, done: list, candidates: list = None):
    """The view with the most unseen cells per second from `at`, not one of `done`; None when none shows MIN_GAIN new cells."""
    best = None
    for q in candidates if candidates is not None else views():
        if any(math.dist(q, d) < 1.0 for d in done):
            continue
        g = cov.gain({"x": q[0], "y": q[1], "z": q[2], "theta": 0.0})
        if g < MIN_GAIN:
            continue
        score = g / move_s(at, q)
        if best is None or score > best[0]:
            best = (score, q, g)
    return None if best is None else (best[1], best[2])
