"""
The drop square on the table from any view of it, whole or cut by the image border: centre, size, yaw, confidence.

WHY: the user moves the paper square around freely, so it is often only partly in view (080/082: its far tape edge
left the observe view) and may be turned. Every view gives the full rectangle from the tape edges it does see
(perception.find_drop_square's hull, frame-cut edges excluded) plus the interior size; views from several poses
add up side by side.
{VIAM CENTERAL 2026-10-29 USER: "THAT IS NOT ALLOWED, WE NEED TO BE ABLE TO MOVE THIS AROUND"; "SEVERAL PARTIAL VIEWS
 FROM DIFFERENT POSES COMBINE INTO THE FULL RECTANGLE"} [CONFIDENCE: CONFIRMED 85% — see tests/test_drop_square.py]

Model: the white interior is a rectangle with axes a = (cos yaw, sin yaw), b = (-sin yaw, cos yaw), yaw in [-45, 45)
(a is the axis nearer world x). Its sides a0 < a1 (across a) and b0 < b1 (across b) are kept as table points on
the tape edge ("anchors"), so the rectangle stays right when the yaw estimate improves. A side never seen is the
opposite side +- the size; the size along an axis is learned whenever both of its sides were seen.

Map entry (tmap["square"]): {xy (estimated full centre), size_mm: [along a, along b] (estimated interior), yaw_deg,
corners: 4 x [x, y], partial (not all four sides seen: look closer before placing), confidence (0.25 per side seen,
x0.8 while a one-sided axis uses an unmeasured size), sides: {a0|a1|b0|b1: [x, y]}, size_learned, size_seen: [bool,
bool], vis, since_ts (first view of this placement of the paper), seen_ts, misses}.
"""
import math

import numpy as np

from sorter import perception

# Interior along a / b before any view measured it. 24 full views (four tape edges): 113.4-115.0 x 174.0-177.7 mm,
# the same before and after the paper was moved and turned (yaw +7.9 -> -1.2 deg). The 137 x 188 read earlier was the
# axis-aligned box of the turned rectangle. {DROPSQ_EVAL "076 … SIZE=[114.2, 174.2] YAW=7.92" "084 … SIZE=[113.4, 175.9]
# YAW=-1.21"} [CONFIDENCE: CONFIRMED 90% — one paper; any other is learned from its first view of two opposite sides]
SIZE_MM = (114.5, 175.5)
YAW_TOL_DEG = 8.0  # a view's yaw this far from the map's: the paper was turned, start over
MOVED_MM = 20.0  # a seen side this far from the map's (or from opposite side + measured size): moved, start over
EDGE_TOL_DEG = 15.0  # a hull edge within this of an axis direction lies along a side
SIDE_MIN_MM = 20.0  # visible tape edge length that counts a side as seen
EXTREME_MM = 15.0  # ... and it must lie this close to the visible extreme on that side
ALPHA = 0.5  # weight of a new view in a side anchor / size
MARGIN_MM = 10.0  # an object this close to the interior counts as inside it
SIDES = ("a0", "a1", "b0", "b1")


def observe(hit: dict, pose: dict, intr: dict) -> dict:
    """
    find_drop_square() hit -> the observation record's square: the hole's hull in table xy and which hull edges are
    tape (real) rather than the image border. xy / size_mm / partial / cut describe the visible part (display only).
    """
    from sorter import camera_model  # (imports perception, which this module's callers import first)

    hull = [camera_model.world_xy_full(pose, intr, u, v, d) for u, v, d in hit["hull_px"]]
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    return {"hull_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in hull],
            "real": [not c for c in hit["hull_cut"]], "cut": list(hit.get("cut", [])), "partial": bool(hit.get("partial")),
            "xy": [round((min(xs) + max(xs)) / 2, 2), round((min(ys) + max(ys)) / 2, 2)],
            "size_mm": [round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)]}


def _wrap(deg: float, period: float) -> float:
    """deg into [-period/2, period/2)."""
    return (deg + period / 2) % period - period / 2


def _axes(yaw: float) -> tuple:
    t = math.radians(yaw)
    return np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def _edges(rec: dict) -> list:
    """[(midpoint, length mm, direction deg)] of one view's tape edges (real hull edges), in table coordinates."""
    h = np.asarray(rec["hull_xy"], float)
    out = []
    for k, real in enumerate(rec["real"]):
        if not real:
            continue
        p, q = h[k], h[(k + 1) % len(h)]
        L = float(np.hypot(*(q - p)))
        if L > 0:
            out.append(((p + q) / 2, L, math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))))
    return out


def view_yaw(rec: dict) -> tuple:
    """(yaw deg in [-45, 45), weight): length-weighted mean of the tape edge directions modulo 90 degrees."""
    z = sum(L * complex(math.cos(math.radians(4 * a)), math.sin(math.radians(4 * a))) for _, L, a in _edges(rec))
    if abs(z) < 1e-6:
        return None, 0.0
    return _wrap(math.degrees(math.atan2(z.imag, z.real)) / 4, 90.0), abs(z)


def view_sides(rec: dict, yaw: float) -> tuple:
    """
    ({side: anchor [x, y]} for the sides this view sees, visible extents (a_min, a_max, b_min, b_max)) at `yaw`.

    A tape edge along the a axis belongs to side b0 or b1, whichever visible extreme it lies at (within EXTREME_MM);
    edges in between (an object standing on the tape bulges the hull) and sides with < SIDE_MIN_MM of edge are dropped.
    """
    ax, bx = _axes(yaw)
    h = np.asarray(rec["hull_xy"], float)
    A, B = h @ ax, h @ bx
    ext = (float(A.min()), float(A.max()), float(B.min()), float(B.max()))
    acc = {s: [np.zeros(2), 0.0] for s in SIDES}
    for mid, L, ang in _edges(rec):
        rel = _wrap(ang - yaw, 180.0)
        if abs(rel) <= EDGE_TOL_DEG:  # runs along a: it is a b side
            pos, lo, hi, keys = float(mid @ bx), ext[2], ext[3], ("b0", "b1")
        elif abs(abs(rel) - 90.0) <= EDGE_TOL_DEG:  # runs along b: an a side
            pos, lo, hi, keys = float(mid @ ax), ext[0], ext[1], ("a0", "a1")
        else:
            continue
        d0, d1 = pos - lo, hi - pos
        if min(d0, d1) > EXTREME_MM:
            continue
        s = keys[0] if d0 <= d1 else keys[1]
        acc[s][0] = acc[s][0] + mid * L
        acc[s][1] += L
    return {s: (acc[s][0] / acc[s][1]).tolist() for s in SIDES if acc[s][1] >= SIDE_MIN_MM}, ext


def _fits_swapped(size: list, ext: tuple) -> bool:
    """The visible part only fits the interior with a and b sizes exchanged (the paper was turned ~90 degrees)."""
    ea, eb = ext[1] - ext[0], ext[3] - ext[2]
    fits = ea <= size[0] + MOVED_MM / 2 and eb <= size[1] + MOVED_MM / 2
    swapped = ea <= size[1] + MOVED_MM / 2 and eb <= size[0] + MOVED_MM / 2
    return not fits and swapped


def _moved(sq: dict, seen: dict) -> bool:
    """A side this view sees is far from where the map has it, or from its opposite side + the measured size."""
    ax, bx = _axes(sq["yaw_deg"])
    for s, p in seen.items():
        u = ax if s[0] == "a" else bx
        other = ("a1" if s == "a0" else "a0") if s[0] == "a" else ("b1" if s == "b0" else "b0")
        k = 0 if s[0] == "a" else 1
        if s in sq["sides"] and abs(np.dot(p, u) - np.dot(sq["sides"][s], u)) > MOVED_MM:
            return True
        if s not in sq["sides"] and other in sq["sides"] and sq["size_seen"][k]:
            want = np.dot(sq["sides"][other], u) + (sq["size_learned"][k] if s.endswith("1") else -sq["size_learned"][k])
            if abs(np.dot(p, u) - want) > MOVED_MM:
                return True
    return False


def update(sq: dict, rec: dict, now: float) -> dict:
    """
    Fold one view (observe() record) into the map's square entry; returns the new entry (sq None: no square yet).

    Same paper (yaw within YAW_TOL_DEG, every seen side within MOVED_MM): yaw and side anchors move by ALPHA towards
    the view, sizes are learned from axes seen on both sides. Otherwise the paper moved or turned: the entry restarts
    from this view, keeping the learned sizes (the same paper, maybe turned by 90 degrees: sizes swap if only that fits).
    """
    yaw_v, w_v = view_yaw(rec)
    if yaw_v is None:
        return sq  # no tape edge in this view: nothing to learn
    if sq is not None and abs(_wrap(yaw_v - sq["yaw_deg"], 90.0)) <= YAW_TOL_DEG:
        yaw = sq["yaw_deg"] + _wrap(yaw_v - sq["yaw_deg"], 90.0) * ALPHA
        seen, ext = view_sides(rec, yaw)
        trial = {**sq, "yaw_deg": yaw}
        restart = _moved(trial, seen)
    else:
        restart = True
    if restart:
        size = list(sq["size_learned"]) if sq else list(SIZE_MM)
        size_seen = list(sq["size_seen"]) if sq else [False, False]
        seen, ext = view_sides(rec, yaw_v)
        if _fits_swapped(size, ext):
            size, size_seen = size[::-1], size_seen[::-1]
        sq = {"yaw_deg": yaw_v, "sides": {}, "size_learned": size, "size_seen": size_seen, "since_ts": now}
        yaw = yaw_v
    sides = dict(sq["sides"])
    for s, p in seen.items():
        old = sides.get(s)
        sides[s] = list(p) if old is None else [(1 - ALPHA) * old[0] + ALPHA * p[0], (1 - ALPHA) * old[1] + ALPHA * p[1]]
    new = {**sq, "yaw_deg": yaw, "sides": sides, "seen_ts": now, "misses": 0,
           "vis": [round((ext[0] + ext[1]) / 2, 2), round((ext[2] + ext[3]) / 2, 2)]}
    # Size along an axis this view saw on both sides (one view: no paper motion in between)
    ax, bx = _axes(yaw)
    for k, (s0, s1, u) in enumerate((("a0", "a1", ax), ("b0", "b1", bx))):
        if s0 in seen and s1 in seen:
            m = float(np.dot(seen[s1], u) - np.dot(seen[s0], u))
            new["size_learned"] = list(new["size_learned"])
            new["size_learned"][k] = m if not new["size_seen"][k] else (1 - ALPHA) * new["size_learned"][k] + ALPHA * m
            new["size_seen"] = list(new["size_seen"])
            new["size_seen"][k] = True
    return derive(new)


def derive(sq: dict) -> dict:
    """Fill xy / extent / corners / partial / confidence from the sides, yaw and sizes (in place; returns it)."""
    ax, bx = _axes(sq["yaw_deg"])
    S, vis = sq["sides"], sq.get("vis") or [0.0, 0.0]
    centre, extent, conf = [], [], 1.0
    for k, (s0, s1, u) in enumerate((("a0", "a1", ax), ("b0", "b1", bx))):
        size = sq["size_learned"][k]
        if s0 in S and s1 in S:
            lo, hi = float(np.dot(S[s0], u)), float(np.dot(S[s1], u))
            centre.append((lo + hi) / 2)
            extent.append(hi - lo)
        elif s0 in S or s1 in S:
            edge = float(np.dot(S[s0] if s0 in S else S[s1], u))
            centre.append(edge + size / 2 if s0 in S else edge - size / 2)
            extent.append(size)
            conf *= 1.0 if sq["size_seen"][k] else 0.8
        else:
            centre.append(vis[k])  # no side across this axis seen yet: centred on what is visible
            extent.append(size)
    c = centre[0] * ax + centre[1] * bx
    ha, hb = ax * extent[0] / 2, bx * extent[1] / 2
    sq.update(xy=[round(float(c[0]), 1), round(float(c[1]), 1)], size_mm=[round(e, 1) for e in extent],
              corners=[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in (c - ha - hb, c + ha - hb, c + ha + hb, c - ha + hb)],
              partial=len(S) < 4, confidence=round(conf * len(S) / 4, 2), yaw_deg=round(float(sq["yaw_deg"]), 2))
    return sq


def contains(sq: dict, xy, margin: float = MARGIN_MM) -> bool:
    """Is table point xy inside the estimated full interior (partial or not), with margin?"""
    if not sq or "size_mm" not in sq or "sides" not in sq:
        return False
    ax, bx = _axes(sq["yaw_deg"])
    d = np.asarray(xy, float) - np.asarray(sq["xy"], float)
    return abs(float(d @ ax)) <= sq["size_mm"][0] / 2 + margin and abs(float(d @ bx)) <= sq["size_mm"][1] / 2 + margin


def radius(sq: dict) -> float:
    """Half the interior diagonal: the whole square is in a view that contains this circle."""
    return 0.5 * math.hypot(*sq["size_mm"])


def from_legacy(sq: dict) -> dict:
    """
    A square entry saved before this module ({xy, size_mm, partial}) as a model entry; None for a legacy partial one
    (its xy was only the visible part: the next view of the square rebuilds it).
    """
    if not sq or "sides" in sq:
        return sq
    if sq.get("partial") or not sq.get("xy") or not sq.get("size_mm"):
        return None
    (x, y), (A, B) = sq["xy"], sq["size_mm"]
    out = {"yaw_deg": 0.0, "size_learned": [float(A), float(B)], "size_seen": [True, True], "seen_ts": sq.get("seen_ts", 0.0),
           "since_ts": sq.get("seen_ts", 0.0),
           "misses": sq.get("misses", 0), "sides": {"a0": [x - A / 2, y], "a1": [x + A / 2, y], "b0": [x, y - B / 2],
                                                    "b1": [x, y + B / 2]}}
    return derive(out)
