"""
Self-calibration of the jaw-axis pixel from released blocks.

WHY: the servo moves the arm until the target sits on the pixel where the jaw axis is *predicted* to
appear, so a grasp alone can never expose an error in that prediction (the arm goes exactly where
vision says). After a release the block sits on the real jaw axis, so looking at it again from the
servo heights measures where that axis actually projects. Each place gives two samples (two depths);
the overnight loop turns that into hundreds.

Limit: the jaws centre the block only along their closing direction (image u), so only that
component of the error is observable. That is the component that put the fingers on the block edge.

Model (pinhole, one per image axis): pixel = a + k / depth.
  hand-measured default: a = principal point, k = (JAW_TIP_PX - principal point) * TIP_BELOW_CAM
  k / focal length = lateral offset (mm) of the jaw axis from the camera axis, in camera coordinates

Files (gitignored, machine specific): data/jaw_samples.jsonl, data/jaw_calib.json
CLI: python scripts/jaw_calib.py fit   |   python scripts/jaw_calib.py show
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLES = DATA / "jaw_samples.jsonl"
CALIB = DATA / "jaw_calib.json"
MIN_SAMPLES = 6
# Below this spread in 1/depth the depths are too alike to separate a from k; then only a is refit
MIN_INV_SPREAD = 0.0004
# Residual floor for outlier rejection, so a tiny MAD on clean data does not discard good samples
OUTLIER_FLOOR_PX = 8.0
CAM_YAW_DEG = -97.731173  # camera rotation about the tool axis, from the Viam frame config


def record(depth: float, obs, model, intr, **meta) -> None:
    """
    Append one sample: block seen at pixel `obs` at block-top `depth` mm.

    `model` = (a, k) in use when recorded; `intr` = (fx, fy) so offsets can be converted to mm later.
    """
    DATA.mkdir(exist_ok=True)
    row = {"t": time.time(), "depth": float(depth), "obs": [float(obs[0]), float(obs[1])],
           "model": [list(map(float, model[0])), list(map(float, model[1]))],
           "intr": [float(intr[0]), float(intr[1])], **meta}
    with SAMPLES.open("a") as f:
        f.write(json.dumps(row) + "\n")


def load():
    """Fitted (a, k) from data/jaw_calib.json, or None when no fit exists yet."""
    if not CALIB.exists():
        return None
    c = json.loads(CALIB.read_text())
    return tuple(c["a"]), tuple(c["k"])


def load_samples() -> list:
    if not SAMPLES.exists():
        return []
    return [json.loads(line) for line in SAMPLES.read_text().splitlines() if line.strip()]


def fit(samples: list) -> dict:
    """
    Least-squares (a, k) per image axis, with iterative outlier rejection.

    Before: the model stored with the newest sample (hand-measured or a previous fit).
    After: the model that best explains where released blocks were actually seen.
    A sample that moved on release (bounced, tipped) shows up as a large residual and is dropped.
    """
    a0, k0 = samples[-1]["model"]
    inv = np.array([1.0 / s["depth"] for s in samples])
    obs = np.array([s["obs"] for s in samples], float)
    keep = np.ones(len(samples), bool)
    for _ in range(4):
        a, k = [], []
        for ax in range(2):
            if np.ptp(inv[keep]) >= MIN_INV_SPREAD:
                A = np.vstack([np.ones(int(keep.sum())), inv[keep]]).T
                (ai, ki), *_ = np.linalg.lstsq(A, obs[keep, ax], rcond=None)
            else:
                # Depths too similar: keep the old scale, only shift the offset
                ki = k0[ax]
                ai = float(np.mean(obs[keep, ax] - ki * inv[keep]))
            a.append(float(ai))
            k.append(float(ki))
        res = np.linalg.norm(obs - (np.array(a) + np.outer(inv, k)), axis=1)
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        new_keep = res < max(np.median(res[keep]) + 3 * 1.4826 * mad, OUTLIER_FLOOR_PX)
        if (new_keep == keep).all() or new_keep.sum() < MIN_SAMPLES:
            break
        keep = new_keep
    before = np.linalg.norm(obs[keep] - (np.array(a0) + np.outer(inv[keep], k0)), axis=1)
    return {"a": a, "k": k, "n_used": int(keep.sum()), "n_total": len(samples),
            "rmse_before_px": float(np.sqrt(np.mean(before ** 2))),
            "rmse_after_px": float(np.sqrt(np.mean(res[keep] ** 2))),
            "fitted_scale": bool(np.ptp(inv[keep]) >= MIN_INV_SPREAD)}


def cam_translation(k, intr) -> tuple:
    """
    Camera origin in the flange frame (x, y mm) implied by k, assuming the jaw axis is the flange axis.

    WHY: this is the value to type into the cam frame in the Viam app, so Viam's own frame system
    (motion service, detections-to-segments, 3D scene) gets the same correction as our servo.
    """
    X, Y = k[0] / intr[0], k[1] / intr[1]  # jaw axis relative to the camera, camera coordinates (mm)
    t = math.radians(CAM_YAW_DEG)
    xc, yc = (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))
    return -(X * xc[0] + Y * yc[0]), -(X * xc[1] + Y * yc[1])


def main(cmd: str) -> None:
    samples = load_samples()
    if cmd == "show":
        print(f"{len(samples)} samples in {SAMPLES}")
        print("current fit:", json.loads(CALIB.read_text()) if CALIB.exists() else None)
        return
    if len(samples) < MIN_SAMPLES:
        sys.exit(f"need >= {MIN_SAMPLES} samples, have {len(samples)} (each pick-and-place adds 2)")
    r = fit(samples)
    intr = samples[-1]["intr"]
    r["cam_translation_xy_mm"] = cam_translation(r["k"], intr)
    CALIB.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    print(f"wrote {CALIB}; pick_place_auto.py uses it on the next run")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fit")
