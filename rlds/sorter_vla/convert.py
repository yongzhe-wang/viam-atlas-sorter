"""
Raw recorder episodes (sorter/vla_recorder.py) -> RLDS episode dicts, numpy only (no TensorFlow).

The TFDS builder next to this file declares its features from SPEC and yields what episode_to_rlds() returns,
so the conversion and the schema are tested here without TensorFlow (scripts/test_vla_recorder.py), and SPEC is
the single source of truth for both.

Conventions (Open X-Embodiment style; units: metres, radians):
  observation.image        wrist colour, resized (squashed) to IMAGE_HW, uint8
  observation.depth        wrist depth in mm, same size, uint16 (config "rgbd" only)
  observation.state        [x, y, z, o_x, o_y, o_z, yaw, gripper]: end effector in the arm base frame; o_* is Viam's
                           orientation vector (the tool axis; (0, 0, -1) = pointing straight down), yaw = Viam theta
                           in radians about that axis; gripper = commanded state (0 open, 1 closed)
  observation.joint_state  6 joint angles, radians
  action                   [dx, dy, dz, droll, dpitch, dyaw, gripper]: change of the end effector from this step to
                           the next (measured), and the gripper command in effect at the next step. This robot's
                           code only moves with the tool pointing down, so droll = dpitch = 0 and the rotation is yaw.
  action_abs               [x, y, z, yaw, gripper]: the commanded goto target in effect at this step (absolute),
                           or the current state when nothing was commanded yet
  reward                   1.0 on the last step of a successful episode, else 0.0
  is_first / is_last / is_terminal, discount = 1.0, language_instruction, timestamp (state read time, unix s)
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_HW = (256, 256)
MIN_STEPS = 5  # shorter episodes (an instant "not found") carry nothing to learn from

SPEC_VERSION = "1.0.0"


def spec(depth: bool = False) -> dict:
    """Feature spec: ("image", shape, dtype) | ("tensor", shape, dtype) | ("scalar", dtype) | ("text",)."""
    obs = {"image": ("image", IMAGE_HW + (3,), "uint8"),
           "state": ("tensor", (8,), "float32"),
           "joint_state": ("tensor", (6,), "float32")}
    if depth:
        obs["depth"] = ("image", IMAGE_HW + (1,), "uint16")
    return {
        "steps": {
            "observation": obs,
            "action": ("tensor", (7,), "float32"),
            "action_abs": ("tensor", (5,), "float32"),
            "discount": ("scalar", "float32"),
            "reward": ("scalar", "float32"),
            "is_first": ("scalar", "bool"),
            "is_last": ("scalar", "bool"),
            "is_terminal": ("scalar", "bool"),
            "language_instruction": ("text",),
            "timestamp": ("scalar", "float64"),
        },
        "episode_metadata": {
            "file_path": ("text",), "episode_id": ("text",), "session": ("text",), "kind": ("text",),
            "status": ("text",), "success": ("scalar", "bool"), "source": ("text",), "robot": ("text",),
            "hz": ("scalar", "float32"), "n_gaps": ("scalar", "int32"),
        },
    }


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _state(step: dict) -> np.ndarray:
    e = step["ee"]
    return np.array([e["x"] / 1000, e["y"] / 1000, e["z"] / 1000, e["o_x"], e["o_y"], e["o_z"],
                     math.radians(e["theta"]), float(step["gripper_cmd"])], np.float32)


def _image(path: Path, mode_hw: tuple, resample) -> np.ndarray:
    im = Image.open(path)
    return np.asarray(im.resize((mode_hw[1], mode_hw[0]), resample))


def load_steps(ep_dir: Path) -> list:
    """steps.jsonl in step order, keeping only steps whose image was written."""
    steps = [json.loads(l) for l in (ep_dir / "steps.jsonl").read_text().splitlines() if l.strip()]
    return sorted((s for s in steps if s.get("image")), key=lambda s: s["i"])


def keep_episode(meta: dict, n_steps: int, only_success: bool = False, kinds=None) -> bool:
    if n_steps < MIN_STEPS:
        return False
    if only_success and not meta.get("success"):
        return False
    return kinds is None or meta.get("kind") in kinds


def episode_to_rlds(ep_dir, depth: bool = False) -> dict:
    """One raw episode folder -> {"steps": [...], "episode_metadata": {...}} matching spec(depth)."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    steps = load_steps(ep_dir)
    n = len(steps)
    states = [_state(s) for s in steps]
    success = bool(meta.get("success"))
    out = []
    for k, s in enumerate(steps):
        st = states[k]
        if k + 1 < n:
            nx = states[k + 1]
            action = np.array([nx[0] - st[0], nx[1] - st[1], nx[2] - st[2], 0.0, 0.0, _wrap(nx[6] - st[6]), nx[7]],
                              np.float32)
        else:
            action = np.array([0, 0, 0, 0, 0, 0, st[7]], np.float32)
        tg = s.get("target")
        action_abs = (np.array([tg["x"] / 1000, tg["y"] / 1000, tg["z"] / 1000, math.radians(tg["theta"]), st[7]],
                               np.float32) if tg else np.array([st[0], st[1], st[2], st[6], st[7]], np.float32))
        j = s.get("joints") or []
        joints = np.zeros(6, np.float32)
        joints[:min(6, len(j))] = np.radians(np.asarray(j[:6], np.float32))
        obs = {"image": _image(ep_dir / s["image"], IMAGE_HW, Image.BILINEAR).astype(np.uint8),
               "state": st, "joint_state": joints}
        if depth:
            if s.get("depth"):
                d = _image(ep_dir / s["depth"], IMAGE_HW, Image.NEAREST).astype(np.uint16)
            else:
                d = np.zeros(IMAGE_HW, np.uint16)
            obs["depth"] = d.reshape(IMAGE_HW + (1,))
        last = k == n - 1
        out.append({"observation": obs, "action": action, "action_abs": action_abs,
                    "discount": np.float32(1.0), "reward": np.float32(1.0 if (last and success) else 0.0),
                    "is_first": k == 0, "is_last": last, "is_terminal": last,
                    "language_instruction": meta.get("language_instruction") or "",
                    "timestamp": np.float64(s.get("t_state") or s["t"])})
    md = {"file_path": str(ep_dir), "episode_id": meta.get("episode_id") or ep_dir.name,
          "session": meta.get("session") or ep_dir.parent.name, "kind": meta.get("kind") or "",
          "status": meta.get("status") or "", "success": success, "source": meta.get("source") or "",
          "robot": meta.get("robot") or "", "hz": np.float32(meta.get("hz") or 0.0),
          "n_gaps": np.int32(meta.get("gaps") or 0)}
    return {"steps": out, "episode_metadata": md}


def find_episodes(raw_root, sessions=None, only_success: bool = False, kinds=None) -> list:
    """Episode folders under raw_root/<session>/<episode>/ with a meta.json, filtered; sorted for determinism."""
    root = Path(raw_root)
    eps = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        ep = meta_path.parent
        if sessions is not None and ep.parent.name not in sessions:
            continue
        meta = json.loads(meta_path.read_text())
        if keep_episode(meta, len(load_steps(ep)), only_success, kinds):
            eps.append(ep)
    return eps


def splits(raw_root) -> dict:
    """{split: [session, ...]} from raw_root/splits.json, else every session in "train".

    Split by session, never by step or episode: neighbouring frames are near-duplicates, so a frame-level split
    leaks the test set into training.
    """
    root = Path(raw_root)
    f = root / "splits.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"train": sorted(p.name for p in root.iterdir() if p.is_dir())}


def check(example: dict, sp: dict) -> list:
    """Problems with an example against a spec (empty list = conforms). Used by the tests."""
    probs = []

    def one(path, value, s):
        kind = s[0]
        if kind == "text":
            if not isinstance(value, str):
                probs.append(f"{path}: expected str, got {type(value).__name__}")
        elif kind == "scalar":
            if np.ndim(value) != 0:
                probs.append(f"{path}: expected scalar")
            elif s[1] == "bool" and not isinstance(value, (bool, np.bool_)):
                probs.append(f"{path}: expected bool")
        else:
            a = np.asarray(value)
            if tuple(a.shape) != tuple(s[1]) or a.dtype != np.dtype(s[2]):
                probs.append(f"{path}: expected {s[1]} {s[2]}, got {a.shape} {a.dtype}")

    def walk(path, value, s):
        if isinstance(s, dict):
            if set(value) != set(s):
                probs.append(f"{path}: keys {sorted(value)} != {sorted(s)}")
            for k in s:
                if k in value:
                    walk(f"{path}.{k}", value[k], s[k])
        else:
            one(path, value, s)

    for i, st in enumerate(example["steps"]):
        walk(f"steps[{i}]", st, sp["steps"])
    walk("episode_metadata", example["episode_metadata"], sp["episode_metadata"])
    return probs
