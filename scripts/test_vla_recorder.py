"""
Offline end-to-end test of VLA recording: fake robot -> VLARecorder -> raw episode -> RLDS dicts (convert.py)
-> schema check; plus the TFDS builder's own code paths against a stub `tensorflow_datasets`.

No robot, network or TensorFlow needed. Run from the repo root: python scripts/test_vla_recorder.py
"""
import asyncio
import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rlds" / "sorter_vla"))
from sorter.vla_recorder import VLARecorder  # noqa: E402
import convert  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


class FakeArm:
    """Moves toward the commanded pose at 400 mm/s; reads are a little slow, like the real link."""

    def __init__(self):
        self.p = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}
        self.fail_every = 0
        self.reads = 0

    async def get_end_position(self):
        self.reads += 1
        await asyncio.sleep(0.005)
        if self.fail_every and self.reads % self.fail_every == 0:
            raise ConnectionError("link dropped")
        return SimpleNamespace(o_x=0.0, o_y=0.0, o_z=-1.0, **self.p)

    async def get_joint_positions(self):
        await asyncio.sleep(0.005)
        return SimpleNamespace(values=[self.p["x"] / 10, self.p["y"] / 10, self.p["z"] / 10, 0.0, 90.0, self.p["theta"]])


class FakeRig:
    def __init__(self):
        self.arm = FakeArm()
        self.cam = SimpleNamespace(get_images=self._images)
        self.theta = 0.0
        self.status = None
        self.fx, self.fy, self.cx, self.cy = 909.9, 909.4, 645.5, 365.4
        self.calls = []

    async def _images(self):
        await asyncio.sleep(0.01)
        return {"z": self.arm.p["z"]}

    async def goto(self, x, y, z, theta=None, settle=True):
        self.calls.append(("goto", x, y, z, theta))
        if theta is not None:
            self.theta = theta
        start = dict(self.arm.p)
        dist = math.dist((start["x"], start["y"], start["z"]), (x, y, z))
        n = max(1, int(dist / 400.0 / 0.02))
        for k in range(1, n + 1):
            f = k / n
            self.arm.p = {"x": start["x"] + f * (x - start["x"]), "y": start["y"] + f * (y - start["y"]),
                          "z": start["z"] + f * (z - start["z"]),
                          "theta": start["theta"] + f * (self.theta - start["theta"])}
            await asyncio.sleep(0.02)
        if settle:
            await asyncio.sleep(0.1)

    async def open(self):
        self.calls.append(("open",))
        await asyncio.sleep(0.05)

    async def grab(self):
        self.calls.append(("grab",))
        await asyncio.sleep(0.05)
        return True


def fake_decode(images):
    rgb = np.full((720, 1280, 3), 120, np.uint8)
    rgb[300:400, 600:700] = (220, 200, 30)  # a "block"
    depth = np.full((720, 1280), 378.0)
    depth[300:400, 600:700] = 346.0
    return rgb, depth


async def scripted_pick(rig):
    await rig.open()
    await rig.goto(300, 50, 400, theta=0.0)
    await rig.goto(300, 50, 400, theta=30.0)
    await rig.goto(300, 50, 250, settle=False)
    await rig.grab()
    await rig.goto(300, 50, 400)
    await rig.goto(380, -60, 400)
    await rig.goto(380, -60, 260, settle=False)
    await rig.open()
    await rig.goto(380, -60, 400)
    return {"status": "done", "success": True, "_id": "att1", "label": "yellow block"}


async def record_all(tmp: Path):
    rig = FakeRig()
    orig_goto = rig.goto
    rec = VLARecorder(rig, tmp, hz=20, session="s1", decode=fake_decode)
    rec.attach()
    check("attach wraps goto", rig.goto is not orig_goto)
    p1 = await rec.start_episode("put the yellow block in the blue square", kind="grasp", target_object="yellow block")
    doc = await scripted_pick(rig)
    await rec.end_episode(doc)
    # Second episode: flaky link (every 4th pose read fails) and an instant refusal (too short to keep)
    rig.arm.fail_every = 4
    p2 = await rec.start_episode("take everything out of the blue square", kind="reset")
    await rig.goto(300, 0, 400)
    await rig.goto(250, 0, 400)
    await rec.end_episode({"status": "done", "success": True})
    rig.arm.fail_every = 0
    p3 = await rec.start_episode("put the purple elephant in the blue square", kind="grasp")
    await rec.end_episode({"status": "not_found", "success": False, "fail_reason": "no purple elephant"})
    rec.detach()
    rec.close()
    check("detach restores goto", rig.goto == orig_goto)
    return p1, p2, p3


def test_raw(p1, p2, p3):
    meta = json.loads((p1 / "meta.json").read_text())
    steps = convert.load_steps(p1)
    check("meta has outcome and instruction", meta["success"] and meta["status"] == "done"
          and meta["language_instruction"].startswith("put the yellow block") and meta["attempt_id"] == "att1")
    check("meta has camera intrinsics", meta["camera_intrinsics_full_res"].get("fx") == 909.9)
    expect = meta["duration_s"] * 20
    check("about hz x duration steps", 0.6 * expect <= len(steps) <= 1.1 * expect, f"{len(steps)} vs ~{expect:.0f}")
    check("every step has its image file", all((p1 / s["image"]).exists() for s in steps))
    check("depth saved as 16-bit PNG", all((p1 / s["depth"]).exists() for s in steps))
    ts = [s["t"] for s in steps]
    check("steps in time order", ts == sorted(ts))
    g = [s["gripper_cmd"] for s in steps]
    check("gripper command goes 0 -> 1 -> 0", 1 in g and g[0] == 0 and g[-1] == 0)
    check("commanded target recorded", any(s["target"] and s["target"]["z"] == 250 for s in steps))
    check("theta=None keeps the rig's current theta",
          any(s["target"] and s["target"]["z"] == 250 and s["target"]["theta"] == 30.0 for s in steps))
    zs = [s["ee"]["z"] for s in steps]
    check("motion captured mid-move (intermediate z)", any(260 < z < 390 for z in zs))
    m2 = json.loads((p2 / "meta.json").read_text())
    check("flaky link -> gaps counted, episode kept", m2["gaps"] > 0 and m2["n_steps"] > 0, f"gaps={m2['gaps']}")
    m3 = json.loads((p3 / "meta.json").read_text())
    check("instant refusal recorded with its reason", m3["status"] == "not_found" and m3["fail_reason"])


def test_convert(tmp: Path, p1):
    for depth in (False, True):
        ex = convert.episode_to_rlds(p1, depth=depth)
        probs = convert.check(ex, convert.spec(depth))
        check(f"RLDS example matches spec (depth={depth})", not probs, "; ".join(probs[:3]))
    ex = convert.episode_to_rlds(p1)
    st = ex["steps"]
    check("is_first / is_last only at the ends", st[0]["is_first"] and st[-1]["is_last"]
          and not any(s["is_first"] for s in st[1:]) and not any(s["is_last"] for s in st[:-1]))
    check("reward 1 only on the last step of a success", st[-1]["reward"] == 1.0 and sum(s["reward"] for s in st) == 1.0)
    total = np.sum([s["action"][:3] for s in st], axis=0)
    span = st[-1]["observation"]["state"][:3] - st[0]["observation"]["state"][:3]
    check("summed delta actions = start-to-end motion", np.allclose(total, span, atol=1e-5), f"{total} vs {span}")
    check("units are metres (z ~0.4 at observe)", 0.3 < st[0]["observation"]["state"][2] < 0.5)
    yaw = [float(s["observation"]["state"][6]) for s in st]
    check("yaw in radians reaches 30 deg", abs(max(yaw) - math.radians(30)) < 1e-3)
    grip_act = [float(s["action"][6]) for s in st]
    grip_obs = [float(s["observation"]["state"][7]) for s in st]
    check("gripper action = next step's command", grip_act[:-1] == grip_obs[1:])
    check("image resized to 256x256", st[0]["observation"]["image"].shape == (256, 256, 3))
    check("instruction on every step", all(s["language_instruction"] == ex["steps"][0]["language_instruction"] for s in st))
    eps = convert.find_episodes(tmp)
    check("too-short episode filtered out", len(eps) == 2, f"{[e.name for e in eps]}")
    check("only_success filter keeps successes", len(convert.find_episodes(tmp, only_success=True)) == 2)
    check("kind filter", len(convert.find_episodes(tmp, kinds={"reset"})) == 1)
    check("default split: every session in train", convert.splits(tmp) == {"train": ["s1"]})


def test_builder_with_stub(tmp: Path):
    """Exercise the builder's own code (spec -> features, splits, generation) against a minimal fake TFDS."""
    made = []

    def feat(kind):
        def f(*a, **k):
            made.append(kind)
            return (kind, a, k)
        return f

    class GB:
        def __init__(self, name="rgb"):
            self.builder_config = SimpleNamespace(name=name)

        def dataset_info_from_configs(self, **k):
            return k

    tfds = types.ModuleType("tensorflow_datasets")
    tfds.features = SimpleNamespace(FeaturesDict=feat("dict"), Dataset=feat("dataset"), Text=feat("text"),
                                    Scalar=feat("scalar"), Image=feat("image"), Tensor=feat("tensor"))
    tfds.core = SimpleNamespace(GeneratorBasedBuilder=GB, Version=lambda v: v,
                                BuilderConfig=lambda **k: SimpleNamespace(**k), DatasetInfo=dict)
    sys.modules["tensorflow_datasets"] = tfds
    import os
    os.environ["SORTER_VLA_RAW"] = str(tmp)
    import importlib
    b = importlib.import_module("sorter_vla_dataset_builder")
    for name in ("rgb", "rgbd"):
        builder = b.Builder(name)
        info = builder._info()
        check(f"builder _info builds ({name})", "features" in info and "image" in made)
        gens = builder._split_generators(None)
        exs = list(gens["train"])
        check(f"builder yields 2 episodes ({name})", len(exs) == 2 and all("/" in k for k, _ in exs))
        probs = convert.check(exs[0][1], convert.spec(name == "rgbd"))
        check(f"builder examples match spec ({name})", not probs, "; ".join(probs[:2]))
    del sys.modules["tensorflow_datasets"]


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        p1, p2, p3 = asyncio.run(record_all(tmp))
        test_raw(p1, p2, p3)
        test_convert(tmp, p1)
        test_builder_with_stub(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASSED" if not FAILED else f"\n{len(FAILED)} FAILED")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
