"""
{"dashboard": {"since": ts}} on the module (module/dashboard_view.py) with a stubbed stack, no robot: one answer with
status, live map, the arm's pose, E-stop and phase; the picture only when newer, grabbed only while idle, an executor
frame during a job; everything Viam's Struct can carry. {VIAM CENTERAL 2026-10-30 "相机：持续刷新…任务进行中也不能停"}
"""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = str(Path(__file__).resolve().parent.parent)
os.environ["SORTER_DATA"] = tempfile.mkdtemp(prefix="dashboard_view_")
for k in ("MONGODB_URI", "VIAM_MODULE_DATA"):
    os.environ.pop(k, None)
sys.path.insert(0, REPO)
sys.modules["torch"] = SimpleNamespace(set_num_threads=lambda n: None)
sys.modules["open_clip"] = SimpleNamespace()
from viam.components.arm import Arm  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.app.robot import ComponentConfig  # noqa: E402
from viam.proto.common import Pose  # noqa: E402
from viam.utils import dict_to_struct  # noqa: E402

import module.sorter_service as svc  # noqa: E402
from sorter import perception  # noqa: E402
from sorter.worker import clean  # noqa: E402

H, W = 720, 1280
CUBE = SimpleNamespace(id=0, bbox=(300, 200, 380, 280), touches_edge=False, mask=np.zeros((H, W), bool),
                       crop=np.zeros((80, 80, 3), np.uint8), colour="green", height_mm=30.0)
GRABS, PHASE = [], ["observe"]
RT = SimpleNamespace(
    ppa=SimpleNamespace(HOME_XY=(250.0, 0.0), ALIGN_Z=400.0, TABLE_DEPTH_MM=378.0, ESTOP={"on": False},
                        decode=lambda imgs: (np.full((H, W, 3), 90, np.uint8), np.full((H, W), 500.0))),
    executor=SimpleNamespace(hole_mask=lambda rgb, depth=None: None, phase_now=lambda rig: PHASE[0]),
    clean=clean, perception=SimpleNamespace(segment_full=lambda rgb, d: SimpleNamespace(objects=[CUBE], intruders=[]),
                                            world_xy=perception.world_xy),
    clip_embed=SimpleNamespace(
        embed_text=lambda t: np.zeros(512), embed_images=lambda c: np.zeros((len(c), 512)),
        recognise_all=lambda rgb, depth, store=None, seg=None: [{"obj": CUBE, "label": "green square", "names": ["green square"],
                                                                "score": 0.9, "emb": None}]),
    table_map=SimpleNamespace(empty_map=lambda: {"ts": time.time(), "objects": [], "next_id": 0, "last_update": None},
                              update_map=lambda m, *a, **k: m or {"objects": []}))


class FakeArm:
    pose = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}

    async def get_end_position(self):
        return Pose(o_z=-1, **self.pose)


class FakeCam:
    async def get_images(self, **kw):
        GRABS.append(kw)
        return [SimpleNamespace(mime_type="image/jpeg", data=b"")], None


async def wait_for(cond, timeout=6.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return False


async def main():
    svc._runtime = lambda: RT
    svc.WATCH_PERIOD_S = 3600.0  # no idle loop in the way: only the looks this test makes
    arm = FakeArm()
    deps = {Arm.get_resource_name("arm"): arm, Gripper.get_resource_name("gripper"): SimpleNamespace(),
            Camera.get_resource_name("cam"): FakeCam()}
    s = svc.GraspByDescription.new(ComponentConfig(name="sorter", attributes=dict_to_struct({})), deps)
    s.watch_period = svc.WATCH_PERIOD_S
    rig = SimpleNamespace(fx=910.0, fy=910.0, cx=640.0, cy=360.0, status=None)

    async def get_rig():
        s.rig = rig
        return rig
    s._get_rig = get_rig
    assert await wait_for(lambda: s.warm), s.warm_error
    await s._fresh_look()  # one idle look: the picture with its recognised boxes

    r = await s.do_command({"dashboard": {"since": 0}})
    f = r["frame"]
    assert r["arm"] == arm.pose and r["estop"] is False and "phase" not in r and r["status"]["warm"] and "map" in r["map"]
    assert f["w"] == 640 and f["h"] == 360 and f["source"] == "idle" and len(f["jpeg_b64"]) > 1000
    assert [b["label"] for b in f["boxes"]] == ["green square"] and f["boxes"][0]["bbox"] == [150.0, 100.0, 190.0, 140.0]
    # The picture's corners on the table from the observe pose: perception.world_xy at the table's depth (378 mm)
    assert f["view"] == [[447.6, 298.7], [447.6, -232.9], [148.6, -232.9], [148.6, 298.7]], f["view"]
    dict_to_struct(r)  # what DoCommand sends: plain numbers, strings, lists, dicts
    print(f"1. one answer: status, map, arm, E-stop; the idle picture 640x360 with its box; view {f['view'][0]}..")

    n = len(GRABS)
    r = await s.do_command({"dashboard": {"since": f["ts"]}})
    assert r["frame"] == {"ts": f["ts"]} and len(GRABS) == n
    await asyncio.sleep(0.45)
    r = await s.do_command({"dashboard": {"since": f["ts"]}})
    g = r["frame"]
    assert len(GRABS) == n + 1 and GRABS[-1] == {"filter_source_names": ["color"]} and g["ts"] > f["ts"] and g["jpeg_b64"]
    assert [b["label"] for b in g["boxes"]] == ["green square"], g["boxes"]  # same pose: the idle look's boxes
    print("2. nothing newer: only its ts, no grab; 0.4 s later a colour grab, with the idle look's boxes (arm not moved)")

    arm.pose = {"x": 300.0, "y": 50.0, "z": 400.0, "theta": 10.0}
    await asyncio.sleep(0.45)
    r = await s.do_command({"dashboard": {"since": g["ts"]}})
    assert r["frame"]["boxes"] == [], r["frame"]["boxes"]
    print("3. the arm moved: a grab gets no boxes from the old view")

    s._busy = lambda: True
    s.job_info = {"job": "j", "kind": "grasp", "text": "tape", "search": {"k": 2, "n": 5}}
    PHASE[0] = "servo"
    n, ts = len(GRABS), r["frame"]["ts"]
    await asyncio.sleep(0.45)
    r = await s.do_command({"dashboard": {"since": ts}})
    assert len(GRABS) == n and r["frame"] == {"ts": ts} and r["phase"] == "servo" and r["map"]["search"]["k"] == 2
    s._on_map_frame(rig, np.full((H, W, 3), 30, np.uint8), None, {"x": 320.0, "y": 0.0, "z": 300.0, "theta": 0.0})
    r = await s.do_command({"dashboard": {"since": ts}})
    assert r["frame"]["source"] == "servo" and r["frame"]["pose"]["z"] == 300.0 and len(GRABS) == n
    print("4. during a job: no grab; every executor frame is the picture (source = its phase)")

    RT.ppa.ESTOP["on"] = True
    assert (await s.do_command({"dashboard": {}}))["estop"] is True
    print("5. E-stop in the answer")
    await s.close()


asyncio.run(main())
print("test_dashboard_view OK")
