"""
{"teach"} and learning from delivered grasps, no robot: real travel, world_xy and segment_full on a simulated arm and
camera (a 30 mm block on a flat table, drawn where the pose would see it); CLIP is a stub, Atlas is mongomock.
{VIAM CENTERAL 2026-10-30 "只移动 ARM 到 Z≥400 的 LOOK POSE，不碰物体，受 ESTOP 和 BUSY 限制"}
"""
import asyncio
import hashlib
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mongomock
import numpy as np
from PIL import Image

REPO = str(Path(__file__).resolve().parent.parent)
DATA = tempfile.mkdtemp(prefix="teach_")
os.environ["SORTER_DATA"] = DATA
for k in ("MONGODB_URI", "VIAM_MODULE_DATA", "SORTER_TEACH"):
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
from sorter import clip_embed, executor, perception  # noqa: E402
from sorter.store import LEARN_MAX_VIEWS, Store  # noqa: E402

ppa = executor.ppa
INTR = {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 360.0}
OBJ = [330.0, 40.0]  # where the block stands on the table
BLOCK = (230, 200, 30)


class World:
    """The arm's pose and what its camera sees there."""
    def __init__(self):
        self.x, self.y, self.z, self.theta = 250.0, 0.0, 400.0, 0.0
        self.moves, self.hand_from, self.on_move = [], None, None

    def frame(self):
        table = ppa.TABLE_DEPTH_MM + self.z - ppa.ALIGN_Z
        top = table - 30.0
        depth, rgb = np.full((720, 1280), table), np.full((720, 1280, 3), 200, np.uint8)
        # world_xy inverted at theta 0
        yc, xc = perception.JAW_IN_CAM_MM[1] - (OBJ[0] - self.x), perception.JAW_IN_CAM_MM[0] - (OBJ[1] - self.y)
        u, v = xc * INTR["fx"] / top + INTR["cx"], yc * INTR["fy"] / top + INTR["cy"]
        half = 40.0 * INTR["fx"] / top
        u0, u1, v0, v1 = [int(round(c)) for c in (u - half, u + half, v - half * 0.6, v + half * 0.6)]
        depth[max(v0, 0):max(v1, 0), max(u0, 0):max(u1, 0)] = top
        rgb[max(v0, 0):max(v1, 0), max(u0, 0):max(u1, 0)] = BLOCK
        return rgb, depth


W = World()


class FakeRig:
    fx, fy, cx, cy = INTR["fx"], INTR["fy"], INTR["cx"], INTR["cy"]
    slowed = False

    async def pose(self):
        return SimpleNamespace(x=W.x, y=W.y, z=W.z, theta=W.theta)

    async def goto(self, x, y, z, theta=None, settle=True, long_ok=False):
        W.x, W.y, W.z = x, y, z
        W.moves.append((round(x), round(y), round(z)))
        if W.on_move:
            W.on_move()

    async def rise(self, z, settle=True):
        await self.goto(W.x, W.y, z)

    async def frame(self):
        return W.frame()

    def record(self, *a, **k):
        pass

    async def lift(self):
        pass


class FakeArm:
    async def get_end_position(self):
        return Pose(x=W.x, y=W.y, z=W.z, o_z=-1, theta=W.theta)


class FakeCam:
    async def get_images(self, **kw):
        return [SimpleNamespace(mime_type="image/jpeg", data=b"")], None


def embed_images(crops):
    """A stable vector per crop and pose: this flat block's crops look alike, a real object's differ from another side."""
    out = []
    for c in crops:
        key = np.asarray(c).tobytes() + f"{W.x:.0f},{W.y:.0f},{W.z:.0f}".encode()
        seed = int(hashlib.md5(key).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=512)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


async def wait_for(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return False


async def main():
    clip_embed.embed_images = embed_images
    clip_embed.embed_text = lambda t: embed_images([np.frombuffer(t.encode(), np.uint8)])[0]
    ppa.decode = lambda r: W.frame()
    svc.WATCH_PERIOD_S = 3600.0
    deps = {Arm.get_resource_name("arm"): FakeArm(), Gripper.get_resource_name("gripper"): SimpleNamespace(),
            Camera.get_resource_name("cam"): FakeCam()}
    s = svc.GraspByDescription.new(ComponentConfig(name="sorter", attributes=dict_to_struct({})), deps)
    s._intrinsics_cache = None
    rig = FakeRig()

    async def get_rig():
        s.rig = rig
        return rig

    async def intrinsics():
        return dict(INTR)
    s._get_rig, s._intrinsics = get_rig, intrinsics
    assert await wait_for(lambda: s.warm), s.warm_error

    r = await s.do_command({"teach": "m&m bag"})
    assert r["accepted"] is False and "SORTER_TEACH" in r["error"] and not W.moves, r
    st = await s.do_command({"status": True})
    assert st["teach"]["enabled"] is False
    print("1. flag off: refused, the arm did not move")

    os.environ["SORTER_TEACH"] = "1"
    for bad in ("", "blue", "the thing"):
        assert (await s.do_command({"teach": bad}))["accepted"] is False, bad
    assert (await s.do_command({"teach": "m&m bag", "point": [0.01, 0.01]}))["status"] == "not_found"
    assert not W.moves
    print("2. no name, a colour for a name, a click on nothing: refused without moving")

    r = await s.do_command({"teach": "M&M bag", "wait": True})
    assert r["accepted"] and r["status"] == "done" and r["label"] == "m&m bag", r
    assert r["views_added"] == 4 and r["views_skipped"] == 0, r["frames"]
    assert [f["z"] for f in r["frames"]] == [400, 400, 400, 580], r["frames"]
    assert r["searchable_s"] is not None and r["top_score"] > 0.99, r
    assert min(z for _, _, z in W.moves) >= 400.0, W.moves
    assert (W.x, W.y, W.z) == (*ppa.HOME_XY, ppa.ALIGN_Z), "ends at the observe pose"
    assert ["m&m bag"] in s.store.list_labels()
    views = s.store._read_offline("views")
    assert {v["source"] for v in views} == {"teach"} and all(v["colour"] and v["height_mm"] > 20 for v in views), views[0]
    assert all(len(v["embedding"]) == 512 and "jpeg" in v for v in views)
    assert s.last["kind"] == "teach" and s.library["views"] == 4
    print(f"3. taught: 4 views (z {sorted({z for _, _, z in W.moves})}), arm back at observe, searchable, heights {[v['height_mm'] for v in views]}")

    W.moves.clear()
    r = await s.do_command({"teach": "m&m bag", "wait": True})
    assert r["views_added"] == 0 and r["views_skipped"] == 4 and all(f["skipped"] == "already known" for f in r["frames"]), r
    print("4. the same again: every view a repeat, none stored")

    r = await s.do_command({"teach": "chocolate", "alias_of": "m&m bag", "wait": True})
    assert r["label"] == "m&m bag" and r["alias_of"] == "m&m bag" and r["views_added"] == 1, r  # the view that holds the name
    assert ["m&m bag", "chocolate"] in s.store.list_labels(), s.store.list_labels()
    r = await s.do_command({"preview": "chocolate"})
    assert r["via"] != "error", r
    r = await s.do_command({"teach": "chocolate", "alias_of": "nothing we know", "wait": True})
    assert r["status"] == "failed" and "not in the library" in r["fail_reason"], r
    print("5. alias_of a known object goes to that label; of an unknown one refused")

    # EStop on the first move: no more moves, no return trip
    W.moves.clear()
    ppa_estop = ppa.ESTOP

    def press():
        ppa_estop["on"] = True
    W.on_move = press
    r = await s.do_command({"teach": "second thing", "wait": True})
    W.on_move = None
    assert r["status"] == "estop" and len(W.moves) <= 3, (r, W.moves)
    moved = len(W.moves)
    assert (await s.do_command({"teach": "third thing"}))["status"] == "estop"
    assert len(W.moves) == moved
    ppa_estop["on"] = False
    W.x, W.y, W.z = (*ppa.HOME_XY, ppa.ALIGN_Z)
    print(f"6. EStop mid-teach: stopped after {moved} move legs, no way back, next teach refused")

    # Busy
    release = asyncio.Event()

    async def slow_travel(*a, **k):
        await release.wait()
    with mock.patch.object(ppa, "travel", slow_travel):
        r = await s.do_command({"teach": "fourth thing"})
        assert r["accepted"]
        r2 = await s.do_command({"teach": "fifth thing"})
        assert r2["accepted"] is False and r2["error"] == "busy", r2
        release.set()
        await s.job
    print("7. busy: a second teach is refused while one runs")

    # A hand in view at a look pose: that frame is not kept
    W.x, W.y, W.z = (*ppa.HOME_XY, ppa.ALIGN_Z)
    real_seg = perception.segment_full

    def with_hand(rgb, depth, align=None):
        seg = real_seg(rgb, depth)
        return SimpleNamespace(objects=seg.objects, intruders=[{"bbox": (0, 0, 1, 1), "reason": "hand"}] if W.z > 450 else [],
                               excluded=seg.excluded)
    with mock.patch.object(perception, "segment_full", with_hand):
        r = await s.do_command({"teach": "sixth thing", "wait": True})
    assert any(f.get("skipped") == "hand in view" and f["z"] == 580 for f in r["frames"]), r["frames"]
    print("8. a hand in a look pose's picture: that view is left out")

    # Delivered grasps
    W.x, W.y, W.z = (*ppa.HOME_XY, ppa.ALIGN_Z)
    crop = np.full((80, 120, 3), 90, np.uint8)
    obj = SimpleNamespace(crop=crop, colour="red", height_mm=31.0, recognised="kind bar")
    outcome = {}

    async def execute(rig_, text, store, mode="demo", **kw):
        rig_.picked_obj = outcome.get("obj")
        return {"_id": "a1", "status": "done", "success": True, "label": "x", "margin": outcome.get("margin", 0.1),
                "grip": {"primitive": "top_down", "grip_depth": 12.0}, "duration_s": 0.1, **outcome.get("doc", {})}

    def unit(seed):
        v = np.random.default_rng(seed).normal(size=512)
        return (v / np.linalg.norm(v)).tolist()
    labels = [["kind bar", "snack bar"], ["granola bar"]]
    with mock.patch.object(executor, "execute", execute), mock.patch.object(s.store, "list_labels", lambda: labels):
        outcome.update(obj=obj, doc={"embedding": unit(1)})
        r = await s.do_command({"grasp": "snack", "wait": True})
        assert r["learned"]["label"] == "kind bar" and r["learned"]["view"], r
        v = next(x for x in s.store._read_offline("views") if x["_id"] == r["learned"]["view"])
        assert v["source"] == "grasp" and v["primitive"] == "top_down" and v["grip"]["grip_depth"] == 12.0 and v["attempt"] == "a1"
        assert await s.do_command({"grasp": "snack", "wait": True}) and s.last["learned"]["skipped"] == "already known"
        print("9. delivered: the crop becomes a view of the recognised name, with primitive and grip; a repeat is skipped")

        outcome["doc"] = {"embedding": unit(2)}
        r = await s.do_command({"grasp": "granola bar", "wait": True})
        assert "recognised 'kind bar'" in r["learned"]["skipped"], r
        n = len(s.store._read_offline("views"))
        obj.recognised = None
        r = await s.do_command({"grasp": "m&m", "wait": True})
        assert r["learned"]["label"] == "m&m" and r["learned"]["view"], r
        outcome.update(margin=0.0, doc={"embedding": unit(3)})
        r = await s.do_command({"grasp": "something new", "wait": True})
        assert "not clear" in r["learned"]["skipped"], r
        outcome.update(margin=0.1)
        r = await s.do_command({"grasp": "red", "wait": True})
        assert "does not name" in r["learned"]["skipped"], r
        assert len(s.store._read_offline("views")) == n + 1
        print("10. not taught: asked for one name and recognised another; a new name on an unclear pick; a colour as a name")

        obj.recognised = "kind bar"
        outcome.update(doc={"status": "failed", "success": False, "fail_reason": "slipped", "fail_phase": "check"})
        r = await s.do_command({"grasp": "kind bar", "wait": True})
        assert "learned" not in r, r
        outcome.update(obj=None, doc={"embedding": unit(4)})
        r = await s.do_command({"grasp": "kind bar", "wait": True})
        assert "learned" not in r, r
        print("11. a failed check or no crop: nothing learned")

        # The cap
        outcome.update(obj=obj)
        have = sum(v["label"] == "kind bar" for v in s.store._read_offline("views"))
        for k in range(have, LEARN_MAX_VIEWS):
            s.store.add_view("kind bar", unit(100 + k), None, {"source": "grasp"})
        outcome["doc"] = {"embedding": unit(5)}
        r = await s.do_command({"grasp": "kind bar", "wait": True})
        assert "already has 40" in r["learned"]["skipped"], r
        print("12. a label at 40 views takes no more from grasps")

    await s.close()

    # Atlas (mongomock): the library in memory, the panel, the free tier's limit
    def jpeg_of():
        b = io.BytesIO()
        Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(b, "JPEG")
        return b.getvalue()
    on = Store(uri="", data_dir=tempfile.mkdtemp())
    on.db = mongomock.MongoClient()["sorter"]
    on.add_view("tape", unit(10), jpeg_of(), {"source": "phone"})
    on.library_matrix()
    res = on.add_learned_view("m&m", unit(11), jpeg_of(), {"source": "teach"}, same_label_only=True)
    matrix, labels2 = on.library_matrix()
    assert res["id"] and matrix.shape == (2, 512) and labels2 == ["tape", "m&m"]
    # A new teach ignores the same picture under another label; a delivered grasp does not
    assert on.add_learned_view("glue", unit(11), jpeg_of(), {"source": "teach"}, same_label_only=True).get("id")
    assert on.add_learned_view("glue2", unit(11), jpeg_of(), {"source": "grasp"})["skipped"] == "already known"
    p = on.panel()
    assert p["views"] == 3 and p["labels"] == 3 and p["attempts"] == 0
    assert {e["label"]: e["new"] for e in p["learned"]} == {"m&m": True, "glue": True}, p
    on.add_view("tape", unit(12), jpeg_of(), {"source": "grasp"})
    assert {e["label"]: e["new"] for e in on.panel()["learned"]}["tape"] is False
    print("13. Atlas: new view in the library at once, counted in the panel; repeats by source rule")

    # Search answers from memory; $vectorSearch runs beside it for the panel; a teach asks Atlas itself
    hit = [{"_id": "x", "label": "m&m", "score": 0.97, "source": "teach"}]
    asked = []

    def aggregate(self, pipeline):
        asked.append(pipeline[0]["$vectorSearch"]["limit"])
        return list(hit)
    with mock.patch.object(type(on.db.views), "aggregate", aggregate):
        got = on.search_views(unit(11), k=3)
        assert got[0]["label"] == "m&m" and got[0]["score"] > 0.99 and "embedding" not in got[0], got[0]
        assert await wait_for(lambda: on.last_search is not None, 2.0)
        assert on.last_search["via"] == "$vectorSearch" and on.last_search["hits"][0]["score"] == 0.97 and asked == [3]
        assert on.atlas_search(unit(11), k=5) == hit and asked == [3, 5]
    print("14. search from memory, $vectorSearch in the background for the panel; atlas_search asks Atlas itself")

    # What this process learns is searchable and named at once, without reading the library from Atlas again
    reads = []
    real_find = type(on.db.views).find

    def find(self, *a, **k):
        reads.append(a[:1])
        return real_find(self, *a, **k)
    with mock.patch.object(type(on.db.views), "find", find):
        new = on.add_learned_view("kitkat", unit(14), jpeg_of(), {"source": "teach", "aliases": ["chocolate bar"]},
                                  same_label_only=True)
        assert on.search_views(unit(14), k=1)[0]["_id"] == new["id"]
        assert ["kitkat", "chocolate bar"] in on.list_labels()
        matrix, labels3 = on.library_matrix()
    assert reads == [], reads
    assert matrix.shape[0] == len(labels3) == on.db.views.count_documents({}) == 5, (matrix.shape, labels3)
    print("15. a learned view: searchable and named from memory at once, no read from Atlas")

    on._atlas_usage = (time.monotonic(), {"used_mb": 460.0, "limit_mb": 512})
    assert on.log_observation({"ts": time.time(), "source": "idle"}, jpeg_of()) and on.db.observations.count_documents({}) == 0
    assert on.add_view("after", unit(13), jpeg_of(), {"source": "teach"})
    print("16. near the free tier's 512 MB: observations stay on the disk, views still go up")


def unit(seed):
    v = np.random.default_rng(seed).normal(size=512)
    return (v / np.linalg.norm(v)).tolist()


asyncio.run(main())
print("test_teach OK")
