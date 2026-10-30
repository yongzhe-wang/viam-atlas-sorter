"""A list of objects: one job, item by item, same decision and retries; stubbed stack, no robot."""
import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = str(Path(__file__).resolve().parent.parent)
DATA = tempfile.mkdtemp(prefix="grasp_many_")
os.environ["SORTER_DATA"] = DATA
os.environ.pop("MONGODB_URI", None)
os.environ.pop("VIAM_MODULE_DATA", None)
sys.path.insert(0, REPO)
sys.modules["torch"] = SimpleNamespace(set_num_threads=lambda n: None)
sys.modules["open_clip"] = SimpleNamespace()
from viam.components.arm import Arm  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.app.robot import ComponentConfig  # noqa: E402
from viam.proto.common import Pose  # noqa: E402
from viam.utils import dict_to_struct, struct_to_dict  # noqa: E402

import module.sorter_service as svc  # noqa: E402
from module.grasp_policy import Delivered  # noqa: E402
from sorter.clip_embed import name_matches  # noqa: E402
from sorter.worker import clean  # noqa: E402

H, W = 720, 1280
CUBE = SimpleNamespace(id=0, bbox=(300, 200, 380, 280), touches_edge=False, mask=np.zeros((H, W), bool),
                       crop=np.zeros((80, 80, 3), np.uint8), colour="green", height_mm=30.0)

calls, OUTCOME, LOOKS = [], {}, [0]
DONE = {"status": "done", "duration_s": 0.05, "_id": "a"}


async def execute(rig, text, store, mode="demo", target=None, target_xy=None, progress=None, skip_current=False,
                  accept_ambiguous=False):
    calls.append({"text": text, "target": target is not None, "target_xy": target_xy, "skip": skip_current,
                  "any": accept_ambiguous})
    await asyncio.sleep(HOLD[0])
    out = OUTCOME.get(text, {**DONE, "label": text})
    if isinstance(out, list):  # one per try
        out = out.pop(0)
    if out.get("status") == "ambiguous" and accept_ambiguous:  # the real execute takes the best match instead
        return {**DONE, "label": text + " (best)"}
    return dict(out)


HOLD = [0.05]
MAP = {}  # label -> entry on the live map


def best_on_map(m, text, store=None):
    o = MAP.get(text)
    if o == "delivered":
        return {"obj": None, "map_obj": None, "xy": None, "ambiguous": False, "already_in_square": True,
                "reason": f"{text!r} is already in the square", "searchable": False}
    return {"obj": None, "map_obj": o, "xy": tuple(o["xy"]) if o else None, "ambiguous": bool(o and o.get("tie")),
            "score": 0.5 if o else 0.1, "searchable": o is None}


def best(objs, text, store=None, recognised=None):
    if text == "green square":
        return {"obj": recognised[0]["obj"], "score": 0.31, "margin": 0.1, "ambiguous": False, "recognised": text}
    if text == "screwdriver":  # a red and a blue one in view
        return {"obj": recognised[0]["obj"], "score": 0.3, "margin": 0.01, "ambiguous": True, "recognised": "red screwdriver"}
    return {"obj": None, "score": 0.1, "margin": 0.0, "ambiguous": False, "reason": "not in view", "searchable": True}


RT = SimpleNamespace(
    ppa=SimpleNamespace(HOME_XY=(250.0, 0.0), ALIGN_Z=400.0,
                        decode=lambda imgs: (np.zeros((H, W, 3), np.uint8), np.full((H, W), 500.0))),
    executor=SimpleNamespace(execute=execute, back_to_observe=None, clear_square=None, hole_mask=lambda rgb, depth=None: None),
    clean=clean, perception=SimpleNamespace(segment_full=lambda rgb, d: SimpleNamespace(objects=[CUBE], intruders=[])),
    clip_embed=SimpleNamespace(
        embed_text=lambda t: np.zeros(512), embed_images=lambda c: np.zeros((len(c), 512)), best=best, name_matches=name_matches,
        recognise_all=lambda rgb, depth, store=None, seg=None: [{"obj": CUBE, "label": "green square", "names": [], "score": 0.9, "emb": None}]),
    table_map=SimpleNamespace(empty_map=lambda: {"ts": time.time(), "objects": [], "next_id": 0, "last_update": None},
                              best_on_map=best_on_map, update_map=lambda m, *a, **k: m or {"objects": []}))


class FakeArm:
    async def get_end_position(self):
        return Pose(x=250, y=0, z=400, o_z=-1, theta=0.0)


class FakeCam:
    async def get_images(self, **kw):
        LOOKS[0] += 1
        return [SimpleNamespace(mime_type="image/jpeg", data=b"")], None


async def wait_for(cond, timeout=6.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return False


def st(r):
    return [i["status"] for i in r["items"]]


async def main():
    svc._runtime = lambda: RT
    svc.WATCH_PERIOD_S, svc.GRASP_ATTEMPTS = 3600.0, 2  # no idle loop in the way: only the looks a decision takes
    deps = {Arm.get_resource_name("arm"): FakeArm(), Gripper.get_resource_name("gripper"): SimpleNamespace(),
            Camera.get_resource_name("cam"): FakeCam()}
    s = svc.GraspByDescription.new(ComponentConfig(name="sorter", attributes=dict_to_struct({})), deps)
    RIG = SimpleNamespace(lift=lambda: asyncio.sleep(0), fx=900.0, fy=900.0, cx=640.0, cy=360.0)

    async def rig():
        s.rig = RIG
        return RIG
    s._get_rig = rig
    s.store.list_labels = lambda: [["blue screwdriver"], ["red screwdriver"], ["black tape", "tape"], ["kind bar", "snack bar"],
                                   ["blue and black screwdriver"]]
    assert await wait_for(lambda: s.warm)
    # Steps 1-10 test the list's mechanics on a table where every object stays: nothing goes into the bucket memory
    remember, s.in_bucket.add = s.in_bucket.add, lambda *a, **k: None

    # The single path is what it was
    for raw, text in (("grasp the green square", "the green square"), ("pumpkin", "pumpkin"), ("i want tape", "tape"),
                      ("blue and black screwdriver", "blue and black screwdriver")):
        calls.clear()
        r = await s.do_command({"grasp": raw, "wait": True})
        assert [c["text"] for c in calls] == [text] and s.last["kind"] == "grasp" and "items" not in r and not calls[0]["any"], (raw, calls, r)
    r = await s.do_command({"grasp": "screw driver"})  # spelled as the library does; two in view, alone: ambiguous as ever
    assert r["status"] == "ambiguous" and "'screwdriver'" in r["reason"], r
    print("1. one object: the single path, text as before (spoken lead-ins and 'screw driver' now understood)")

    calls.clear()
    OUTCOME["pumpkin"] = {"status": "not_found", "fail_phase": "find", "fail_reason": "no 'pumpkin' after 5 views", "_id": "a"}
    r = await s.do_command({"grasp": "i want tape, and pumpkin, and snack", "wait": True})
    assert [c["text"] for c in calls] == ["tape", "pumpkin", "snack"] and st(r) == ["done", "not_found", "done"], (calls, r)
    assert r["status"] == "partial" and r["done"] == 2 and r["total"] == 3 and "pumpkin: not_found" in r["fail_reason"], r
    assert s.last["kind"] == "grasp_many" and s.last["status"] == "partial" and r["label"] == "tape, snack", s.last
    i = r["items"][1]
    assert i["text"] == "pumpkin" and i["fail_reason"] == "no 'pumpkin' after 5 views" and r["items"][0]["label"] == "tape"
    print(f"2. mid-list not_found -> next item still done: {st(r)}, status {r['status']!r}")

    OUTCOME.clear()
    r = await s.do_command({"grasp": "tape and snack", "wait": True})
    assert r["status"] == "done" and r["done"] == 2 and "fail_reason" not in r, r
    for t in ("tape", "snack"):
        OUTCOME[t] = {"status": "not_found", "fail_reason": "nope", "_id": "a"}
    r = await s.do_command({"grasp": "tape and snack", "wait": True})
    assert r["status"] == "failed" and r["done"] == 0, r
    OUTCOME.clear()
    print("3. all done -> done; none -> failed")

    for name, bad, mark in (("E-stop", {"status": "estop", "fail_phase": "find", "fail_reason": "E-STOP pressed"}, "E-stop"),
                            ("link", {"status": "failed", "fail_phase": "carry", "fail_reason": "RuntimeError: move: link still down after 4 reconnects"}, "link"),
                            ("overcurrent", {"status": "failed", "fail_phase": "descend", "fail_reason": "MoveRefused: move refused: collision caused overcurrent"}, "overcurrent"),
                            ("holding", {"status": "failed", "fail_phase": "carry", "fail_reason": "x | recovery failed: RuntimeError: still holding: pick spot unknown"}, "holding"),
                            ("error", {"status": "error", "error": "RuntimeError: boom"}, "unexpected")):
        calls.clear()
        OUTCOME["snack"] = bad
        r = await s.do_command({"grasp": "tape, snack, mouse", "wait": True})
        assert [c["text"] for c in calls] == ["tape", "snack"], (name, calls)  # the third is never tried
        assert st(r) == ["done", bad["status"], "skipped"] and r["status"] == "partial" and mark in r["stopped_by"], (name, r)
        assert "not tried" in r["items"][2]["fail_reason"], r
    OUTCOME["tape"] = {"status": "estop", "fail_reason": "E-STOP pressed"}
    calls.clear()
    r = await s.do_command({"grasp": "tape, snack, mouse", "wait": True})
    assert r["status"] == "failed" and len(calls) == 1 and st(r) == ["estop", "skipped", "skipped"], r
    OUTCOME.clear()
    print("4. E-stop / dead link / overcurrent / still holding / unexpected error -> stops at once, the rest skipped")

    for bad in ("too_close", "too_wide", "square_full", "unsafe_grasp"):
        OUTCOME["snack"] = {"status": bad, "fail_phase": "find", "fail_reason": f"{bad} reason", "_id": "a"}
        calls.clear()
        r = await s.do_command({"grasp": "tape, snack, mouse", "wait": True})
        assert st(r) == ["done", bad, "done"] and len(calls) == 3 and "stopped_by" not in r, r
    OUTCOME["snack"] = {"status": "failed", "fail_phase": "carry", "fail_reason": "slipped out", "_id": "a", "duration_s": 0.1}
    calls.clear()
    r = await s.do_command({"grasp": "tape, snack, mouse", "wait": True})
    assert [c["text"] for c in calls] == ["tape", "snack", "snack", "mouse"] and st(r) == ["done", "failed", "done"], (calls, r)
    assert r["items"][1]["tries"] == 2 and "stopped_by" not in r, r
    OUTCOME.clear()
    print("5. too_close / too_wide / square_full / unsafe_grasp / failed -> next item; a slip is retried as a single grasp would be")

    # Several objects fit one name: refused alone, the best one in a list
    r = await s.do_command({"grasp": "screwdriver"})
    assert r["status"] == "ambiguous" and not s._busy(), r
    calls.clear()
    r = await s.do_command({"grasp": "screwdriver and tape", "wait": True})
    assert st(r) == ["done", "done"] and calls[0]["target"] and calls[0]["any"] and calls[1]["any"], (calls, r)
    MAP["tied"] = {"label": "kind bar", "xy": [300.0, 100.0], "colour": "x", "seen_ts": time.time(), "tie": True}
    s.live.map = {"ts": time.time(), "objects": [MAP["tied"]]}
    r = await s.do_command({"grasp": "tied"})
    assert r["via"] == "search", r  # alone: the map's tie is no answer, the search decides
    await s.job
    calls.clear()
    r = await s.do_command({"grasp": "tied, tape", "wait": True})
    assert calls[0]["target_xy"] == (300.0, 100.0) and st(r) == ["done", "done"], (calls, r)
    OUTCOME["tape"] = {"status": "ambiguous", "fail_reason": "two match", "_id": "a"}
    r = await s.do_command({"grasp": "snack, tape", "wait": True})
    assert st(r) == ["done", "done"], r  # the search path too: execute(accept_ambiguous=True)
    OUTCOME.clear()
    print("6. several objects fit a name: alone -> ambiguous as before; in a list -> best match (frame, map and search paths)")

    MAP["delivered"] = "delivered"
    r = await s.do_command({"grasp": "delivered, tape", "wait": True})
    assert st(r) == ["already_in_square", "done"] and r["status"] == "done" and not [c for c in calls if c["text"] == "delivered"], r
    print("7. already in the square counts as satisfied, nothing moved for it")

    # Progress while it runs; `last` only at the end; busy; looks between items
    HOLD[0] = 0.4
    calls.clear()
    s.last = {}
    L0 = LOOKS[0]
    r = await s.do_command({"grasp": "tape, snack, mouse"})
    assert r["accepted"] and r["via"] == "list" and [i["text"] for i in r["items"]] == ["tape", "snack", "mouse"], r
    assert (await s.do_command({"grasp": "x and y"}))["error"] == "busy"
    assert (await s.do_command({"reset": True}))["error"] == "busy"
    assert await wait_for(lambda: len(calls) == 2)
    status = await s.do_command({"status": True})
    run = status["running"]
    assert run["kind"] == "grasp_many" and run["progress"] == "item 2/3: snack" and run["item"]["k"] == 2, run
    assert [i["status"] for i in run["items"]] == ["done", "running", "pending"] and not s.last, (run, s.last)
    back = struct_to_dict(dict_to_struct(status))  # as the dashboard gets it
    assert back["running"]["items"][0]["text"] == "tape"
    await s.job
    HOLD[0] = 0.05
    assert LOOKS[0] - L0 >= 3 and s._between_items is False and s.last["job"] == r["job"], (LOOKS[0] - L0, s.last)
    print(f"8. running: {run['progress']!r}, items {[i['status'] for i in run['items']]}; busy refused; {LOOKS[0] - L0} looks "
          f"from the observe pose (one before each item); last only at the end")

    calls.clear()
    MAP["remembered"] = {"label": "blue block", "xy": [410.0, 40.0], "colour": "blue", "seen_ts": time.time()}
    s.live.map = {"ts": time.time(), "objects": [MAP["remembered"]]}
    p = await s.do_command({"preview": "green square, screwdriver, remembered, delivered, pumpkin"})
    got = {i["text"]: (i.get("via"), i.get("status")) for i in p["items"]}
    assert p["via"] == "list" and not calls and not s._busy(), (p, calls)
    assert got == {"green square": ("frame", None), "screwdriver": ("frame", None), "remembered": ("map", None),
                   "delivered": (None, "already_in_square"), "pumpkin": ("search", "would_search")}, got
    p1 = await s.do_command({"preview": "green square"})
    assert p1["via"] == "frame" and "items" not in p1
    print(f"9. preview of a list: each item's answer, nothing moved: {got}")

    calls.clear()
    r = await s.do_command({"grasp": "a1, a2, a3, a4, a5, a6, a7", "wait": True})
    assert len(calls) == 5 and r["ignored"] == ["a6", "a7"], r
    r = await s.do_command({"grasp": "a1 and a1 and , and"})
    assert "items" not in r
    await s.job
    print("10. more than 5 -> the first 5, the rest reported as ignored; repeats and empty parts dropped")

    # Into the bucket: the bucket is cut out of the picture, so the object leaves the map; asked again -> answered at once
    # {VIAM CENTERAL 2026-10-30 "已经投进桶、并且 MAP 上没再看到它 → 回答 ALREADY_IN_BUCKET，算 DONE，不动 ARM"}
    s.in_bucket.add = remember
    s.live.map = {"ts": time.time(), "objects": [MAP["remembered"]]}
    calls.clear()
    r = await s.do_command({"grasp": "tape", "wait": True})
    assert r["status"] == "done" and len(calls) == 1
    for ask in ("tape", "the black tape"):
        r = await s.do_command({"grasp": ask})
        assert r["status"] == "already_in_bucket" and not r["accepted"] and len(calls) == 1 and "'tape' went into the bucket" in r["reason"], r
    r = await s.do_command({"grasp": "tape and snack", "wait": True})
    assert st(r) == ["already_in_bucket", "done"] and r["status"] == "done" and [c["text"] for c in calls] == ["tape", "snack"], (r, calls)
    p = await s.do_command({"preview": "tape"})
    assert p["status"] == "already_in_bucket", p
    assert [e["label"] for e in (await s.do_command({"status": True}))["bucket"]] == ["tape", "snack"]
    print("11. delivered this run, not seen since: already_in_bucket at once (alone, in a list, preview), counts as done; status lists it")

    # Someone takes the tape out: the map sees it after it went in -> grasped as usual, and no longer listed
    MAP["tape"] = {"label": "tape", "names": ["tape", "black tape"], "xy": [350.0, -60.0], "colour": "black", "seen_ts": time.time() + 1}
    s.live.map = {"ts": time.time(), "objects": [MAP["remembered"], MAP["tape"]]}
    calls.clear()
    r = await s.do_command({"grasp": "tape"})
    assert r["accepted"] and r["via"] == "map", r
    await s.job
    assert len(calls) == 1
    print("12. seen on the table after it went in (taken out): grasped as usual")
    await s.close()

    # The rule itself: another object of the same name on the table, a reset
    d = Delivered(name_matches)
    d.add("red screwdriver", "screwdriver", [["red screwdriver", "flathead screwdriver"]])
    old_blue = {"label": "blue screwdriver", "names": ["blue screwdriver"], "seen_ts": time.time() - 60}
    assert d.check("screwdriver", {"objects": [old_blue]}) is None  # another screwdriver is on the table
    assert d.check("red screwdriver", {"objects": [old_blue]})["label"] == "red screwdriver"
    assert d.check("flathead screwdriver", {"objects": []})["label"] == "red screwdriver"  # the library's other name
    old_red = {"label": "red screwdriver", "names": [], "seen_ts": time.time() - 60}
    assert d.check("red screwdriver", {"objects": [old_red]}) is not None  # its spot from before it went in
    assert d.check("red screwdriver", {"objects": [{**old_red, "superseded": True, "seen_ts": time.time() + 5}]}) is not None
    assert [e["label"] for e in d.listing({"objects": [old_red]})] == ["red screwdriver"]
    d.clear()
    assert d.check("red screwdriver", {"objects": []}) is None and d.listing(None) == []
    print("13. another object of the name on the table -> not in the bucket; old spots do not count; a reset forgets")


asyncio.run(main())
