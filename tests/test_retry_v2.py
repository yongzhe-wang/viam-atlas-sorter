"""SEARCH_V2 retries: a servo miss goes straight back to the same object; a clear refusal answers at once. No robot."""
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = str(Path(__file__).resolve().parent.parent)
os.environ["SORTER_DATA"] = tempfile.mkdtemp(prefix="retry_v2_")
os.environ.pop("MONGODB_URI", None)
os.environ.pop("VIAM_MODULE_DATA", None)
sys.path.insert(0, REPO)
sys.modules["torch"] = SimpleNamespace(set_num_threads=lambda n: None)
sys.modules["open_clip"] = SimpleNamespace()

import module.sorter_service as svc  # noqa: E402

XY = [505.0, -112.0]
SERVO = {"status": "failed", "fail_phase": "servo", "fail_reason": "RuntimeError: block: did not converge (8 iterations or 12 s)",
         "target_xy": XY, "margin": 1.0}
SLIP = {"status": "failed", "fail_phase": "carry", "fail_reason": "slipped out", "target_xy": XY, "margin": 1.0}
DONE = {"status": "done", "target_xy": XY}
NOT_FOUND = {"status": "not_found", "fail_reason": "no 'x' found"}


def service(outcomes: list) -> tuple:
    """A service whose _once plays `outcomes` in turn and records (target_xy, excluded) of every call."""
    s = object.__new__(svc.GraspByDescription)
    s.job_info, calls = {}, []
    s.in_bucket = SimpleNamespace(add=lambda *a, **k: None)
    s.store = SimpleNamespace(list_labels=lambda: [])
    s._search_v2 = lambda: True

    async def once(kind, text, target, target_xy, skip, any_match, excluded, stay=False):
        calls.append({"target_xy": target_xy, "excluded": list(excluded)})
        return dict(outcomes.pop(0)), "search"
    s._once = once
    return s, calls


async def main():
    svc.GRASP_ATTEMPTS = 3
    # Sour patch: the servo lost it -> the next try goes back to that spot with nothing left out, and picks it
    # {SOUR PATCH 14:04:14 "THE BEST MATCH FOR 'SOUR PATCH SNACK' IS THE OBJECT A FAILED TRY ALREADY WENT FOR"}
    s, calls = service([SERVO, DONE])
    r = await s._attempts("grasp", "sour patch snack")
    assert r["status"] == "done" and len(calls) == 2, (r, calls)
    assert calls[1]["target_xy"] == XY and calls[1]["excluded"] == [], calls
    print("1. a servo miss: the retry goes straight back to the same object (not excluded)")

    # Twice lost: the third try leaves it out; nothing else there -> the servo failure is the answer
    s, calls = service([SERVO, SERVO, NOT_FOUND])
    r = await s._attempts("grasp", "sour patch snack")
    assert calls[2]["excluded"] == [XY] and r["status"] == "failed" and "converge" in r["fail_reason"], (r, calls)
    print("2. lost twice: the third try searches the others; nothing else -> the servo failure stands")

    # A slip is still what it was: that object left out of the next search
    s, calls = service([SLIP, DONE])
    await s._attempts("grasp", "kind snack")
    assert calls[1]["excluded"] == [XY] and calls[1]["target_xy"] is None, calls
    print("3. a slip: unchanged, the next search leaves that object out")

    # Red box cutter: too close, and clearly the one named (margin 0.40) -> answered at once, no 40 s search
    # {RED BOX CUTTER 14:11:31 TOO_CLOSE MARGIN 0.401 -> 14:11:45 NOT_FOUND AFTER 40.8 S}
    s, calls = service([{"status": "too_close", "fail_reason": "too close to the arm base (r=223 mm)", "target_xy": XY,
                         "margin": 0.401}])
    r = await s._attempts("grasp", "red box cutter")
    assert r["status"] == "too_close" and len(calls) == 1, (r, calls)
    # ... but a close call between two objects still tries the other one
    s, calls = service([{"status": "too_close", "fail_reason": "too close", "target_xy": XY, "margin": 0.01}, DONE])
    r = await s._attempts("grasp", "screwdriver")
    assert r["status"] == "done" and calls[1]["excluded"] == [XY], (r, calls)
    print("4. a refusal of the clearly named object answers at once; of a near tie, the next candidate is tried")
    print("test_retry_v2 OK")


asyncio.run(main())
