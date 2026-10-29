"""
Regression tests for sorter.table_map: the dynamic map accepts any starting map, and a replay equals the live map.

Synthetic frames (flat table + one 30 mm block) and hand-made recognitions, so no CLIP model, no Store and no
data/scan frames are needed. Run: python tests/test_table_map.py (or pytest).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sorter import perception, table_map as TM  # noqa: E402

POSE = {"x": 250.0, "y": 0.0, "z": 400.0, "theta": 0.0}


def frame(u0: int = 600, v0: int = 300):
    """1280x720 frame: wood-coloured table at 380 mm, a green 100x100 px block 30 mm tall at (u0, v0)."""
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:] = (180, 140, 100)
    depth = np.full((720, 1280), 380, np.uint16)
    rgb[v0:v0 + 100, u0:u0 + 100] = (40, 180, 60)
    depth[v0:v0 + 100, u0:u0 + 100] = 350
    return rgb, depth


def recognised(rgb, depth, label="green cube"):
    """segment_full() plus a fixed recognition per object, standing in for clip_embed.recognise_all()."""
    seg = perception.segment_full(rgb, depth)
    emb = np.ones(TM.EMB_DIM, np.float32) / np.sqrt(TM.EMB_DIM)
    recs = [{"obj": o, "label": label, "names": [label, "green square"], "score": 0.9, "emb": emb, "partial": False}
            for o in seg.objects]
    return recs, seg


def update(tmap, u0=600, now=1000.0):
    rgb, depth = frame(u0)
    recs, seg = recognised(rgb, depth)
    return TM.update_map(tmap, rgb, depth, POSE, recognised=recs, seg=seg, now=now)


def _core(m):
    """The map without the fields that legitimately differ (wall-clock ts, timings, the logged record)."""
    m = copy.deepcopy(m)
    m.pop("ts", None)
    lu = m.get("last_update") or {}
    lu.pop("ms", None)
    lu.pop("observation", None)
    return json.loads(json.dumps(m))


def test_starting_maps():
    # {VIAM CENTERAL 2026-10-29 "UPDATE_MAP(EMPTY_MAP(), …) AND UPDATE_MAP({}, …) BOTH WORK"}
    # [CONFIDENCE: CONFIRMED 95% — the KeyError: 'next_id' on sfhack4 came from a {"objects": []} start]
    cleared = {"ts": 1.0, "objects": [], "frames": [], "cleared_ts": 1.0, "cleared_reason": "test", "cleared_objects": 3}
    for start in (TM.empty_map(), {}, {"objects": []}, cleared, None):
        m = update(copy.deepcopy(start))
        assert [o["label"] for o in m["objects"]] == ["green cube"], (start, m["objects"])
        assert m["next_id"] == 1 and m["objects"][0]["id"] == 0 and m["objects"][0]["n_seen"] == 1


def test_survey_style_map():
    # A map_from_frames() / stored map: entries without n_seen, misses, emb or partial; ids continue after the max
    old = {"objects": [{"id": 7, "label": "green cube", "names": ["green cube"], "xy": None, "height_mm": 30.0},
                       {"id": 4, "label": "green cube", "names": ["green cube"], "height_mm": 30.0, "score": 0.8,
                        "colour": "green", "frame_ids": ["a", "b"], "size_mm": [40.0, 40.0], "xy": None}]}
    first = update(None)
    old["objects"][1]["xy"] = list(first["objects"][0]["xy"])  # same place as the next sighting
    m = update(old)
    assert len(m["objects"]) == 1, m["objects"]  # the xy-less entry is dropped, the other one matched by label
    e = m["objects"][0]
    assert e["id"] == 4 and e["n_seen"] == 3 and e["emb"] is not None
    assert m["next_id"] == 5


def test_replay_equals_live():
    # The map is a fold over the logged records: fold_observations() must rebuild exactly the live map
    # [CONFIDENCE: CONFIRMED 95% — same apply_observation() on both paths; the record survives a JSON round trip]
    live, log = TM.empty_map(), []
    for i, u0 in enumerate((600, 610, 900)):  # still, nudged, moved 300 px (~125 mm): the entry jumps
        live = update(live, u0=u0, now=1000.0 + 2 * i)
        log.append(json.loads(json.dumps(live["last_update"]["observation"])))
    assert _core(TM.fold_observations(log)) == _core(live)
    assert _core(TM.fold_observations(log, {})) == _core(live)
    assert _core(TM.rebuild_map(log, since_ts=1002.0)) == _core(TM.fold_observations(log[1:]))


def test_light_record_refused():
    live = update(None)
    rec = {k: v for k, v in live["last_update"]["observation"].items() if k != "depth_valid"}
    try:
        TM.fold_observations([rec])
    except ValueError as e:
        assert "depth_valid" in str(e)
    else:
        raise AssertionError("a record without depth_valid must be refused")


def test_best_on_minimal_maps():
    # Queries must not crash on maps made elsewhere either (no objects key, entries without embeddings)
    assert TM.best_on_map({}, "green cube")["map_obj"] is None
    m = {"objects": [{"id": 0, "label": "green cube", "names": ["green cube"], "xy": [300.0, 0.0], "height_mm": 30.0,
                      "colour": "green", "score": 0.9}]}
    b = TM.best_on_map(m, "green cube")  # no store and no embedding: a refusal is fine, an exception is not
    assert {"map_obj", "xy", "reason", "already_in_square"} <= set(b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
