"""
Store._flush_observations at each start: sends only what Atlas lacks and never counts an image against the budget twice.

{SFHACK4 LOG 2026-10-30 "2191 OBSERVATIONS, 124 MB"} a backlog that big went up again on every reload, and after a
restart new frames' images stopped reaching Atlas with the budget half empty. Mongo is mongomock, no network.
"""
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import mongomock
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter.store import Store  # noqa: E402

buf = io.BytesIO()
Image.fromarray(np.random.randint(0, 255, (360, 640, 3), np.uint8)).save(buf, "JPEG")
jpeg, D, t0 = buf.getvalue(), tempfile.mkdtemp(), time.time()
db = mongomock.MongoClient()["sorter"]


def start(budget_frames: float) -> Store:
    """A module start: new Store, empty budget counter, the same Atlas and disk."""
    os.environ["SORTER_OBS_ATLAS_MB"] = str(budget_frames * len(jpeg) / 1e6)
    s = Store(uri="", data_dir=D)
    s.db = db
    return s


def images() -> int:
    return db.observation_images.count_documents({})


s = start(6.5)
ids = [s.log_observation({"ts": t0 + k, "source": "idle"}, jpeg) for k in range(4)]
assert db.observations.count_documents({}) == 4 and images() == 4

# Restarts: nothing is sent, and the room left in the budget is still there for new frames
for restart in (1, 2):
    s = start(6.5)
    assert s._flush_observations() == 0
    ids.append(s.log_observation({"ts": t0 + 3 + restart, "source": "idle"}, jpeg))
    assert images() == 4 + restart, (restart, images())
print("1. restarts send nothing; new frames' images still fit the budget (6 of 6)")

s = start(6.5)
ids.append(s.log_observation({"ts": t0 + 6, "source": "idle"}, jpeg))
assert images() == 6 and db.observations.count_documents({}) == 7
print("2. a full budget keeps the 7th frame's image out, its doc goes up")

db.observations.delete_one({"_id": ids[0]})
s = start(6.5)
assert s._flush_observations() == 1 and db.observations.count_documents({}) == 7 and images() == 6
print("3. a doc Atlas lost comes back; over budget no other write")

rec = next(r for r in s.list_observations() if r["_id"] == ids[6])
Path(D, rec["jpeg_path"]).unlink()  # thinned under disk pressure
s = start(100)
assert s._flush_observations() == 0 and db.observation_images.find_one({"_id": ids[6]}) is None
print("4. an image thinned from the disk writes no empty doc and costs no budget")

s = start(100)
s2 = s.log_observation({"ts": t0 + 7, "source": "idle"}, jpeg)
db.observation_images.delete_one({"_id": s2})
assert start(100)._flush_observations() == 1 and db.observation_images.find_one({"_id": s2})["bytes"] == len(jpeg)
print("5. a raised budget lets a missing image through")
