"""
Create the sorter's Atlas collections, plain indexes and the Vector Search index. Idempotent.

WHY: everything the arm worker, perception and dashboard need in Atlas exists after one command, and
running it again changes nothing. The vector index lets Store.search_views use $vectorSearch.

Vector index definition follows the Atlas "vectorSearch" index syntax:
{ATLAS DOCS vector-search-type "\"TYPE\": \"VECTOR\", \"PATH\": ..., \"NUMDIMENSIONS\": ..., \"SIMILARITY\": \"EUCLIDEAN | COSINE | DOTPRODUCT\""}
[CONFIDENCE: SINGLE-SRC 85% — syntax read from the docs page; not yet run against the hackathon cluster]

Usage: MONGODB_URI=... python scripts/atlas_setup.py      (MONGODB_DB optional, default "sorter")
"""
import os
import sys
import time
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.operations import SearchIndexModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter.store import COLLECTIONS, EMBED_DIM, LIVE_COLLECTIONS, VECTOR_INDEX  # noqa: E402

VECTOR_DEF = {
    "fields": [
        # CLIP embeddings are L2-normalised; cosine matches Store's offline scoring
        {"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"},
        # Lets search_views(..., label=...) pre-filter to one object
        {"type": "filter", "path": "label"},
    ]
}
PLAIN_INDEXES = {
    "attempts": [[("ts", DESCENDING)], [("label", ASCENDING), ("ts", DESCENDING)], [("run_id", ASCENDING)]],
    "commands": [[("status", ASCENDING), ("ts", ASCENDING)], [("ts", DESCENDING)]],
    "calib_samples": [[("ts", DESCENDING)]],
    "views": [[("label", ASCENDING)]],
    "status": [[("ts", DESCENDING)]],
    "frames": [[("ts", DESCENDING)]],
    "clips": [[("attempt_id", ASCENDING), ("seq", ASCENDING)]],
    "table_map": [[("ts", DESCENDING)]],
    "sightings": [[("ts", DESCENDING)]],
}


def main() -> None:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("set MONGODB_URI (Atlas connection string) first")
    db = MongoClient(uri, serverSelectionTimeoutMS=8000)[os.environ.get("MONGODB_DB", "sorter")]

    existing = set(db.list_collection_names())
    for coll in COLLECTIONS + LIVE_COLLECTIONS + ("clips", "table_map", "sightings"):
        if coll not in existing:
            db.create_collection(coll)
            print("created collection", coll)
    for coll, specs in PLAIN_INDEXES.items():
        for keys in specs:
            # create_index is a no-op when the same index already exists
            print("index", coll, db[coll].create_index(keys))

    names = {ix["name"] for ix in db.views.list_search_indexes()}
    if VECTOR_INDEX in names:
        print("vector index", VECTOR_INDEX, "already exists")
    else:
        db.views.create_search_index(SearchIndexModel(definition=VECTOR_DEF, name=VECTOR_INDEX, type="vectorSearch"))
        print("vector index", VECTOR_INDEX, "requested")

    # Atlas builds search indexes asynchronously; wait until it can serve queries
    deadline = time.time() + 180
    while time.time() < deadline:
        ix = next((i for i in db.views.list_search_indexes(VECTOR_INDEX)), None)
        if ix and ix.get("queryable"):
            print("vector index ready:", ix.get("status"))
            return
        print("  waiting for vector index:", ix.get("status") if ix else "missing")
        time.sleep(5)
    sys.exit("vector index not queryable after 3 min; check Atlas UI > Atlas Search")


if __name__ == "__main__":
    main()
