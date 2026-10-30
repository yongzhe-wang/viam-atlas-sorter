"""
What counts as done for a grasp, when a list stops, and what this module run has already put into the bucket.

WHY (the bucket memory): since AGENT3's d1ce088 the bucket and everything in it are cut out of the segmentation, so an
object delivered into it leaves the map, and asking for it again answered "not on the table" after a whole search.
{VIAM CENTERAL 2026-10-30 "用本次 MODULE 运行期间的成功记录…判断：这个物体已经投进桶、并且 MAP 上没再看到它 → 回答
ALREADY_IN_BUCKET，算 DONE，不动 ARM"} Each delivered grasp is kept with its label, the words used and the library's
other names for it; a later request it answers is already_in_bucket, unless something it names is on the table: an
object of another label (another screwdriver), or this one seen on the map after it went in (someone took it out).
Kept for this module run only; a reset that empties the drop target forgets it.
"""
import time
from typing import Optional

# A list item that needs nothing more: it is in the drop square, or in the bucket already
ITEM_SATISFIED = ("done", "already_in_square", "already_in_bucket")


def list_stop(item: dict) -> Optional[str]:
    """
    Why the rest of a list must not be tried, else None (VIAM CENTERAL): the E-stop, a dead link, an arm stopped
    by overcurrent, something still in the gripper, or an error that leaves the arm's state unknown. Everything
    else (not found, ambiguous, too close, too wide, square full, an ordinary failure) moves on to the next item.
    """
    status, why = item.get("status"), str(item.get("fail_reason") or "")
    low = why.lower()
    if status == "estop":
        return "E-stop"
    if status == "error":
        return f"unexpected error: {why}"
    if "link still down" in low or "could not connect" in low:
        return f"robot link down: {why}"
    if "overcurrent" in low:
        return f"arm stopped on overcurrent: {why}"
    # "... still holding | put back where it was picked (x,y)": recover() let go on the table, the jaws are empty
    # {RUN 14:46 "GREEN CYLINDER FAILED ... STILL HOLDING | PUT BACK" -> "BLUE LOLLIPOP NOT TRIED"}
    put_back = "put back where it was picked" in low
    if "recovery failed" in low or ("still holding" in low and not put_back):
        return f"may still be holding something: {why}"
    return None


def words_match(query: str, names: list) -> bool:
    """clip_embed.name_matches' rule (the query's words in a name or a name's words in the query), for a stack without it."""
    q = set(str(query or "").lower().split())
    return bool(q) and any(q <= set(str(n).lower().split()) or (str(n).split() and set(str(n).lower().split()) <= q)
                           for n in names)


class Delivered:
    """The objects this module run put into the bucket: label -> {"label", "names", "ts", "attempt"}."""

    def __init__(self, name_matches):
        self.name_matches = name_matches  # clip_embed.name_matches: word containment either way
        self.items = {}

    def add(self, label: Optional[str], text: str, library: list, attempt: Optional[str] = None) -> None:
        """A grasp ended done: `label` (recognised, or None) asked for as `text`; library: Store.list_labels() rows."""
        key = label or text
        names = {key, text}
        for row in library:
            if label in row:
                names.update(row)
        self.items[key] = {"label": key, "names": sorted(n for n in names if n), "ts": time.time(), "attempt": attempt}

    def clear(self) -> None:
        self.items.clear()

    def _on_table(self, entry: dict, text: str, tmap: Optional[dict]) -> bool:
        """
        Something `text` names is on the live map's table: an object of another label at all, or this object seen after
        it went into the bucket. Old spots (superseded) and objects in the drop target do not count.
        """
        for o in (tmap or {}).get("objects") or []:
            if o.get("superseded") or o.get("in_square") or not o.get("label"):
                continue
            if not self.name_matches(text, [o["label"], *(o.get("names") or [])]):
                continue
            if o["label"] != entry["label"] or float(o.get("seen_ts") or 0.0) > entry["ts"]:
                return True
        return False

    def check(self, text: str, tmap: Optional[dict]) -> Optional[dict]:
        """
        The delivered entry `text` asks for while nothing it names is on the table, else None. An entry whose object was
        seen on the table after it went in is forgotten here: someone took it out.
        """
        for key, entry in list(self.items.items()):
            if not self.name_matches(text, entry["names"]):
                continue
            if not self._on_table(entry, text, tmap):
                return entry
            if any(o.get("label") == entry["label"] and float(o.get("seen_ts") or 0.0) > entry["ts"]
                   for o in (tmap or {}).get("objects") or [] if not o.get("superseded") and not o.get("in_square")):
                del self.items[key]
        return None

    def listing(self, tmap: Optional[dict]) -> list:
        """What is in the bucket now as far as this run knows: [{"label", "ts"}], oldest first."""
        out = []
        for e in self.items.values():
            seen = any(o.get("label") == e["label"] and float(o.get("seen_ts") or 0.0) > e["ts"] and not o.get("superseded")
                       for o in (tmap or {}).get("objects") or [])
            if not seen:
                out.append({"label": e["label"], "ts": e["ts"]})
        return sorted(out, key=lambda e: e["ts"])
