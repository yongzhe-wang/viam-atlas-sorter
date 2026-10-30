"""
Offline tests for the LLM fallback (sorter/llm_choose.py + clip_embed.best(..., rgb=...)).

No network, no CLIP weights, no robot: the API call and CLIP ranking are replaced with stubs.
Run from the repo root: python scripts/test_llm_choose.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import clip_embed, llm_choose  # noqa: E402


def obj(bbox, colour="yellow"):
    return SimpleNamespace(bbox=bbox, colour=colour, recognised=None, recognised_names=[])


RGB = np.full((720, 1280, 3), 200, np.uint8)
A, B = obj((100, 100, 200, 200)), obj((600, 300, 700, 420), "green")
calls = []


def fake_rank(scores):
    return lambda objs, text, store=None, recognised=None, **_: sorted(zip(objs, scores), key=lambda t: -t[1])


def fake_call(reply):
    def _call(jpeg, text):
        calls.append((jpeg, text))
        return reply
    return _call


def setenv(mode, key=True):
    os.environ["SORTER_LLM"] = mode
    if key:
        os.environ["ANTHROPIC_API_KEY"] = "test"
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed = True


check.failed = False

# --- parse ---
check("parse ok", llm_choose.parse('{"status":"ok","number":2,"reason":"green"}', 2) == {"index": 1, "status": "ok", "reason": "green"})
check("parse tolerates text around JSON", llm_choose.parse('Sure: {"status":"none","reason":"x"} done', 2)["status"] == "none")
check("parse ambiguous", llm_choose.parse('{"status":"ambiguous","numbers":[1,2]}', 2)["status"] == "ambiguous")
for bad in ('{"status":"ok","number":5}', "no json", '{"status":"maybe"}'):
    try:
        llm_choose.parse(bad, 2)
        check(f"parse rejects {bad!r}", False)
    except ValueError:
        check(f"parse rejects {bad!r}", True)

# --- annotate produces a JPEG ---
jpg = llm_choose.annotate(RGB, [A, B])
check("annotate returns JPEG", jpg[:2] == b"\xff\xd8" and len(jpg) > 1000)

# --- best(): CLIP confident -> LLM not called in fallback mode ---
clip_embed.rank = fake_rank([0.35, 0.20])
llm_choose._call = fake_call('{"status":"ok","number":2,"reason":"x"}')
setenv("fallback")
calls.clear()
r = clip_embed.best([A, B], "yellow block", rgb=RGB)
check("fallback: confident CLIP keeps its pick, no API call", r["obj"] is A and r["decided_by"] == "clip" and not calls)

# --- CLIP weak (below MIN_SCORE) -> LLM picks ---
clip_embed.rank = fake_rank([0.15, 0.10])
calls.clear()
r = clip_embed.best([A, B], "the thing on the right", rgb=RGB)
check("fallback: weak CLIP -> LLM picks object 2", r["obj"] is B and r["decided_by"] == "llm" and len(calls) == 1)
check("fallback: CLIP's refusal kept for the log", "best match only" in r["llm"]["clip_reason"])

# --- CLIP ambiguous -> LLM says ambiguous -> obj None, ambiguous True ---
clip_embed.rank = fake_rank([0.30, 0.295])
llm_choose._call = fake_call('{"status":"ambiguous","numbers":[1,2],"reason":"both yellow"}')
r = clip_embed.best([A, B], "yellow", rgb=RGB)
check("ambiguous verdict -> no pick, ambiguous", r["obj"] is None and r["ambiguous"] and "both yellow" in r["reason"])

# --- LLM says none ---
clip_embed.rank = fake_rank([0.15, 0.10])
llm_choose._call = fake_call('{"status":"none","reason":"no elephant"}')
r = clip_embed.best([A, B], "purple elephant", rgb=RGB)
check("none verdict -> refuse with LLM reason", r["obj"] is None and not r["ambiguous"] and r["reason"] == "no elephant")

# --- failures fall back to CLIP's decision ---
def boom(jpeg, text):
    raise OSError("network down")
llm_choose._call = boom
r = clip_embed.best([A, B], "x", rgb=RGB)
check("API error -> CLIP decision stands", r["decided_by"] == "clip" and r["obj"] is None and llm_choose.LAST["status"] == "error")
llm_choose._call = fake_call("I think it's the yellow one")
r = clip_embed.best([A, B], "x", rgb=RGB)
check("unparsable answer -> CLIP decision stands", r["decided_by"] == "clip")

# --- off by default / no key / no frame ---
llm_choose._call = fake_call('{"status":"ok","number":1}')
calls.clear()
os.environ.pop("SORTER_LLM", None)
r = clip_embed.best([A, B], "x", rgb=RGB)
check("off by default", r["decided_by"] == "clip" and not calls)
setenv("fallback", key=False)
r = clip_embed.best([A, B], "x", rgb=RGB)
check("no API key -> not called", r["decided_by"] == "clip" and not calls)
setenv("fallback")
r = clip_embed.best([A, B], "x")
check("no rgb -> not called (old callers unchanged)", r["decided_by"] == "clip" and not calls)

# --- always mode overrides a confident CLIP pick ---
setenv("always")
clip_embed.rank = fake_rank([0.35, 0.20])
llm_choose._call = fake_call('{"status":"ok","number":2,"reason":"right one"}')
r = clip_embed.best([A, B], "the right one", rgb=RGB)
check("always: LLM overrides", r["obj"] is B and r["decided_by"] == "llm")

# --- a name the library knows, out of view, does not spend a call at every search view ---
setenv("fallback")
clip_embed.rank = fake_rank([0.15, 0.10])
llm_choose._call = fake_call('{"status":"ok","number":1}')
calls.clear()
st = SimpleNamespace(list_labels=lambda: [["blue block"]])
r = clip_embed.best([A, B], "blue block", st, rgb=RGB)
check("fallback: known name out of view -> searchable, no API call", r["obj"] is None and r["searchable"] and not calls)
# --- one description gets at most MAX_CALLS_PER_WINDOW calls ---
calls.clear()
llm_choose._CALLS.clear()
for _ in range(llm_choose.MAX_CALLS_PER_WINDOW + 2):
    clip_embed.best([A, B], "the thing", rgb=RGB)
check("fallback: calls capped per description", len(calls) == llm_choose.MAX_CALLS_PER_WINDOW)
llm_choose._CALLS.clear()
setenv("always")
clip_embed.rank = fake_rank([0.35, 0.20])
llm_choose._call = fake_call('{"status":"ok","number":2,"reason":"right one"}')
clip_embed.best([A, B], "the right one", rgb=RGB)

# --- verb stripping reaches the prompt ---
check("prompt gets the cleaned query", calls[-1][1] == "the right one")

print("\nALL PASSED" if not check.failed else "\nSOME TESTS FAILED")
sys.exit(1 if check.failed else 0)
