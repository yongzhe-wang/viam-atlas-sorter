# Execution plan, Thu Oct 29 afternoon → Fri demo

Goal for the demo: **a dashboard where you type "grasp green square", the arm finds that object among
whatever is on the table, picks it, and drops it in the blue tape square**, with every attempt logged to
Atlas and the success rate on Atlas Charts. `docs/PLAN.md` is the long-range design; this file is what
we execute now and who owns what.

**The blue square moves.** The tape square is stuck on a sheet of paper that a person moves around the
table between (or during) runs. Never assume its location: find it fresh for every place, keep
re-detecting during the servo, search the reachable table (grid/spiral from the observe height) when it
is not in view, and keep self-reset spots away from the ring and the paper.

## Judging (email from Shannon Bradshaw, Viam, Oct 29 11:16) — 5-minute demo per team

| Criterion | Weight | What we do about it |
|---|---|---|
| Functionality (does it work well) | 40% | reliability first: ≥8/10 runs, reset loop, person-intervention handling; rehearse the exact demo script |
| Ambition | 20% | any object by description, moving target square, self-reset, data-driven grasp memory |
| Creativity | 20% | language → object → square, live Atlas dashboard of attempts |
| Modularity (reusable by other robot builders) | 15% | ship it as a Viam module: a generic service `sorter` with DoCommand `{"grasp": "<description>"}`, `{"reset": true}`, `{"status": true}`; config-only setup (arm, gripper, cam names, target colour); README for other teams |
| Halloween style (spooky, cute, funny) | 5% | Halloween objects (mini pumpkin, candy) in the demo, a "trick or treat" prompt, spooky dashboard theme toggle |

## Done criteria (from the user)

| # | Workstream | Done when |
|---|---|---|
| 1 | Pick-and-place hardening | 10 consecutive runs, ≥ 8 succeed, per-run time known; first 3 runs' samples fit the jaw calibration and the camera offset is written back to Viam |
| 2 | Atlas | every attempt goes to `attempts`, calibration samples to `calib_samples` (replaces jsonl); one run adds one point on the Atlas Chart |
| 3 | Any object | objects segmented by height (not colour); CLIP embeddings in `views` with a vector index; 15–25 objects scanned; a scanned object placed anywhere is recognised |
| 4 | Demo | dashboard text box "grasp green square" → correct object in the square |

## Owners (one writer per file; nobody edits another owner's files)

| Agent | Session | Owns | Files |
|---|---|---|---|
| AGENT1 | robot-hackathon-planning | **the arm — the only session allowed to move it**; hardening, 10-run test, calibration, person intervention, executor, command worker | `scripts/pick_place_auto.py`, `scripts/jaw_calib.py`, `scripts/run_trials.py`, `sorter/arm.py`, `sorter/executor.py`, `sorter/worker.py` |
| AGENT 2 | plan-fork | Atlas data layer + dashboard + Viam module packaging | `sorter/store.py`, `scripts/atlas_setup.py`, `dashboard/**`, `module/**`, `meta.json` |
| AGENT3 | plan-fork-2 | perception: height segmentation, CLIP, text ranking, view capture | `sorter/perception.py`, `sorter/clip_embed.py`, `scripts/capture_views.py`, `scripts/eval_perception.py` |
| main | coordinator | plan, interfaces, review, integration test | `docs/EXEC_PLAN.md`, `docs/PLAN.md`, `README.md` |

Camera reads (`get_images`) are read-only and any agent may do them. **Only AGENT1 sends motion or
gripper commands.** Viam config changes go through AGENT1 (they change what the arm code sees).

Git in the shared checkout: `git add <your own paths>` only (never `-A`), commit as
`Yongzhe Wang <yzwang2020@gmail.com>` with no Claude trailer, never reset/checkout/stash others' files,
`git pull --ff-only origin main` then push. No API keys or Mongo URIs in files: env vars only
(`VIAM_*`, `MONGODB_URI`).

## Interfaces (code to these; stubs are fine until the real thing lands)

```python
# sorter/perception.py  (AGENT3)
@dataclass
class Obj:
    id: int
    bbox: tuple            # (u0, v0, u1, v1) colour pixels
    centroid_px: tuple     # (u, v)
    yaw_deg: float         # long axis in the image, ccw from +u, [0, 180)
    length_px: float
    width_px: float
    height_mm: float       # top above the table plane
    top_depth_mm: float    # camera -> top surface
    mask: np.ndarray       # bool (H, W), colour image grid
    crop: np.ndarray       # uint8 RGB crop around bbox
def segment(rgb, depth) -> tuple[float, list[Obj]]      # (table_depth_mm, objects); depth aligned to rgb

# sorter/clip_embed.py  (AGENT3)
def embed_images(crops: list) -> np.ndarray             # (N, 512), L2-normalised
def embed_text(text: str) -> np.ndarray                 # (512,), uses "a photo of a {text}"
def rank(objs, text, store=None) -> list[tuple]         # [(Obj, score)] best first; uses views/aliases if store given

# sorter/store.py  (AGENT 2) — Atlas when MONGODB_URI is set, else local JSONL queue that flushes later
class Store:
    def log_attempt(self, doc: dict) -> str
    def log_calib(self, doc: dict) -> None
    def add_view(self, label: str, embedding, jpeg: bytes, meta: dict) -> str
    def search_views(self, embedding, k: int = 5) -> list[dict]   # $vectorSearch on views.embedding
    def add_command(self, text: str) -> str                      # status: pending
    def next_command(self) -> dict | None                        # oldest pending -> running
    def finish_command(self, cmd_id, status: str, attempt_id=None, chosen=None) -> None
```

`attempts` doc (minimum): `ts, run_id, mode (trial|demo|teach), target_text, label, score, height_mm,
grasp {z, theta, grip_depth}, phase_times {servo, descend, grab, carry, place}, duration_s, success,
fail_phase, fail_reason, img_before, img_after` (image paths or small JPEG bytes).

## Command flow (the demo)

dashboard text box → `Store.add_command("green square")` → `sorter/worker.py` (AGENT1, runs next to the
arm) → observe pose → `perception.segment` → `clip_embed.rank(objs, text, store)` → executor picks the
top object (re-detect + servo + depth heights, as in `pick_place_auto.py`) → place in the blue square →
camera check → `Store.log_attempt` + `finish_command` → dashboard updates.
Verbs ("grasp", "pick up") are stripped; the destination is always the blue square.

## Person intervention (AGENT1)

- Re-detect the target before every phase (servo, descend, grab, place). Moved → re-servo. Gone → go back
  to the observe pose and search; still gone after 3 tries → park and report.
- Hand or arm in the workspace = depth blob > 100 mm above the table (blocks are < 80 mm) → pause at a
  safe height until clear for 2 s, then re-detect.
- Grab returns False → open, lift, re-detect, retry (max 2). `is_holding_something()` False during the
  carry → the object was dropped or taken → go back and search.
- Connection drop → reconnect with backoff and resume from the last finished phase (moves are absolute,
  so repeating one is safe). Better: run the worker on the `sfhack4-main` Linux box over LAN.

## Reset: clear the blue square (AGENT1; button by AGENT 2)

Runs before every trial, after every successful place (so trials loop unattended), and on demand from
the dashboard ("Clear square" button → `Store.add_command("__reset__")`).

1. **Pick-out (preferred):** segment the square region; for each object inside the hole, pick it with the
   normal executor and drop it at a random free spot in a safe rectangle away from the square
   (free = no depth blob within 60 mm, not under the tape, inside reach).
2. **Wipe (fallback when a pick fails twice or the object is too wide/flat to grasp):** gripper closed,
   fingertips ~10 mm above the measured table, one straight slow pass across the hole toward a side that
   depth shows is empty; never toward the arm base or the room side. Then re-check; at most 2 passes.
3. **Verify:** hole is empty by depth (no blob > 8 mm inside the hole). Log `attempts` with `mode: reset`.
   If it is still not empty → park and report on the dashboard; do not start the next trial.

## Order of work

| When | AGENT1 (arm) | AGENT 2 (Atlas + dashboard) | AGENT3 (perception) |
|---|---|---|---|
| now → 1 PM | finish the place the arm is holding; reconnect/resume; set `align_color_depth: true` on `cam` and drop `DEPTH_DU/DV` in the same change; self-reset (block → random free spot) | `store.py` with JSONL fallback; `atlas_setup.py` (collections + vector index); dashboard skeleton (live snapshot, command box, attempts table) | `segment()` + `embed_*()` + `rank()` on frames captured from the observe pose; eval script with labelled frames |
| 1 → 3 PM | 10-run test via `run_trials.py` (self-reset between runs); jaw-calib fit from first 3 runs; write cam offset to Viam; intervention handling | Atlas live as soon as `MONGODB_URI` arrives; Charts: success rate over attempts, per label; embed in dashboard | `capture_views.py`: object placed by hand, 4 views each, label typed → `Store.add_view`; scan 15–25 objects |
| 3 → 6 PM | `executor.py` on `Obj` (any object, grasp across `width_px`); `worker.py` command loop | command status + last attempt images on the dashboard | "green square" style queries: CLIP + colour-word boost; measure top-1 on the table set |
| 6 → 9 PM (hard close) | end-to-end rehearsal ×10 from the dashboard | polish, demo view | fix misses found in rehearsal |
| Fri AM | freeze 12:30, rehearse, backup video | | |

## Blocked on the user

- `MONGODB_URI` for the hackathon Atlas cluster (and Charts access in the Atlas UI). Everything else proceeds
  without it; data queues locally and flushes when it arrives.
- Overnight running: the event page says "9:00 PM: Hard close: Robots power down … (no overnight hacking)";
  `PLAN.md` says organizers allowed it. Treat overnight as off unless an organizer confirms in writing.

## Facts measured on sfhack4 (use these, not the config)

- Real table top ≈ z 0 mm in the arm frame (the `table` obstacle z=-123 is the centre of a 200 mm box).
- Fingertips ≈ 155 mm past the flange; camera ≈ 137 mm above the fingertips along the tool axis.
- Jaw axis pixel at fingertip depth ≈ (864, 685); configured cam offset is 3–4 cm off.
- Frames right after a move are stale for ~0.5 s; wait 1 s.
- RealSense depth invalid closer than ~280 mm. `align_color_depth: true` set on `cam` at ~11:50 Oct 29 (fragment override; delete it to roll back): depth is now pixel-aligned to colour (IoU 0.31 at identity vs 0.06 at the old (-60,+22) shift).
- Laptop ↔ arm WebRTC link dropped twice in ~10 minutes.
