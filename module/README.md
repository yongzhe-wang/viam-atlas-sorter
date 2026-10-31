# `sorter-team:sorter:grasp-by-description`

A Viam **generic service**: tell it what to pick up in plain words, and it puts that object into an orange bucket
(`SORTER_TARGET=bucket`, the default; set `square` for the original blue tape square). It works with whatever is on
the table: no training, no taught poses. The bucket may be moved at any time: it is searched for and confirmed again
over its mouth before the gripper opens.

```json
{"grasp": "the green square"}   →   finds it, picks it, drops it in the bucket, logs the attempt
```

Under the hood: the wrist depth camera finds every object by height above the table, CLIP picks the one that
matches your words, the arm aligns on it with closed-loop visual servoing, takes all heights from depth,
grasps across the object's short side, and releases it inside the square. Every attempt (what was asked,
what was picked, success, timing, before/after images) goes to MongoDB Atlas, so you can chart your success
rate and search past objects by image embedding.

## What your robot needs

- An arm with a parallel gripper that reports `is_holding_something` (built and tested on a UFactory xArm6 +
  xArm gripper G2).
- A wrist-mounted depth camera returning colour + depth from `GetImages` (tested: RealSense D435 with
  `align_color_depth: true`).
- An orange bucket (about 200 mm across, 180 mm tall) on the table within reach, or, with `SORTER_TARGET=square`, a
  square of **blue** painter's tape roughly 15–25 cm across.
- Objects 10–80 mm tall and narrower than your gripper's opening (about 78 mm on the xArm gripper).

## Add it to your machine

1. In the Viam app: **CONFIGURE → + → Service → generic**, search the registry for `sorter`, add
   `grasp-by-description`. (Before it is published: build it and add it as a local module, see below.)
2. Attributes (all optional except where your names differ):

```json
{
  "arm": "arm",
  "gripper": "gripper",
  "camera": "cam",
  "square_colour": "blue",
  "consume_dashboard_commands": false,
  "home_xy": [250, 0],
  "jaw_tip_px": [864, 685],
  "tip_below_cam_mm": 137
}
```

| Attribute | Meaning |
|---|---|
| `arm`, `gripper`, `camera` | Names of your resources. They become dependencies of the service. |
| `square_colour` | Colour of the drop square. Only `"blue"` is supported today. |
| `mongodb_uri` | Atlas connection string. Better: set the env var `MONGODB_URI` in the module's environment, so the secret stays out of the attributes. Without either, attempts go to local JSONL files on the robot. |
| `consume_dashboard_commands` | `true` = also run commands queued from our web dashboard (`dashboard/`), replacing `sorter/worker.py`. |
| `home_xy` | Flange x, y (mm) of the pose the arm observes the table from, 400 mm up, tool pointing down. |
| `jaw_tip_px`, `tip_below_cam_mm` | Where the midpoint between the fingertips appears in the colour image, and how far the fingertips are past the camera along the tool axis. These depend on your camera mount: see **Calibrate** below. |

Validation runs before anything moves: a bad attribute fails config with a message saying what to fix.

## Use it

From the app's **CONTROL** tab (DoCommand panel on the service) or any SDK:

```python
from viam.services.generic import Generic

sorter = Generic.from_robot(machine, "sorter")
await sorter.do_command({"grasp": "candy corn"})            # returns at once: {"accepted": true, "job": "..."}
await sorter.do_command({"status": True})                   # {"busy": ..., "running": {...}, "last": {...}}
await sorter.do_command({"grasp": "the pumpkin", "wait": True})  # blocks, returns the attempt summary
await sorter.do_command({"stop": True})                     # end the current task only; the queue goes on
await sorter.do_command({"reset": True})                    # bucket: let go, back to the observe pose (square: empty it)
```

- Leading verbs are ignored ("grasp", "pick up", "get", ...); the destination is always the bucket (or the square).
- **Always ready:** `run.sh` downloads the CLIP weights during install into `$VIAM_MODULE_DATA/cache` (kept
  across redeploys). At startup a background task loads CLIP (eval, all CPU threads) and the object library
  into memory and runs one dummy pass (`"warm": true`, `"warm_ms"` in status); until then a grasp answers
  `{"ok": false, "status": "warming_up"}` at once. `{"preview": "..."}` gives the same decision without moving.
  While idle with the arm at the observe pose (`home_xy`, z 400), the service looks at the table about once a
  second (every 2 s if a look takes over 800 ms) and caches what it recognises (`"seen"` in status: labels,
  confidence, boxes, intruders, per-step timings `"seen_ms"`). If that
  frame is at most 1.5 s old, a grasp decides from it at once (`"decide_ms"`). It either starts moving with
  that target or answers immediately `{"ok": false, "status": "not_found", "reason": "no 'pumpkin' on the table"}`.
  Otherwise, for example right after a move, it falls back to the executor's own observe (`"why_slow"`).
  `{"seen_image": true}` returns that frame as a small base64 JPEG with the boxes scaled to it.
- **Live table map** (`module/live_map.py`): every picture the module sees updates one in-memory map through
  `sorter/table_map.py`'s `update_map` (AGENT3: negative evidence drops what is gone; `partial` = seen only cut by
  a picture's edge, position approximate). Sources:
  - idle frames, passed with their `seg` and `recognise_all` output, so the update costs about 40 ms;
  - executor frames through AGENT1's `Rig.on_map_frame` hook, with the pose each frame was taken at and the
    executor phase as source. Outside a sweep at most one per second; theta 0 only while
    `ppa.MAP_ONLY_THETA0`;
  - every sweep frame, none dropped, applied while the arm moves on.

  One update runs at a time in a worker thread, and a newer frame replaces one still waiting, so a slow update
  only drops pictures and never holds up the arm. The map is saved with `Store.put_table_map` at most every 2 s.
  It is **never** loaded back: every module start begins with an empty map, because the table may have changed
  while the module was down. The old map and the sightings move to the Store's history (logs, Atlas), never to
  finding. `{"clear_map": true}` (dashboard **Clear map**) does the same on demand when someone rearranges the
  table; it is refused while a job runs. `{"status": true}` → `"map"` reports objects, `updates`, `dropped` and `last_ms`.
  Sweep frames are also saved by the executor under `$VIAM_MODULE_DATA/data/sweeps` (the module sets
  `SORTER_DATA_DIR`). Only the newest 30 sweeps (about 8 MB each) are kept.
- **Observation log** (user: "all data should be saved to continuously form the map"): every frame folded into the
  live map is appended with `Store.log_observation`. The record is `update_map`'s own
  `last_update["observation"]`, unchanged (AGENT3: ts, source, pose, intrinsics, detections with embeddings,
  square, intruders, depth-validity grid), plus epoch and map version. `table_map.fold_observations` replays it
  to the same map as live; `module/selftest_map.py` checks that on every build. Next to each record go a JPEG of
  at most 640 px and a 320×180 16-bit depth PNG, kept for the dashboard and debugging. An idle picture enters
  the map, and so the log, only when what it shows changed (labels, box centres, intruders), or every 20 s:
  a still table no longer writes a record a second. Records go to `$VIAM_MODULE_DATA/data/observations/<YYYYMMDD>/`;
  with Atlas, also to `observations` (images to `observation_images` up to `SORTER_OBS_ATLAS_MB`, default 300).
  Nothing is ever deleted by Clear map or a restart: they only move the map's epoch. `{"rebuild_map": {"since":
  ts}}` folds the log since `ts` (default: the epoch) back into the live map with AGENT3's
  `table_map.fold_observations`, e.g. after a crash. When free disk drops under 5 GB, the images of the oldest idle
  frames are removed first; their records stay, and search, grasp and sweep frames are never thinned.
  `{"status": true}` → `"observations"` gives `count`, `bytes` and `free_bytes`. To read the log off the robot
  (e.g. to make library views from its frames), use the read-only
  `{"observations": {"since": ts, "until": ts, "images": false, "limit": 200, "source": "idle"}}`. It returns
  the records as logged, oldest first; with `images`, each record also carries `jpeg_b64` and `depth_png_b64`.
  A page holds at most 200 records (40 with images) and 3 MB, and `"more"` plus `"next_since"` page on. It never
  touches the map, a job or the arm, and it answers while busy.
- **Finding objects anywhere on the table**: a grasp starts picking at once, with no scan first (the user's
  rule: "just search, and if found just start grasp; do NOT have an initial step").
  1. Cached idle frame shows it → go.
  2. The live map saw it within the last 60 s → AGENT1's `search_and_find` looks there first
     (`"remembered at (x, y) on the table map, seen N s ago · looking there first"`).
  3. Otherwise → it searches the table pose by pose.

  The search stops at the first view whose match is accepted and the pick starts right there. Status
  `running.search` = `{k, n, path, found}`: "searching… view k/N", then "found at view k, grasping". Immediate
  answers come only for two equal matches in the cached frame, or an object already in the drop square.
  `{"scan_table": true}` (dashboard **Scan table**) scans the whole table into the live map, and only when asked
  (progress `pose k/N`). The idle loop also records each recognised object's last-seen table position
  (`Store.log_sighting`). `{"table_map": true}` returns the live map (each object with `seen_ts`), the sightings
  and the scan / search progress for the dashboard.
- **Several objects in one command:** `{"grasp": "i want screw driver, and snack, and tape"}` becomes one job of up to
  5 objects, in order (`sorter/multi_objects.py` cuts at `,` `and` `&` `then` `+`, drops the spoken lead-in, ignores
  spaces when comparing with library names, and keeps a name like "blue and black screwdriver" whole). Colours sharing
  a noun never leave a bare colour: "red and blue screwdriver" is two objects because the library has both, "black and
  yellow tape" is one described by both colours. Each item goes
  through the same decision and retries as a single grasp, with a fresh look from the observe pose before it, so what
  was just delivered counts. If several objects fit a name, all of them are picked, one after another (`lollipop (1/2)`, `(2/2)`). `running.progress` reads `item 2/3: snack`, and the result is `done` / `partial` / `failed`
  with `items: [{text, status, label, duration_s, fail_reason}]`. An item already in the square counts as done.
  - Moves on to the next item: `not_found`, `ambiguous`, `too_close`, `too_wide`, `square_full`, `unsafe_grasp`,
    `failed`.
  - Stops, the rest `skipped`, `stopped_by` says why: the E-stop, a dead link, overcurrent, something possibly still
    held, an unexpected error.

  `{"preview": "..."}` answers item by item without moving. One object takes the single path, as before.
- **The library learns (`SORTER_TEACH=1`, off by default; `sorter/teach.py`):**
  - `{"teach": "m&m bag"}` takes the object clicked (`"point": [u, v]`, 0-1 of the picture) or else the best match for
    the name, then looks at it from where the arm stands, from two poses beside it at z 400 and from one at z 580. Each
    view (CLIP embedding, height, colour, small JPEG, `source: "teach"`) goes to the Store at once, so `$vectorSearch`
    answers with it; the summary says after how many seconds (`searchable_s`). The arm stays at z >= 400 and touches
    nothing; it runs as a job (busy, E-stop) and ends at the observe pose. A known name adds views to that object, and
    `"alias_of": "kind bar"` makes the name another word for one.
  - A grasp that ends `done` with its check passed keeps the crop that found the object as a new view (`source: "grasp"`,
    with `primitive` and `grip`), under its recognised name, or under the words used if it was not recognised and the
    pick was clear. `learned` in the summary says what happened. Nothing is learned when the words name one object and
    recognition says another: that may be a wrong pick, and `{"teach"}` settles it.
  - A view within cosine 0.98 of one already there is skipped (a teach compares with its own label only). A label takes
    at most 40 views from grasps. Atlas: observations stop going up at 450 MB (`SORTER_ATLAS_STOP_MB`) of the free
    tier's 512 MB, so views and attempts still fit; they stay on the robot's disk and `flush()` sends them later.
  - `{"status": true}` has `atlas` (views, labels, attempts, usage, `learned`, `last_search`) for the dashboard's panel.
- One job at a time. A grasp while busy is queued: `{"accepted": true, "queued": true, "position": n}`, each object of a
  comma list its own entry; queued jobs start by themselves when the arm is free. `{"clear_queue": true}` and a reset
  empty the queue. `{"stop": true}` ends only the running task: it stops at its next arm command, lifts, puts back
  what it holds, is not retried, and reports `stopped`.
- Summaries contain `status` (`done`, `failed`, `not_found`, `stopped`, `already_in_square` = the named object is only in
  the drop square (square mode), `too_wide`, `too_close` = too near the arm base, please move it out, `error`), `label`, `score`, `duration_s`, and `fail_phase` / `fail_reason` on failure.
- There is no hand detection (removed on 2026-10-30: scenery past the table's edge kept pausing the arm). Keep a
  person on the E-stop.

Our web dashboard (no Atlas needed; `DASH_MODULE_NAME` if your service is not called `sorter`) shows three live
views: the command box (Grasp, Stop, Reset) with the queue and the job's state,
progress and per-item results; the arm camera with the recognised boxes; and the module's map of the table with the
bucket, the arm and what the camera sees. Its robot link sends this module one `{"dashboard": {"since": ts}}` at most
twice a second, however many browsers are open; the picture is the newest of the idle look, every executor frame of a
job, and a colour frame grabbed for the call while idle (`module/dashboard_view.py`).

```bash
VIAM_MACHINE_ADDRESS=... VIAM_API_KEY_ID=... VIAM_API_KEY=... python dashboard/app.py
```

## Calibrate (once per camera mount)

1. Put the arm at `home_xy`, z = 400 mm, tool down. Close the gripper slowly until the fingertips appear at
   the bottom of the colour image. The pixel midway between them is `jaw_tip_px`.
2. `tip_below_cam_mm` is the distance from the camera to the fingertips along the tool axis (measure, or take
   it from your frame config: fingertip offset minus camera offset).
3. Optional refinement: every place records where the released object really lands, and
   `python scripts/jaw_calib.py fit` turns those samples into a corrected jaw-axis model (`data/jaw_calib.json`,
   bundled by the build if present).
4. Optional: `data/workspace.json`, a polygon `[[u, v], ...]` in colour pixels, limits where objects may be
   picked from.

## Recording

- **Every attempt:** the arm code calls `Store.put_clip_frame(attempt_id, phase, jpeg, meta)` at each phase and
  servo step; the dashboard's attempts table then has a **Replay** button (frame-by-frame player with phase and
  time, speed control, GIF / MP4 download; MP4 needs `ffmpeg` on the dashboard machine).
- **Timelapse:** the service saves one colour frame every 5 s, stamped with the time and what it is doing, to
  `$VIAM_MODULE_DATA/timelapse/<epoch_ms>.jpg`, keeping the last 2 hours (`"timelapse"` in status). To cut a
  backup video on the robot:
  `ffmpeg -framerate 12 -pattern_type glob -i "$VIAM_MODULE_DATA/timelapse/*.jpg" -pix_fmt yuv420p timelapse.mp4`

## Object library (what "green square" is matched against)

`build.sh` bundles `data/views.jsonl` (labelled CLIP embeddings from `scripts/capture_views.py` /
`scripts/label_scan.py`) and the crop images they reference into `library/` in the archive. Without Atlas, the
module copies that library into `$VIAM_MODULE_DATA/data` at startup, but only when it is missing or the
bundle's content hash changed. That folder also holds everything the module writes (attempts, commands,
status), so redeploys never lose run data. `{"status": true}` reports `"library": {"labels": N, "views": M}`; check it
after a deploy. With `MONGODB_URI` set, the library is read from Atlas instead (run `atlas_setup.py` and flush
the views from the scanning machine first).

## LLM fallback for object choice (optional, off by default)

CLIP decides which object your words mean, and it struggles with relations and comparisons ("the smaller
cube", "the one left of the cup", "something to write with"). With the LLM fallback on, when CLIP refuses or
is unsure, the service draws a numbered box on every object perception found and asks Claude which number
you meant. Perception still supplies the grasp geometry; the model only picks a number.

Set these in the module's environment (Viam app: CONFIGURE → the module → env), not in the attributes:

| Env var | Meaning |
|---|---|
| `SORTER_LLM` | `off` (default), `fallback` (only when CLIP refuses or is ambiguous) or `always` (every decision) |
| `ANTHROPIC_API_KEY` | required when enabled |
| `SORTER_LLM_MODEL` | default `claude-sonnet-5-5`; `claude-haiku-5-5` is faster |
| `SORTER_LLM_TIMEOUT_S` | default 8 |

- Each call adds roughly 1–5 s and needs internet. If the call fails (no key, offline, timeout, odd answer),
  CLIP's own decision stands, so the robot behaves exactly as before.
- `{"status": true}` shows `"llm": {"mode", "key_set", "last"}`. Each logged attempt records `decided_by`
  (`clip` or `llm`) and the model's verdict and reason under `llm`.
- Offline tests (no network or robot): `python scripts/test_llm_choose.py`.

## Atlas (optional)

```bash
MONGODB_URI="mongodb+srv://..." python scripts/atlas_setup.py   # collections + vector search index, idempotent
```

Then set `MONGODB_URI` for the module. Charts: `dashboard/CHARTS.md`. The live dashboard is `dashboard/app.py`.

## Build, test, publish (maintainers)

```bash
sh module/build.sh                                    # -> module/dist/archive.tar.gz (run.sh + meta.json at its root)
PYTHON=.venv/bin/python sh module/build.sh            # same, plus the import check (needs viam-sdk in that Python)
```

The archive must contain `meta.json`: `viam module reload-local` rejects a local tarball without one ("local
tarballs must contain a meta.json with the 'entrypoint' field"). `build.sh` copies it in.

Every build ends with `module/check_archive.py`, which unpacks the archive into a temp dir and fails the build when
the module could not start from it. It checks:
- `meta.json` and an executable entrypoint;
- `requirements.txt`;
- that every project import in the shipped `.py` files names a shipped file;
- that no macOS `._*` metadata files are included;
- with viam-sdk in `$PYTHON`, that `import module.main` works from the unpacked copy;
- with numpy and Pillow there too, `module/selftest_map.py` from the copy. It runs the shipped live map on the
  shipped `table_map.update_map` (real segmentation on a synthetic frame) for a just-started and a just-cleared
  map. On sfhack4 a hand-made empty map once failed every update with `KeyError: 'next_id'`; this catches that.

`build.sh` also lists any uncommitted changes under `module/`, `sorter/` and `scripts/` that go into the archive.
It packs the working tree, so another agent's half-done edit would otherwise ship unnoticed.

A tarball once shipped without `module/live_map.py` and died on the robot with "ModuleNotFoundError: No module
named 'module.live_map'"; this check now fails such a build first.

First start on a robot: `run.sh` creates a virtualenv in `$VIAM_MODULE_DATA/venv` (kept across module
versions), installs **CPU-only** torch and torchvision as a pair from `https://download.pytorch.org/whl/cpu` (the
default Linux wheel pulls more than 2 GB of CUDA libraries; a torchvision from PyPI does not match that torch and
CLIP dies on import with `operator torchvision::nms does not exist`), then `requirements.txt`. Before marking the
venv installed it imports torch, torchvision and open_clip and runs `torchvision.ops.nms`; if that fails, `run.sh`
exits with the installed versions on stderr and installs again on the next start. On our robot (sfhack4, linux/amd64) this took about
10 minutes before the service came up. Restarts and redeploys reuse that venv; it is reinstalled only when
`requirements.txt` changes (the install marker is keyed on its hash). If viam-server gives up waiting on the
first start, run `./run.sh` once by hand on the robot.

Test on one machine without publishing. This runs as a local module under the placeholder namespace
`sorter-team`, so no registry namespace is needed yet. Tested on sfhack4 on 2026-10-29: the machine lists the
generic service `sorter`, and `{"status": true}` returns `{"busy": false, "store": "jsonl", ...}`.

```bash
viam login
viam module reload-local --file module/dist/archive.tar.gz --part-id <machine-part-id> --model-name sorter-team:sorter:grasp-by-description --resource-name sorter
```

Right after a viam-server restart, the CLI path (`reload-local`, `upload` to the machine) can fail for about
10 minutes with "host appears to be offline" from signaling. During that time the SDK still reaches the machine
over the LAN, so the robot is fine: wait and retry, rather than restarting it again.

Before publishing, create the module under your registry namespace (Viam app → org Settings → public
namespace, or the CLI), then put that namespace in **both** `meta.json` (`module_id`, `models[].model`) and
`NAMESPACE` in `module/sorter_service.py`. `viam module create` writes a new `meta.json` into the current
directory, so run it outside the repo or copy just the `module_id`.

Publish:

```bash
viam module upload --version=0.1.0 --platform=linux/amd64 module/dist/archive.tar.gz
```

Both commands change a real machine or the public registry. On the shared hackathon machine, only run them with the team's go-ahead.

## Limits

- Top-down grasps only; flat things (< 10 mm), very wide things and transparent things (no depth) are skipped
  and reported as such.
- The drop target is found by blue tape; other colours need a new detector.
- One robot per service instance; the arm/perception code assumes the camera rides on the wrist.
