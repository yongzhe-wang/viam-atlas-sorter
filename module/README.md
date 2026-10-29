# `sorter-team:sorter:grasp-by-description`

A Viam **generic service**: tell it what to pick up in plain words, and it puts that object into a blue tape
square on the table. It works with whatever is on the table: no training, no taught poses.

```json
{"grasp": "the green square"}   →   finds it, picks it, drops it in the square, logs the attempt
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
- A square of **blue** painter's tape on the table, roughly 15–25 cm across, within reach.
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
await sorter.do_command({"reset": True})                    # empty the square (pick out, wipe if needed)
```

- Leading verbs are ignored ("grasp", "pick up", "get", ...); the destination is always the square.
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
  only drops pictures and never holds up the arm. The map is saved with `Store.put_table_map` at most every 2 s
  and loaded back on restart. `{"status": true}` → `"map"` reports objects, `updates`, `dropped` and `last_ms`.
  Sweep frames are also saved by the executor under `$VIAM_MODULE_DATA/data/sweeps` (the module sets
  `SORTER_DATA_DIR`). Only the newest 30 sweeps (about 8 MB each) are kept.
- **Finding objects anywhere on the table**, in this order: (1) the cached idle frame → go at once; (2) the
  live map has the object, seen within the last 60 s → `"remembered at (x, y) on the table map, seen N s ago ·
  going there"` (an edge-only sighting adds "looking there first"); if the whole table was swept within the last 60 s and the map does not have it, the answer is
  an immediate "not on the table"; (3) otherwise the arm sweeps the table (`"scanning the table…"`, progress
  `pose k/N` in status), every frame goes into the map, and it picks from it. The idle loop also records each
  recognised object's last-seen table position (`Store.log_sighting`). `{"table_map": true}` returns the live
  map (each object with `seen_ts`) and the sightings for the dashboard.
- One job at a time. A request while busy returns `{"accepted": false, "error": "busy", "running": {...}}`.
- Summaries contain `status` (`done`, `failed`, `not_found`, `ambiguous` = two objects match about equally,
  `already_in_square` = the named object is only in the drop square (answered at once, no sweep), `too_wide`,
  `error`), `label`, `score`, `duration_s`, and `fail_phase` / `fail_reason` on failure.
- A hand or arm reaching into the workspace pauses the robot; it resumes 2 s after the workspace is clear.

From our web dashboard (no Atlas needed): start it with `DASH_COMMAND_PATH=module` (and `DASH_MODULE_NAME` if
your service is not called `sorter`). Its "Grasp" box and "Clear square" button then call this DoCommand, and
the worker panel and command list show `{"status": true}`: the running job, then the result.

```bash
DASH_COMMAND_PATH=module VIAM_MACHINE_ADDRESS=... VIAM_API_KEY_ID=... VIAM_API_KEY=... python dashboard/app.py
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

## Atlas (optional)

```bash
MONGODB_URI="mongodb+srv://..." python scripts/atlas_setup.py   # collections + vector search index, idempotent
```

Then set `MONGODB_URI` for the module. Charts: `dashboard/CHARTS.md`. The live dashboard is `dashboard/app.py`.

## Build, test, publish (maintainers)

```bash
sh module/build.sh                                    # -> module/dist/archive.tar.gz (run.sh + meta.json at its root)
```

The archive must contain `meta.json`: `viam module reload-local` rejects a local tarball without one ("local
tarballs must contain a meta.json with the 'entrypoint' field"). `build.sh` copies it in.

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
