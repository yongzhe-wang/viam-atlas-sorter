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
- One job at a time. A request while busy returns `{"accepted": false, "error": "busy", "running": {...}}`.
- Summaries contain `status` (`done`, `failed`, `not_found`, `ambiguous` = two objects match about equally,
  `too_wide`, `error`), `label`, `score`, `duration_s`, and `fail_phase` / `fail_reason` on failure.
- A hand or arm reaching into the workspace pauses the robot; it resumes 2 s after the workspace is clear.

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

First start on a robot: `run.sh` creates a virtualenv, installs **CPU-only** torch from
`https://download.pytorch.org/whl/cpu` (the default Linux wheel pulls more than 2 GB of CUDA libraries), then
`requirements.txt`. On our robot (sfhack4, linux/amd64) this took about 10 minutes before the service came up.
Later starts skip the install. If viam-server gives up waiting, run `./run.sh` once by hand on the robot.

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
