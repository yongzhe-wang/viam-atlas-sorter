# VLA dataset: record episodes, export RLDS

Record the robot doing tasks, then export the recordings as an RLDS dataset (Open X-Embodiment style) for training
vision-language-action policies later.

```
scripts/collect_vla.py ──► sorter/vla_recorder.py ──► dataset_raw/<session>/<episode>/ ──► rlds/sorter_vla ──► RLDS (TFRecord)
 (runs the tasks)           (samples at 10 Hz)          (meta.json, steps.jsonl, images)     (tfds build)
```

The team's pick-and-place code is the "expert". The recorder does not drive anything: it samples the camera
and the arm on its own timer, and sees every command by wrapping `Rig.goto`, `Rig.open` and `Rig.grab`.

## 1. Collect (on the machine next to the arm)

Someone stands at the e-stop. Do not send the `sorter` module commands while this runs; both would drive the arm.

```bash
export VIAM_MACHINE_ADDRESS=... VIAM_API_KEY_ID=... VIAM_API_KEY=...
.venv/bin/python scripts/collect_vla.py --episodes 5 --objects "yellow block" "green cube"   # first, a short run
.venv/bin/python scripts/collect_vla.py --episodes 50                                        # objects from the library
```

| Option | Default | Meaning |
|---|---|---|
| `--episodes N` | 10 | grasp episodes; each successful one is followed by a recorded reset episode |
| `--objects ...` | library labels | what to ask for |
| `--hz` | 10 | sampling rate (the real rate is what the camera allows; see `n_steps / duration_s` in meta.json) |
| `--no-depth` | off | skip depth images (saves disk) |
| `--no-reset-episodes` | off | still reset, but do not record it |
| `--session` | start time | folder name for this run |
| `--out` | `dataset_raw/` | where episodes go (in `.gitignore`) |

Each episode records two kinds of task, with paraphrased instructions:
- **grasp**: "put the green cube in the blue square" (and paraphrases)
- **reset**: "take everything out of the blue square"

Every episode is kept, including failures, with `status`, `success` and `fail_reason` in `meta.json`. Filter at
export time.

### Raw episode format

```
dataset_raw/<session>/
  episodes.jsonl                  one summary line per episode
  <episode_id>/
    meta.json                     instruction, kind, outcome, timing, camera intrinsics, robot, settings
    steps.jsonl                   per step: t, t_state, t_image, ee pose (mm, Viam orientation vector, theta deg),
                                  joints (deg), gripper_cmd (0/1), target (commanded goto), image/depth paths
    rgb/000000.jpg ...            wrist colour, 640x360
    depth/000000.png ...          wrist depth, 16-bit millimetres, 640x360
```

## 2. Export to RLDS

Needs TensorFlow: `pip install tensorflow tensorflow-datasets`.

```bash
cd rlds/sorter_vla
SORTER_VLA_RAW=../../dataset_raw tfds build                  # config "rgb"
SORTER_VLA_RAW=../../dataset_raw tfds build --config rgbd    # with depth
```

Output goes to `~/tensorflow_datasets/sorter_vla/<config>/1.0.0/` (change it with `--data_dir`).

- `SORTER_VLA_ONLY_SUCCESS=1`: successful episodes only.
- `SORTER_VLA_KINDS=grasp`: one task kind only.
- **Splits**: put `splits.json` in the raw folder, e.g. `{"train": ["20261030-1400", ...], "val": ["20261031-0900"]}`.
  Without it, everything is `train`. Always split by **session**, never by step: neighbouring frames are
  near-duplicates and would leak the test set into training.

Load it:

```python
import tensorflow_datasets as tfds
ds = tfds.load("sorter_vla/rgb", split="train")
for episode in ds.take(1):
    for step in episode["steps"]:
        step["observation"]["image"], step["action"], step["language_instruction"]
```

### RLDS schema (units: metres, radians)

| Field | Shape | Meaning |
|---|---|---|
| `observation/image` | 256×256×3 uint8 | wrist colour |
| `observation/depth` | 256×256×1 uint16 | wrist depth in mm (`rgbd` only) |
| `observation/state` | 8 float32 | x, y, z, o_x, o_y, o_z, yaw, gripper (0 open / 1 closed) |
| `observation/joint_state` | 6 float32 | joint angles |
| `action` | 7 float32 | dx, dy, dz, droll, dpitch, dyaw to the next step; gripper command at the next step. Roll and pitch are always 0 (the tool always points down) |
| `action_abs` | 5 float32 | commanded goto target in effect: x, y, z, yaw, gripper |
| `reward` | float32 | 1 on the last step of a successful episode |
| `is_first`, `is_last`, `is_terminal`, `discount`, `language_instruction`, `timestamp` | | standard RLDS |
| `episode_metadata` | | file_path, episode_id, session, kind, status, success, source, robot, hz, n_gaps |

The schema is defined once, in `rlds/sorter_vla/convert.py` (`spec()`), and used by both the builder and the tests.

## Tests

```bash
python scripts/test_vla_recorder.py     # fake robot -> recorder -> raw -> RLDS dicts -> schema; no robot or TensorFlow
```

## Known limits (check on the first real run)

- **Not yet run on the arm.** Do a 5-episode run first, then check `meta.json`. `n_steps / duration_s` is the real
  rate, and `gaps` counts missed reads.
- **Extra camera reads.** The recorder asks the camera for frames alongside the control code. If the control code
  gets slower, lower `--hz`.
- **Stale images.** Right after a move the camera returns frames up to ~0.5 s old. Each step stores `t_state` and
  `t_image`, so images can be re-aligned later.
- **Wrist camera only.** A fixed camera over the table would make the data much more useful. Add it to the Viam
  machine and the recorder can be extended to save it.
- **No `language_embedding`.** Some OXE loaders expect one; compute it at training time if needed.
- **`tfds build` has not been run yet.** It was written to the TFDS builder API, but only tested against a stub.
  The first build will show any API mismatch.
