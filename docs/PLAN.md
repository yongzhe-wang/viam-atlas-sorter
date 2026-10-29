# Full plan: a self-improving, language-driven sorter

Hackathon: Viam x MongoDB, Oct 29-30 2026. Demo Friday 3:30 PM. Hardware on `sfhack4`: xArm6, ufactory gripper, wrist RealSense (color + depth), table top at z ≈ 0 mm in the arm base frame (the `table` obstacle z = -123 is the centre of a 200 mm collision box; measured with depth).

## Demo story

1. **Overnight practice.** The robot practiced unattended on ~6 objects. The dashboard shows success rate over time, per object and per grasp type, plus a fixed before/after test run Friday morning.
2. **Language.** Type "the thing you drink from" → it picks the cup (CLIP text→image match, not color) with the grasp it learned for cups overnight, and drops it in the blue square. The destination is always the blue square.
3. **Teach once.** The words pick the wrong object → one click on the right one → next time the same words find it.

Atlas does three jobs: vector search (what is this / which object did the user mean), memory that drives control (which grasp to use), and the ledger behind the dashboard.

## Architecture

| Module | Job | Owner |
|---|---|---|
| Perception | Find every object on the table, measure its shape, embed it | B |
| Memory (Atlas) | Object library, grasp stats, attempts, commands | B |
| Grasp planner | Pick a grasp primitive + params per object | A |
| Executor | Move safely, run the primitive, check success | A |
| Language | Match a text description to an object on the table | C |
| Autonomy | Self-reset loop + watchdog for overnight | A |
| UI | Live view, text box, click-to-correct, Charts | C |

The control loop runs on the `sfhack4-main` Linux box (LAN, survives laptop sleep). The laptop only hosts the UI.

## 1. Perception

- Set `align_color_depth: true` on `cam` so depth pixels line up with color pixels (off by default).
- **Segment by height, not color:** depth points more than 8 mm above the table plane, inside a fixed workspace rectangle → connected components → objects. Blue tape is flat, so it is ignored automatically. HSV color stays as a fallback.
- Per object: mask, centroid, footprint (min-area rect length × width, yaw), height `h`, and `hollow` (centre of the top surface lower than the rim).
- Embedding: CLIP ViT-B/32 on the crop (512-d). The same model's text encoder serves the language step.
- **Fix first:** frames returned right after a move are stale (measured: block at v=475 at t+0 s, v=496 at t+0.5 s). Wait ≥ 0.6 s after every move, or check `captured_at` > move end. This is the most likely cause of the servo divergence seen on the first autonomous run.

## 2. Grasp primitives (different objects, different grasps)

A small fixed library. Each primitive has the same parameter set, so learning is just choosing a primitive + tuning a few numbers.

| Primitive | When | How |
|---|---|---|
| `TOP_PINCH` | default: blocks, boxes, small items | top-down, jaws across the short side, fingertips at `top - min(h/2, 25)` |
| `TOP_LOW` | low objects, `h` 10–25 mm | top-down, fingertips ~3 mm above table, slow close |
| `RIM` | hollow: cups, bowls, open boxes | top-down at the rim, one finger inside one outside, yaw radial |
| `SOFT` | sponges, plush | top-down deeper, higher force |
| `SIDE` (stretch) | tall: `h > 1.5·max(l,w)` and `h > 80` mm, e.g. bottles | horizontal approach at `0.6·h` |
| `SKIP` | wider than max opening, `h < 8` mm, transparent/reflective (depth fails) | log and leave it |

Parameters (all relative to the object, never absolute poses): `yaw_offset`, `z_offset`, `open_width`, `force`, `speed`, `lift_height`, plus the grasp point offset for `RIM`.

**Choosing a grasp:**

1. **Known object** (nearest neighbour similarity ≥ τ): candidates are that object's stored grasps. Overnight, choose with Thompson sampling on `Beta(n_ok+1, n_fail+1)` so it keeps exploring. At demo time choose the best mean (greedy).
2. **Unknown object:** geometry rules pick the primitive; params come from the top-3 neighbours' best grasps.
3. **After each attempt:** update `n_try / n_ok`. On failure, add a repaired copy as a new candidate, chosen by the phase where it failed (section 3).

**Success check (all three):** `grab()` result, the gripper's own holding bit (G2 `IsHoldingSomething`), and after the lift the depth at the original spot is back to table height.

## 3. Using the overnight data

For top-down primitives the path is fully determined by (object pose, primitive, params), so storing full waypoint trajectories adds nothing beyond debugging. Record only what the uses below need:

- `est_pose`: object `(x, y, yaw, h)` estimated by vision before the grasp
- `tcp_close`: fingertip position and yaw at the moment the gripper closed
- primitive + params, `success`, `fail_phase`
- crop embedding + images (observe, close, after)

Uses, most valuable first:

1. **Self-calibration from successful grasps.** On a success the object sat between the fingers, so `tcp_close` is ground truth for where it really was. `tcp_close - est_pose` over all successes is the systematic vision error (camera extrinsics, depth, TCP offset). Fit a 2D affine correction by least squares and apply it to every future grasp. A few lines of numpy, and it improves every object at once.
2. **Self-labelled views.** The robot knows which object it put where, so the next observation of that spot is a free labelled view (new position, rotation, lighting). Append it to `views` with `source: overnight`. The library grows from 8 views to hundreds per object, which makes retrieval and the language match more robust.
3. **Grasp success model.** Train `P(success | embedding (PCA to 16-d), geometry, primitive, params)` with logistic regression or gradient boosting on every attempt. For any object, including unseen ones, sample ~50 candidate grasps, score them, and run the best. This generalizes across objects where per-object stats cannot. It is the self-supervised idea of Pinto & Gupta 2016 ("Learning to Grasp from 50K Tries and 700 Robot Hours") at hackathon scale: we will have ~850 attempts, so low-dimensional features only, never raw pixels. Hold out 20% and report accuracy.
4. **Before/after test for the demo.** The overnight curve is confounded (random positions, changing light). Friday morning run a fixed test: 5 objects × 4 placements, once with night-0 defaults and once with calibration + model. That A/B number is the honest claim to make on stage.
5. **Baseline during the night:** per-object Thompson stats + failure-phase repair keep improving the robot while it runs. The model in 3 is trained in the morning (or retrained every ~2 h overnight if time allows).

Kept only for replay/debug, not for decisions: phase timestamps and Viam data capture of the arm joints.

## 4. Atlas data model

```
views     { object_id, label, embedding[512], cam_pose, img_path, source: scan|overnight }            # vector index on embedding, filter: label
objects   { _id, label, aliases[], geometry{l,w,h,hollow},
            grasps: [{ id, primitive, params{...}, n_try, n_ok }] }
attempts  { ts, run_id, mode: overnight|demo|teach, object_id, sim, est_pose{x,y,yaw,h}, embedding[512], grasp_id, primitive, params,
            tcp_close{x,y,z,yaw}, phase_times{...},
            success, fail_phase, fail_reason, img_observe, img_close, img_after, duration_s }
commands  { ts, text, chosen_object, score, margin, corrected_to, success }
```

Several `views` per object (multi-angle), so a query takes the top-k views and votes by label.

## 5. Object library pre-collection (Day 1 afternoon)

`scripts/scan_object.py`: put an object in the centre → the arm visits 8 poses (2 heights × 4 wrist angles) → 8 views embedded → asks for label and aliases → writes `views` + `objects`. About 1 minute per object. Target 15–25 objects: everything the organizers provide plus our own.

## 6. Overnight autonomy (organizers confirmed overnight running is allowed)

Loop: observe → pick a random object in the workspace → choose grasp (Thompson) → execute → success check → **place it at a random free spot in the workspace** (self-reset, not into the square) → log → repeat. 4–6 different objects on the table for variety.

Watchdog:

- Supervisor restarts the process if it crashes (`tmux` + `while true`).
- Stall stop: 10 consecutive failures, or 20 minutes without a success → park at a safe pose and exit.
- Lost objects: nothing detected 3 times in a row → park and exit.
- Viam trigger emails us when the machine goes offline.

Before leaving (by ~8:30 PM): 30 attended minutes with no table hits and no objects lost; collision sensitivity on (config currently `0`, check what that means on xArm); room lights stay on overnight; desk cleared of phones and bottles; one person checks the dashboard and camera remotely every ~2 h (Viam app CONTROL tab has camera and STOP).

Expected volume: ~12 h × (3600 / ~50 s per attempt) ≈ 850 attempts (cycle time not measured yet).

## 7. Language: describe the object, it goes in the blue square

The destination is always the blue square, so there is nothing to parse. No LLM, no templates.

1. Input: a text box (optional mic via the browser Web Speech API) where the user describes **only the object**: "the cup", "something to drink from", "黄色的方块".
2. Ground: CLIP text embedding of `"a photo of " + text` vs. each detected crop on the table → argmax.
3. If the top two are close, highlight both and let the user click the right one. Compare candidates against each other, not against a fixed threshold (text-image scores run lower than image-image).
4. That click is the teach-once step: store the clicked crop's embedding with the typed text as a label/alias, so the same words find it next time.
5. Run the normal grasp pipeline → drop in the blue square; log to `commands`.

## 8. UI (laptop browser)

Live camera with detection boxes and the chosen grasp drawn on top · command box · click-to-correct on the live view · embedded Atlas Charts (success over time, per object, per primitive, language success).

## 9. Timeline

| When | A: robot / executor | B: perception / Atlas | C: language / UI |
|---|---|---|---|
| Thu → 1 PM | fix frame lag; stable `TOP_PINCH` pick-and-place | `align_color_depth`; depth segmentation; CLIP; Atlas collections + indexes | UI skeleton; Charts wired to `attempts` |
| 2–5 PM | `RIM`, `SOFT`, success check; log `est_pose` / `tcp_close` | `scan_object.py`; scan 15–25 objects | text → object grounding |
| 5–8 PM | self-reset loop + watchdog | Thompson selection + stats updates | command flow end-to-end |
| 8–8:45 PM | attended 30-minute test, checklist, start overnight | | |
| Fri 9–10:30 AM | before/after test (5 objects × 4 placements) | fit calibration correction; train success model | dashboard from overnight data + test result |
| 10 AM–12:30 PM | demo hardening | teach-once polish | language polish, mic |
| 12:30–3 PM | feature freeze, rehearse 5×, record a backup video | | |
| 3:30 PM | demo | | |

## Cut lines

- `SIDE` grasp is stretch only.
- Grasp success model → per-object Thompson stats + calibration only.
- Mic → text box only.
- Depth segmentation not working by 1 PM → color-only objects.
- Self-reset loop not stable by 8 PM → run the overnight loop on one known object at a fixed spot, just to get a reliable curve.

## Still to verify on the hardware

Gripper maximum opening · decoding the `image/vnd.viam.dep` depth format · what `collision_sensitivity: 0` means on xArm · lights overnight · real cycle time per attempt.
