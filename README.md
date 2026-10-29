# viam-atlas-sorter

Teach-once recycling sorter for the Viam x MongoDB Robot Hackathon (Oct 29-30, 2026, Robotics Center of Silicon Valley).

## Idea

Wrist camera sees an object -> embed it -> MongoDB Atlas `$vectorSearch` finds the nearest known objects -> read back the bin and the grasp parameters that worked before -> arm picks and places -> the result is written to both Viam data capture and Atlas.

Unknown object (similarity below threshold): arm stops, a human taps the correct bin once, the embedding is stored, and the next time it is recognized.

## Division of labor

| Concern | Viam | Atlas |
|---|---|---|
| Hardware | arm, gripper, camera as components | - |
| Perception | vision service (mlmodel / color_detector), OpenCV segmentation | - |
| Embedding | custom Python module `sorter` | stores vectors, Vector Search index |
| Memory | - | `objects`: embedding, bin, grasp params, n_try, n_ok |
| Run log | data capture on the `sorter` sensor, synced to Viam cloud | `attempts`: append-only ledger |
| Analysis | Viam Query (MQL) | Atlas Charts dashboard |

## Data model

```
objects   { _id, label, bin, embedding[512], grasp{width, force, approach_height, yaw_offset},
            n_try, n_ok, taught_by, created_at }
bins      { _id: "recycle|trash|compost", pregrasp_pose, drop_pose }
attempts  { _id, attempt_id, ts, object_id, sim, k_used[], grasp_used{},
            success, duration_s, img_before, img_after, error }
```

Grasp parameters are stored as relative values only. Trajectories depend on where the object is and are recomputed from the current pose every time.

## Runtime loop

1. Move to the fixed observe pose.
2. OpenCV segmentation: centroid, long-axis angle, crop.
3. CLIP embedding of the crop (512-d, normalized).
4. Top-3 neighbors from the local numpy cache (Atlas is the source of truth; cache refreshed periodically and after each teach).
5. `sim >= tau` (start at 0.85, calibrate on site): use the neighbor's bin. Otherwise: teach flow.
6. Choose grasp params from neighbors weighted by `n_ok / max(n_try, 1)`.
7. Pixel -> table (x, y) via homography; yaw from long axis + `yaw_offset`.
8. Pre-grasp, re-detect, descend, close, lift, drop at the bin, open.
9. Judge success, write `attempts`, update `n_try` / `n_ok`, flush the local queue to Atlas when online.

## Network resilience

Run the loop on the Linux machine next to the arm and connect over LAN. Keep the embeddings in a local cache and queue writes locally; flush to Atlas when the connection returns.

## Schedule

| When | Must have | Cut line |
|---|---|---|
| Day 1 AM | machine online, arm controllable, homography calibrated | - |
| Day 1 PM | fixed-position pick and place | no pick by 5 PM -> point at bins only |
| Day 1 evening (hard close 9 PM) | embedding + Atlas write/search | use `input()` instead of teach UI |
| Day 2 AM | any-position pick, teach-once, `attempts` logging | parameter exploration -> record only |
| Day 2 PM | Charts, rehearse with 5 objects | no new features |
| 3:30 PM | Demo | - |

## Unverified items (check on site)

- Atlas `$vectorSearch` pre-filter syntax and index definition format
- Gripper "is holding" API
- `motion.Move` frame-system configuration
- Python SDK method names (`Camera.from_robot`, `arm.move_to_joint_positions`)

## Layout

```
sorter/     Viam module (embedding, control loop)
scripts/    Atlas setup, seed data, homography calibration
teach_ui/   minimal web page for teaching new objects
docs/       notes
```
