# Demo queries (rehearsal list)

Typed into the dashboard; the arm picks the object and drops it in the blue square. The worker uses
`clip_embed.best()`, which picks only an object the scan library recognises as what was named, and refuses
otherwise.

| # | Query | Expected | Checked on arm-scan frames (score / lead over runner-up) |
|---|---|---|---|
| 1 | `grasp green square` | green cube (not the green cylinder next to it) | 3/3, 0.52–0.53 / 0.33–0.37 |
| 2 | `pick up the yellow block` | yellow block | 4/4, ≥ 0.37 lead (frames where it is fully in view) |
| 3 | `trick or treat: candy` | KIND bar ("candy" is an alias) | 3/3, 0.37–0.38 / 0.24–0.30 |
| 4 | `the blue cylinder` | blue cylinder | 3/3, 0.47–0.51 / 0.25–0.28 |
| 5 | `red ball` | refuses: "no red object found" | 3/3 refused |

Each frame was tested with its own views removed from the library (the library comes from the scan, the
demo frame is new). Re-check after any library change: `python scripts/eval_perception.py demo`
(16/16 on 2026-10-29).

Before the run:

- The target must be fully inside the camera view at the observe pose (250, 0, 400). Anything cut by the
  image edge is excluded on purpose.
- Keep objects at least 3 cm apart. Objects that touch are segmented as one (the blue-handle screwdriver
  touching the blue block did this).
- No transparent objects as targets: the depth camera cannot see the measuring cups.
- A refusal is the safe outcome, not a failure. The dashboard shows the reason, e.g. "no 'candy' recognised
  on the table" if the KIND bar is out of view.
