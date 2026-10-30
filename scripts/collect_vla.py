"""
Collect VLA training episodes unattended: the team's pick-and-place code is the "expert", the recorder
(sorter/vla_recorder.py) writes every command as one episode. Convert the result with rlds/sorter_vla/.

Loop: pick a command -> record "put the <object> in the blue square" (executor.execute) -> if something went into
the square, record "take everything out of the blue square" (executor.clear_square, which scatters the objects to
random free spots, so the next episode starts from a new layout). Every episode is kept with its outcome
(success / not_found / failed ...) in meta.json; filter at conversion time, not here.

Usage (from the repo root, on the machine next to the arm, someone at the e-stop):
  python scripts/collect_vla.py --episodes 20 --objects "yellow block" "green cube" "blue cylinder"
  python scripts/collect_vla.py --episodes 50                 # objects: every label in the object library
Options: --out dataset_raw  --hz 10  --no-depth  --no-reset-episodes  --session NAME
Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY (MONGODB_URI optional, as for every script).

Do not send the sorter module commands (dashboard / CONTROL tab) while this runs: both would drive the arm.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GRASP_TEMPLATES = ["put the {o} in the blue square", "pick up the {o} and place it in the square",
                   "move the {o} into the blue square", "place the {o} inside the square"]
RESET_INSTRUCTION = "take everything out of the blue square"
SUMMARY_KEYS = ("status", "success", "fail_phase", "fail_reason", "label", "duration_s")


def instruction_for(obj: str, rng: random.Random) -> str:
    """A full-sentence instruction; the paraphrases keep a policy from overfitting to one phrasing."""
    return rng.choice(GRASP_TEMPLATES).format(o=obj)


async def record(rec, instruction: str, kind: str, run, **meta) -> dict:
    """One episode around `run()` (an executor call returning its attempt doc); never raises."""
    path = await rec.start_episode(instruction, kind=kind, **meta)
    doc = {}
    try:
        doc = await run() or {}
    except Exception as e:
        doc = {"status": "error", "success": False, "fail_reason": f"{type(e).__name__}: {e}"}
    finally:
        await rec.end_episode(doc)
    summary = {"episode": path.name, "kind": kind, "instruction": instruction, **{k: doc.get(k) for k in SUMMARY_KEYS}}
    with (path.parent / "episodes.jsonl").open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    print(f"  episode {path.name}: {summary['status']} ({rec_steps(path)} steps)")
    return doc


def rec_steps(path: Path) -> int:
    try:
        return json.loads((path / "meta.json").read_text())["n_steps"]
    except Exception:
        return 0


async def main(args) -> None:
    import pick_place_auto as ppa
    from sorter import executor
    from sorter.store import Store
    from sorter.vla_recorder import VLARecorder

    rng = random.Random(args.seed)
    store = Store()
    objects = args.objects or [names[0] for names in store.list_labels()]
    if not objects:
        sys.exit("no objects: pass --objects or build the object library first")
    print(f"objects: {', '.join(objects)}")

    rig = ppa.Rig()
    await rig.start()
    rec = VLARecorder(rig, args.out, hz=args.hz, save_depth=not args.no_depth, session=args.session)
    rec.attach()
    try:
        await executor.apply_limits(rig)
        await executor.resume(rig, store)
        await executor.clear_square(rig, store)  # start with an empty square (not recorded)
        for i in range(1, args.episodes + 1):
            obj = rng.choice(objects)
            text = instruction_for(obj, rng)
            print(f"=== episode {i}/{args.episodes}: {text} ===")
            doc = await record(rec, text, "grasp", lambda: executor.execute(rig, obj, store, mode="vla"),
                               target_object=obj, command=obj)
            if doc.get("status") in ("done", "failed", "error"):
                # Something may be in the square now: empty it, recorded as its own task unless turned off
                if args.no_reset_episodes:
                    await executor.clear_square(rig, store)
                else:
                    await record(rec, RESET_INSTRUCTION, "reset", lambda: executor.clear_square(rig, store))
            if ppa.ESTOP["on"]:
                print("emergency stop: ending the collection")
                break
    finally:
        if rec._ep is not None:
            await rec.end_episode({"status": "interrupted", "success": False})
        rec.detach()
        rec.close()
        await rig.close()
    print(f"done: episodes in {Path(args.out) / rec.session}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=10, help="grasp episodes to record (resets come extra)")
    ap.add_argument("--objects", nargs="*", help="object names to ask for (default: the object library's labels)")
    ap.add_argument("--out", default=str(ROOT / "dataset_raw"))
    ap.add_argument("--session", help="folder name for this run (default: start time)")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--no-depth", action="store_true", help="do not save depth images")
    ap.add_argument("--no-reset-episodes", action="store_true", help="still reset, but do not record it")
    ap.add_argument("--seed", type=int, default=None)
    asyncio.run(main(ap.parse_args()))
