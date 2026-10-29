"""
Command worker: takes dashboard commands from the Store and runs them on the arm, one at a time.

WHY: the dashboard only queues text ("grasp green square", or "__reset__" from the Clear square button);
this process is the single place that moves the arm, so two commands can never drive it at once.
Runs next to the dashboard on the same machine until MONGODB_URI is set (offline Store is per machine).

Usage: python -m sorter.worker      (env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY, MONGODB_URI)
"""
import asyncio
import re

from sorter import executor
from sorter.store import Store

VERBS = re.compile(r"^\s*(grasp|grab|pick( up)?|get|take|move|put|place)\s+", re.IGNORECASE)


def clean(text: str) -> str:
    """Strip a leading verb; the destination is always the blue square, so the rest names the object."""
    return VERBS.sub("", text or "").strip()


async def main() -> None:
    store = Store()
    rig = executor.ppa.Rig()
    await rig.start()
    print(f"worker: connected, store {'online' if store.online else 'offline (JSONL)'}; waiting for commands")
    try:
        while True:
            cmd = store.next_command()
            if cmd is None:
                await asyncio.sleep(1.0)
                continue
            text = (cmd.get("text") or "").strip()
            print(f"worker: command {cmd['_id']}: {text!r}")
            try:
                if text == "__reset__":
                    doc = await executor.clear_square(rig, store)
                else:
                    doc = await executor.execute(rig, clean(text), store)
                store.finish_command(cmd["_id"], doc["status"], doc.get("_id"), doc.get("label"))
                print(f"worker: {doc['status']} in {doc.get('duration_s')}s {doc.get('fail_reason') or ''}")
            except Exception as e:
                # Executor handles ordinary failures; this is the last line of defence
                store.finish_command(cmd["_id"], "failed")
                print(f"worker: command crashed: {type(e).__name__}: {e}")
                try:
                    await rig.lift()
                except Exception:
                    pass
    finally:
        await rig.close()


if __name__ == "__main__":
    asyncio.run(main())
