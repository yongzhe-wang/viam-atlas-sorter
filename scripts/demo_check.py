"""
Run demo commands through sorter.executor exactly as the worker/module would, clearing the square after each.

WHY: a quick end-to-end check of executor changes on the real arm ("does yellow block / green cube still
pass?") without the dashboard. Logs to the Store like any attempt.
Usage: python scripts/demo_check.py "yellow block" "green cube"
"""
import asyncio
import sys
from pathlib import Path

import pick_place_auto as ppa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import executor  # noqa: E402
from sorter.store import Store  # noqa: E402


async def main(texts: list) -> None:
    store = Store()
    rig = ppa.Rig()
    await rig.start()
    try:
        await executor.apply_limits(rig)
        await executor.resume(rig, store)
        for text in texts:
            doc = await executor.execute(rig, text, store, mode="demo")
            print(f"RESULT {text!r}: {doc['status']} {doc.get('duration_s')} s find={doc.get('find_path')} "
                  f"square={doc.get('square_path')} label={doc.get('label')} phases={doc.get('phase_times')} "
                  f"{doc.get('fail_reason') or ''}")
            clear = await executor.clear_square(rig, store)
            print(f"RESET: {clear['status']} {clear.get('duration_s')} s removed={clear.get('removed')} "
                  f"{clear.get('fail_reason') or ''}")
    finally:
        await rig.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
