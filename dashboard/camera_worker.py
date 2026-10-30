"""
Robot-link subprocess for the dashboard: the module bridge (dashboard/camera.py), mirrored into files in <out_dir>.

WHY: a native abort inside the Viam SDK kills the whole process; isolated here it only kills this link, and
dashboard/app.py restarts it while the web page keeps working.

Env (set by app.py): DASH_MODULE_NAME=<generic service name>.
Usage (started by app.py): python -m dashboard.camera_worker <out_dir>
"""
import asyncio
import os
import sys
from pathlib import Path

from dashboard.camera import Link, ModuleBridge

DUMP_PERIOD_S = 0.5


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    link = Link()
    task = asyncio.create_task(link.run(ModuleBridge(out, os.environ.get("DASH_MODULE_NAME") or "sorter")))
    try:
        while not task.done():
            link.dump(out)  # link.json doubles as this process's heartbeat
            await asyncio.sleep(DUMP_PERIOD_S)
        task.result()
    finally:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
