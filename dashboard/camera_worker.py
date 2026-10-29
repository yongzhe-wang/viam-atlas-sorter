"""
Robot-link subprocess for the dashboard: camera feed and/or the module bridge, mirrored into files in <out_dir>.

WHY: a native abort inside the Viam SDK kills the whole process; isolated here it only kills this link, and
dashboard/app.py restarts it while the web page keeps working.

Env (set by app.py): DASH_CAMERA=1|0, DASH_MODULE_NAME=<generic service name> (empty = no module bridge).
Usage (started by app.py): python -m dashboard.camera_worker <out_dir>
"""
import asyncio
import os
import sys
from pathlib import Path

from dashboard.camera import CameraFeed, ModuleBridge

DUMP_PERIOD_S = 0.5


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    camera_on = os.environ.get("DASH_CAMERA", "1") != "0"
    module_name = os.environ.get("DASH_MODULE_NAME", "")
    feed = CameraFeed()
    bridge = ModuleBridge(out, module_name) if module_name else None
    task = asyncio.create_task(feed.run(camera_on=camera_on, bridge=bridge))
    try:
        while not task.done():
            if camera_on:
                feed.read_highlight(out)
            # state.json doubles as this process's heartbeat, camera or not
            feed.dump(out)
            await asyncio.sleep(DUMP_PERIOD_S)
        task.result()
    finally:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
