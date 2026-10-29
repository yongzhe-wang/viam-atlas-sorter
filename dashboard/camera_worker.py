"""
Camera subprocess for the dashboard: runs CameraFeed and mirrors its state into files in <out_dir>.

WHY: a native abort inside the Viam SDK kills the whole process; isolated here it only kills the camera,
and dashboard/app.py restarts this process while the web page keeps working.

Usage (started by app.py): python -m dashboard.camera_worker <out_dir>
"""
import asyncio
import sys
from pathlib import Path

from dashboard.camera import CameraFeed

DUMP_PERIOD_S = 0.5


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    feed = CameraFeed()
    task = asyncio.create_task(feed.run())
    try:
        while not task.done():
            feed.read_highlight(out)
            feed.dump(out)
            await asyncio.sleep(DUMP_PERIOD_S)
        task.result()
    finally:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
