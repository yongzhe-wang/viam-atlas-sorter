"""
Hand the arm over: gripper empty, parked at the observe pose (HOME_XY, ALIGN_Z, theta 0).

WHY: before another program (the Viam module) takes over the arm, nothing may be left in the gripper and
the arm must sit where that program expects its first frame. A held object is put down at a verified
free spot (sorter.executor.drop_somewhere), never just dropped.
Usage: python scripts/park_empty.py
"""
import asyncio
import sys
from pathlib import Path

import pick_place_auto as ppa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorter import executor  # noqa: E402


async def main() -> None:
    rig = ppa.Rig()
    await rig.start()
    try:
        await executor.apply_limits(rig)
        if await rig.holding():
            st = ppa.load_state()
            xy = await executor.drop_somewhere(rig, st.get("height") or 30.0, st.get("grip") or 15.0)
            print(f"put the held object down at ({xy[0]:.0f},{xy[1]:.0f})")
        else:
            await rig.open()
            ppa.save_state(False)
        await ppa.travel(rig, *ppa.HOME_XY, theta=0.0)
        p = await rig.pose()
        print(f"PARKED x={p.x:.0f} y={p.y:.0f} z={p.z:.0f} theta={p.theta:.1f} holding={await rig.holding()}")
    finally:
        await rig.close()


if __name__ == "__main__":
    asyncio.run(main())
