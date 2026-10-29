"""
Simplest pick-and-place MVP for sfhack4 (xArm6 + ufactory gripper), no vision.

WHY: prove the hardware path (SDK connect -> arm -> gripper) end to end before adding
embeddings / Atlas. Poses are taught by hand once, then replayed.

Flow: `record pick` / `record place` (jog arm in the app's CONTROL tab first) -> `run`.
Machine config read from the Viam app on 2026-10-29: arm=ufactory:xArm6 host 192.168.1.210,
gripper=ufactory:gripper, cam=realsense:realsense, table obstacle z=-123 mm.
{VIAM CONFIG sfhack4 "UFACTORY/XARM6" "UFACTORY/GRIPPER" "TABLE-TRANSLATION": {"Z": -123}}
[CONFIDENCE: CONFIRMED 90% — read from the app UI; SDK method names below are from memory, untested on hardware]
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose
from viam.robot.client import RobotClient

# Taught poses live next to the script and are gitignored (they are specific to this table setup)
POSES_FILE = Path(__file__).with_name("poses.json")
# Approach/retreat height above a taught pose; keeps the gripper clear of the object and table when moving sideways
HOVER_MM = 100.0


async def connect() -> RobotClient:
    """
    Open an SDK connection to the machine.

    WHY: credentials come from env vars so no API key is ever written into the repo.
    Address and key are copied by the user from the machine's CONNECT tab.
    """
    # api_key / api_key_id are the pair shown in the CONNECT tab's code sample
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


def to_dict(p: Pose) -> dict:
    """Serialize a Pose to JSON-able fields so taught poses survive between runs."""
    return {k: getattr(p, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def hover(p: dict) -> Pose:
    """Same pose raised by HOVER_MM in z, used for approach and retreat."""
    return Pose(**{**p, "z": p["z"] + HOVER_MM})


async def record(name: str) -> None:
    """
    Save the arm's current end position under `name` ("pick" or "place").

    WHY: teaching by hand avoids needing hand-eye calibration for the first MVP.
    """
    robot = await connect()
    try:
        arm = Arm.from_robot(robot, "arm")
        # Merge into the existing file so recording "place" does not erase "pick"
        poses = json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else {}
        poses[name] = to_dict(await arm.get_end_position())
        POSES_FILE.write_text(json.dumps(poses, indent=2))
        print(name, poses[name])
    finally:
        # Always release the connection, even if the arm call raised
        await robot.close()


async def run() -> None:
    """
    Replay the taught pick and place.

    Before: arm anywhere. After: object moved from pick to place, gripper open, arm hovering.
    Aborts before moving to place if the gripper reports it grabbed nothing.
    """
    poses = json.loads(POSES_FILE.read_text())
    robot = await connect()
    try:
        arm = Arm.from_robot(robot, "arm")
        gripper = Gripper.from_robot(robot, "gripper")

        # Open first so the jaws are clear before descending onto the object
        await gripper.open()
        await arm.move_to_position(hover(poses["pick"]))
        await arm.move_to_position(Pose(**poses["pick"]))
        # grab() returns whether the jaws closed on something; used as the success check
        grabbed = await gripper.grab()
        print("grab returned", grabbed)
        # Lift straight up before any sideways motion so the object does not drag on the table
        await arm.move_to_position(hover(poses["pick"]))
        if not grabbed:
            # Do not carry an empty gripper to the place spot; release and stop
            await gripper.open()
            sys.exit("nothing in gripper, aborting before place")

        await arm.move_to_position(hover(poses["place"]))
        await arm.move_to_position(Pose(**poses["place"]))
        await gripper.open()
        await arm.move_to_position(hover(poses["place"]))
    finally:
        await robot.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "record":
        asyncio.run(record(sys.argv[2]))
    else:
        asyncio.run(run())
