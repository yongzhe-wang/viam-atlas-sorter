"""
Autonomous pick-and-place for sfhack4: find the yellow block, grab it, drop it in the blue tape square.

WHY: no Claude (or any human) in the control loop. One command runs the whole sequence; the camera
closes the loop on the block and on the square, so objects need not be at taught positions.

Flow: observe pose -> servo block onto the gripper axis -> rotate wrist so jaws close across the
short side -> straight descent -> grab -> lift -> find blue square -> servo onto it -> lower -> release -> home.

Geometry read from the Viam app on 2026-10-29:
  cam     frame: parent arm, translation (-83, 14, 18) mm, rotated -97.73 deg about the tool axis
  gripper frame: parent arm, translation (0, 0, 105) mm  (fingertip centre is 105 mm below the flange)
  table obstacle z = -123 mm; arm.get_end_position() reports the flange (empirical image scale matched this)
{VIAM CONFIG sfhack4 "CAM FRAME: PARENT ARM, TRANSLATION X -83 Y 14 Z 18, TH -97.731173" "GRIPPER FRAME: TRANSLATION Z 105"}
[CONFIDENCE: CONFIRMED 85% — read from the app UI; arm motion, jaw axis and final heights are UNTESTED on hardware]

Env: VIAM_MACHINE_ADDRESS, VIAM_API_KEY_ID, VIAM_API_KEY. Optional: GRASP_Z, PLACE_Z, BLOCK_H.
Run it with a hand near the e-stop. Ctrl-C stops the script; the arm finishes its current move.
"""
import asyncio
import io
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose
from viam.robot.client import RobotClient

TABLE_Z = -123.0  # table top in the arm base frame (mm)
GRIPPER_LEN = 105.0  # flange -> fingertip centre along the tool axis (mm)
CAM_T = (-83.0, 14.0, 18.0)  # camera origin in the flange frame (mm)
CAM_YAW_DEG = -97.731173  # camera rotation about the tool axis
BLOCK_H = float(os.environ.get("BLOCK_H", 35.0))  # block height guess (mm); only used for depth/parallax
# Flange z at grasp: fingertip centre at about half the block height. Tune with GRASP_Z.
GRASP_Z = float(os.environ.get("GRASP_Z", TABLE_Z + BLOCK_H / 2 + GRIPPER_LEN))
PLACE_Z = float(os.environ.get("PLACE_Z", GRASP_Z + 25.0))
# Hard floor for the flange: fingertips must stay a margin above the table no matter what
MIN_Z = TABLE_Z + GRIPPER_LEN + 8.0
ALIGN_Z = 400.0  # observation height; verified to see the whole table scene
HOME_XY = (250.0, 0.0)
MAX_STEP_MM = 250.0
TOL_PX = 7.0


def axis_pixel(fx: float, fy: float, cx: float, cy: float, depth: float) -> tuple:
    """
    Pixel where the gripper (flange) axis pierces a plane `depth` mm in front of the camera.

    WHY: the camera sits beside the gripper, so the point under the fingers is not the image centre.
    Offset of the flange axis from the camera axis, in camera coordinates, scaled by f/depth (pinhole).
    """
    ty = math.radians(CAM_YAW_DEG)
    # Flange origin relative to the camera, expressed in the flange frame
    dx, dy = -CAM_T[0], -CAM_T[1]
    # Camera x (image right) and y (image down) axes in the flange frame
    xc, yc = (math.cos(ty), math.sin(ty)), (-math.sin(ty), math.cos(ty))
    au = dx * xc[0] + dy * xc[1]
    av = dx * yc[0] + dy * yc[1]
    return cx + fx * au / depth, cy + fy * av / depth


def decode_rgb(images) -> np.ndarray:
    """Pick the colour JPEG out of a get_images() result (the list also holds the depth map)."""
    for im in images:
        if im.mime_type == "image/jpeg":
            return np.asarray(Image.open(io.BytesIO(im.data)).convert("RGB"))
    raise RuntimeError("no colour image returned")


def find_block(rgb: np.ndarray):
    """
    Centroid (u, v), long-axis angle and area of the largest yellow blob, or None.

    HSV thresholds measured on this table: block H~39 S~150 V~170, table S~28, orange phone H~15.
    Long axis angle is degrees ccw from image +u, folded into [0, 180).
    """
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 32) & (h <= 48) & (s > 100) & (v > 110)
    lab, n = ndimage.label(mask)
    if not n:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    if sizes[k - 1] < 1500:  # ignore specks
        return None
    ys, xs = np.nonzero(lab == k)
    cx, cy = xs.mean(), ys.mean()
    w, vec = np.linalg.eigh(np.cov(np.vstack([xs - cx, ys - cy])))
    ang = math.degrees(math.atan2(-vec[1, 1], vec[0, 1])) % 180.0
    return cx, cy, ang, len(xs)


def find_square(rgb: np.ndarray):
    """
    Centroid (u, v) of the hole enclosed by the blue tape, or None if the ring is not fully in view.

    The hole is the largest non-blue region that does not touch the image border.
    """
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    blue = ndimage.binary_closing((b > 110) & (b - r > 60) & (b - g > 20), iterations=3)
    lab, n = ndimage.label(~blue)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    best = None
    for i in range(1, n + 1):
        if i in border:
            continue
        ys, xs = np.nonzero(lab == i)
        if len(xs) > 2000 and (best is None or len(xs) > best[2]):
            best = (xs.mean(), ys.mean(), len(xs))
    return best


class Rig:
    """Holds the connection and wraps motion with the safety limits."""

    def __init__(self, robot, arm, cam, gripper, intr):
        self.robot, self.arm, self.cam, self.gripper = robot, arm, cam, gripper
        self.fx, self.fy, self.cx, self.cy = intr.focal_x_px, intr.focal_y_px, intr.center_x_px, intr.center_y_px
        self.theta = 0.0

    async def pose(self):
        return await self.arm.get_end_position()

    async def goto(self, x: float, y: float, z: float, theta: float = None) -> None:
        """
        Move the flange to (x, y, z) with the tool pointing straight down.

        WHY: single choke point for limits. Refuses anything below MIN_Z, outside reach, or a jump larger
        than MAX_STEP_MM, so a bad calculation raises instead of driving the gripper into the table.
        """
        if z < MIN_Z:
            raise RuntimeError(f"refusing z={z:.0f} < MIN_Z={MIN_Z:.0f}")
        if math.hypot(x, y) > 650 or math.hypot(x, y) < 120:
            raise RuntimeError(f"refusing xy=({x:.0f},{y:.0f}) outside safe reach")
        p = await self.pose()
        if math.dist((p.x, p.y, p.z), (x, y, z)) > MAX_STEP_MM:
            raise RuntimeError("refusing step larger than MAX_STEP_MM")
        if theta is not None:
            self.theta = theta
        await self.arm.move_to_position(Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=self.theta))

    async def rgb(self) -> np.ndarray:
        res = await self.cam.get_images()
        return decode_rgb(res[0] if isinstance(res, tuple) else res)

    async def lift(self) -> None:
        """Straight up to ALIGN_Z; the safe response to any failure."""
        p = await self.pose()
        await self.goto(p.x, p.y, ALIGN_Z)

    async def servo(self, finder, plane_z: float, label: str) -> None:
        """
        Bring the target found by `finder` onto the gripper axis, closed loop.

        Before: target anywhere in view. After: its pixel is within TOL_PX of where the gripper axis
        projects for a plane at `plane_z`. Each round probes two small moves to measure the pixel
        Jacobian (so axis/sign conventions never have to be assumed), then solves for the correction.
        """
        for it in range(6):
            p = await self.pose()
            depth = (p.z - CAM_T[2]) - plane_z
            tu, tv = axis_pixel(self.fx, self.fy, self.cx, self.cy, depth)
            hit = finder(await self.rgb())
            if hit is None:
                raise RuntimeError(f"{label}: not visible")
            e = np.array([hit[0] - tu, hit[1] - tv])
            print(f"  {label} iter {it}: target px=({hit[0]:.0f},{hit[1]:.0f}) want=({tu:.0f},{tv:.0f}) err={np.linalg.norm(e):.0f}px")
            if np.linalg.norm(e) < TOL_PX:
                return
            # Probe +PROBE mm along arm x then y, record how the target moves in the image
            probe = 15.0
            cols = []
            for dx, dy in ((probe, 0.0), (0.0, probe)):
                await self.goto(p.x + dx, p.y + dy, p.z)
                h2 = finder(await self.rgb())
                if h2 is None:
                    raise RuntimeError(f"{label}: lost during probe")
                cols.append([(h2[0] - hit[0]) / probe, (h2[1] - hit[1]) / probe])
                await self.goto(p.x, p.y, p.z)
            J = np.array(cols).T  # px per mm, columns = arm x, arm y
            # Arm move that cancels the pixel error (damped); the image moves by J @ move
            move = np.linalg.solve(J, -e) * 0.85
            n = np.linalg.norm(move)
            if n > 120:
                move *= 120 / n
            await self.goto(p.x + move[0], p.y + move[1], p.z)
        raise RuntimeError(f"{label}: did not converge")

    async def align_wrist(self) -> None:
        """
        Rotate the wrist so the block's long axis is vertical in the image (jaws close along image u).

        WHY: the finger axis was seen along image u, so the jaws then clamp across the block's short side.
        The sign of theta is not assumed: try one direction, keep it only if the angle error shrank.
        """
        hit = find_block(await self.rgb())
        err = ((90.0 - hit[2] + 90.0) % 180.0) - 90.0  # signed distance to vertical, in [-90, 90)
        if abs(err) < 4:
            return
        p = await self.pose()
        await self.goto(p.x, p.y, p.z, theta=self.theta + err)
        err2 = ((90.0 - find_block(await self.rgb())[2] + 90.0) % 180.0) - 90.0
        print(f"  wrist: angle err {err:.0f} -> {err2:.0f}")
        if abs(err2) > abs(err) * 0.5:
            await self.goto(p.x, p.y, p.z, theta=self.theta - 2 * err)


async def run() -> None:
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
    )
    robot = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    try:
        cam = Camera.from_robot(robot, "cam")
        intr = (await cam.get_properties()).intrinsic_parameters
        rig = Rig(robot, Arm.from_robot(robot, "arm"), cam, Gripper.from_robot(robot, "gripper"), intr)
        print(f"GRASP_Z={GRASP_Z:.0f} PLACE_Z={PLACE_Z:.0f} MIN_Z={MIN_Z:.0f}")
        try:
            await rig.gripper.open()
            p = await rig.pose()
            await rig.goto(p.x, p.y, ALIGN_Z, theta=0.0)

            print("1/5 find + align over block")
            await rig.servo(find_block, TABLE_Z + BLOCK_H, "block")
            await rig.align_wrist()
            await rig.servo(find_block, TABLE_Z + BLOCK_H, "block")

            print("2/5 descend + grab")
            p = await rig.pose()
            # Descend in 60 mm steps so a bad GRASP_Z can be interrupted before the last step
            z = p.z
            while z > GRASP_Z:
                z = max(GRASP_Z, z - 60.0)
                await rig.goto(p.x, p.y, z)
            grabbed = await rig.gripper.grab()
            print("  grab returned", grabbed)
            await rig.goto(p.x, p.y, ALIGN_Z)
            if not grabbed:
                await rig.gripper.open()
                sys.exit("nothing in gripper; stopped before moving to the square")

            print("3/5 find blue square")
            for _ in range(4):
                if find_square(await rig.rgb()):
                    break
                q = await rig.pose()
                await rig.goto(q.x, q.y + 100.0, ALIGN_Z)  # square was on the +y side of the block
            await rig.servo(lambda im: find_square(im), TABLE_Z, "square")

            print("4/5 lower + release")
            q = await rig.pose()
            await rig.goto(q.x, q.y, PLACE_Z)
            await rig.gripper.open()
            await rig.goto(q.x, q.y, ALIGN_Z)

            print("5/5 home")
            await rig.goto(HOME_XY[0], HOME_XY[1], ALIGN_Z, theta=0.0)
            print("done")
        except Exception:
            # Any failure: go straight up so nothing drags across the table, then surface the error
            try:
                await rig.lift()
            finally:
                raise
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(run())
