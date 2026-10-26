"""Private evaluator check: did a drawer front enter the onboard camera's view?

Geometry only, from the measured torso pose and the calibrated camera mount; it
is never a model input. Occluders are axis-aligned scene boxes taller than the
camera, which hide anything behind them in plan view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.robot_camera import IMAGE_HEIGHT, IMAGE_WIDTH, calibration

MAXIMUM_VIEW_RANGE_M = 3.5
MINIMUM_VIEW_DEPTH_M = 0.25
# The camera must be on the drawer's front side, within about 70 degrees of its normal.
MINIMUM_FRONT_FACING_COSINE = 0.35


@dataclass(frozen=True)
class PlanOccluder:
    center_xy_m: tuple[float, float]
    half_size_xy_m: tuple[float, float]
    top_z_m: float


def segment_crosses_rectangle(start: np.ndarray, end: np.ndarray, occluder: PlanOccluder) -> bool:
    """Slab test of a 2D segment against an axis-aligned rectangle."""
    low = np.asarray(occluder.center_xy_m) - occluder.half_size_xy_m
    high = np.asarray(occluder.center_xy_m) + occluder.half_size_xy_m
    direction = end - start
    entry, exit_ = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if not low[axis] <= start[axis] <= high[axis]:
                return False
            continue
        first = (low[axis] - start[axis]) / direction[axis]
        second = (high[axis] - start[axis]) / direction[axis]
        entry = max(entry, min(first, second))
        exit_ = min(exit_, max(first, second))
        if entry > exit_:
            return False
    return True


def front_in_view(
    torso_position_m: np.ndarray,
    torso_quaternion_xyzw: np.ndarray,
    front_center_m: np.ndarray,
    front_normal_xy: np.ndarray,
    occluders: list[PlanOccluder],
) -> bool:
    camera = calibration()
    torso_rotation = Rotation.from_quat(torso_quaternion_xyzw)
    eye = np.asarray(torso_position_m) + torso_rotation.apply(camera["camera_position_torso_m"])
    optical_to_world = torso_rotation.as_matrix() @ np.asarray(camera["optical_to_torso_rotation"])
    point = optical_to_world.T @ (np.asarray(front_center_m) - eye)
    if not MINIMUM_VIEW_DEPTH_M <= point[2] <= MAXIMUM_VIEW_RANGE_M:
        return False
    fx, fy, cx, cy = camera["intrinsics_fx_fy_cx_cy"]
    column = fx * point[0] / point[2] + cx
    row = fy * point[1] / point[2] + cy
    if not (0 <= column < IMAGE_WIDTH and 0 <= row < IMAGE_HEIGHT):
        return False
    to_eye = eye[:2] - np.asarray(front_center_m)[:2]
    distance = float(np.linalg.norm(to_eye))
    if distance <= 0 or float(np.dot(front_normal_xy, to_eye)) / distance < (
        MINIMUM_FRONT_FACING_COSINE
    ):
        return False
    return not any(
        occluder.top_z_m > eye[2]
        and segment_crosses_rectangle(eye[:2], np.asarray(front_center_m)[:2], occluder)
        for occluder in occluders
    )
