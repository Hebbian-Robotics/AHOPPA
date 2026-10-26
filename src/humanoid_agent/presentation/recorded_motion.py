"""Project recorded references onto observer video without changing model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from humanoid_agent.validated_types import UnitQuaternion


class ObserverCalibration(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    optical_quaternion_world_wxyz: UnitQuaternion | None = None
    eye_world_m: list[float] = Field(min_length=3, max_length=3)
    look_at_world_m: list[float] = Field(min_length=3, max_length=3)
    focal_length_pixels: float = Field(gt=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    composite_header_pixels: int = Field(ge=0)


def project_world_points(
    calibration: ObserverCalibration, points: NDArray[np.float64]
) -> NDArray[np.float64]:
    if calibration.optical_quaternion_world_wxyz is not None:
        camera_points = (
            Rotation.from_quat(calibration.optical_quaternion_world_wxyz, scalar_first=True)
            .inv()
            .apply(points - np.asarray(calibration.eye_world_m))
        )
    else:
        forward = np.asarray(calibration.look_at_world_m) - np.asarray(calibration.eye_world_m)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        camera_points = (points - np.asarray(calibration.eye_world_m)) @ np.column_stack(
            [right, down, forward]
        )
    if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 0):
        raise ValueError("Overlay points must be finite and in front of the observer camera")
    return camera_points[:, :2] / camera_points[:, 2:3] * calibration.focal_length_pixels + [
        calibration.image_width / 2,
        calibration.image_height / 2 + calibration.composite_header_pixels,
    ]


def observer_calibration_from_capture(capture: dict) -> ObserverCalibration:
    audit = capture["private_observer_camera_audit"]
    matrix = np.asarray(audit["intrinsic_matrix"])
    width, height = audit["resolution_width_height"]
    if not np.allclose(matrix[[0, 1], [2, 2]], [width / 2, height / 2]) or not np.isclose(
        matrix[0, 0], matrix[1, 1]
    ):
        raise ValueError("Composite requires centered, square-pixel observer intrinsics")
    rotation = Rotation.from_quat(audit["optical_quaternion_ros_wxyz"], scalar_first=True)
    eye = np.asarray(audit["optical_position_world"])
    return ObserverCalibration(
        optical_quaternion_world_wxyz=audit["optical_quaternion_ros_wxyz"],
        eye_world_m=eye.tolist(),
        look_at_world_m=(eye + rotation.apply([0, 0, 1])).tolist(),
        focal_length_pixels=float(matrix[0, 0] * 640 / width),
        image_width=640,
        image_height=360,
        composite_header_pixels=100,
    )


def recorded_targets_world(sample: dict[str, Any], root: Sequence[float]) -> np.ndarray:
    """Use the first current SDK reference, anchored at the recorded pelvis.

    Hybrid positions have already been rotated into the current body frame and
    command no global pelvis translation.
    """
    points = np.asarray(sample["vr_position_body_frame"]).reshape(3, 3)
    rotation = Rotation.from_quat(root[3:7], scalar_first=True)
    result = rotation.apply(points) + np.asarray(root[:3])
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("Recorded targets have an unsupported shape or nonfinite coordinates")
    return result
