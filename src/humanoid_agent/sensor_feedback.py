"""Read-only kinematics and RGB-D measurements in SONIC's target frame."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from humanoid_agent.canonical_robot import (
    BODY_NAMES,
    JOINT_NAMES,
    POINT_OFFSETS,
    build_encoder_model,
    measured_tracking_points,
)
from humanoid_agent.robot_camera import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_DEPTH_M,
    MINIMUM_DEPTH_M,
    calibration,
    wrist_calibration,
)
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel

mujoco: Any = importlib.import_module("mujoco")

PROTOCOL = "egocentric-rgbd-wrist-rgb-kinematics-depth-tools-v4"
TRACKING_NAMES = ["left_wrist", "right_wrist", "head"]
FRAME_DESCRIPTION = (
    "Heading-normalized, gravity-aligned pelvis-relative metres: x forward, y left, z up. "
    "The origin is the current pelvis, not a room location. No global translation or yaw "
    "estimate is available. This is the numerical target frame of the head/hand SONIC interface; "
    "head positions and orientations describe the torso-mounted tracking frame."
)


class ImagePixel(ValidatedModel):
    x: int = Field(ge=0, lt=IMAGE_WIDTH)
    y: int = Field(ge=0, lt=IMAGE_HEIGHT)


class DepthQuery(ValidatedModel):
    pixels: ImmutableSequence[ImagePixel] = Field(min_length=1, max_length=32)


DEPTH_TOOL = {
    "type": "function",
    "name": "query_onboard_depth",
    "description": (
        "Read aligned metric depth at model-selected pixels in the CURRENT onboard RGB image "
        "(640x480, x right, y down; not the initial image or external view). Returns calibrated "
        "3D surface points in the SONIC head/hand target frame. Select the visible object or "
        "other landmarks yourself. This tool neither identifies objects nor moves the robot. "
        "Measurements belong to the saved snapshot; physics continues while you think."
    ),
    "strict": True,
    "parameters": DepthQuery.model_json_schema(),
}


class EncoderKinematics:
    """Use a robot-only model at zero translation; never load an evaluator state."""

    def __init__(self, urdf_path: Path | None = None) -> None:
        self.model, self.data = build_encoder_model(urdf_path)
        self.position_indices = np.asarray(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES]
        )
        self.calibration = calibration()

    def measure(self, sensors: RobotSensors, previous_targets: Any = None) -> dict[str, Any]:
        sensors = sensors.in_joint_order(JOINT_NAMES)
        gravity = np.asarray(sensors.imu_gravity_direction_body, dtype=float)
        gravity_norm = np.linalg.norm(gravity)
        if not 0.9 <= gravity_norm <= 1.1:
            raise ValueError("IMU gravity direction must have approximately unit length")
        gravity /= gravity_norm
        roll = np.arctan2(-gravity[1], -gravity[2])
        pitch = np.arcsin(np.clip(gravity[0], -1, 1))
        self.data.qpos[:] = 0
        self.data.qpos[3:7] = Rotation.from_euler("xyz", [roll, pitch, 0]).as_quat(
            scalar_first=True
        )
        self.data.qpos[self.position_indices] = sensors.joint_positions_rad
        mujoco.mj_forward(self.model, self.data)
        positions, orientations, _ = measured_tracking_points(self.model, self.data)
        camera_body = self.model.body(self.calibration["camera_body"]).id
        torso_rotation = self.data.xmat[camera_body].reshape(3, 3)
        optical_origin = self.data.xpos[camera_body] + torso_rotation @ np.asarray(
            self.calibration["camera_position_torso_m"]
        )
        optical_rotation = torso_rotation @ np.asarray(
            self.calibration["optical_to_torso_rotation"]
        )
        result = {
            "coordinate_frame": FRAME_DESCRIPTION,
            "tracking_points_m": dict(zip(TRACKING_NAMES, positions.tolist(), strict=True)),
            "tracking_orientations_wxyz": dict(
                zip(TRACKING_NAMES, orientations.tolist(), strict=True)
            ),
            "base_roll_pitch_radians": [float(roll), float(pitch)],
            "torso_roll_pitch_yaw_in_heading_frame_radians": Rotation.from_matrix(torso_rotation)
            .as_euler("xyz")
            .tolist(),
            "camera_optical_origin_m": optical_origin.tolist(),
            "camera_forward_axis_in_heading_frame": optical_rotation[:, 2].tolist(),
            "camera_downward_pitch_degrees": float(
                np.rad2deg(
                    np.arctan2(-optical_rotation[2, 2], np.linalg.norm(optical_rotation[:2, 2]))
                )
            ),
            "camera_optical_to_heading_rotation": optical_rotation.tolist(),
            "tracking_point_definitions": [
                {"name": name, "body": body, "offset_body_m": offset.tolist()}
                for name, body, offset in zip(
                    TRACKING_NAMES, BODY_NAMES, POINT_OFFSETS, strict=True
                )
            ],
            "measurement_source": "joint encoders, IMU gravity and calibrated robot kinematics",
            "global_base_motion": "not estimated; posture changes are not room displacement",
        }
        result["wrist_camera_kinematics"] = {}
        for side in ("left", "right"):
            mount = wrist_calibration(side)
            wrist_body = self.model.body(mount["camera_body"]).id
            wrist_rotation = self.data.xmat[wrist_body].reshape(3, 3)
            result["wrist_camera_kinematics"][side] = {
                "optical_origin_m": (
                    self.data.xpos[wrist_body]
                    + wrist_rotation @ np.asarray(mount["position_body_m"])
                ).tolist(),
                "optical_to_heading_rotation": (
                    wrist_rotation @ np.asarray(mount["optical_to_body_rotation"])
                ).tolist(),
            }
        if previous_targets is not None:
            errors = positions - np.asarray(previous_targets)
            result["measured_minus_previous_command_m"] = dict(
                zip(TRACKING_NAMES, errors.tolist(), strict=True)
            )
        return result


class DepthSnapshot:
    """Deproject only sensor pixels; no task coordinates or action methods exist."""

    def __init__(
        self,
        depth_m: Any,
        kinematics: dict[str, Any],
        simulation_time: float,
        wall_time: float,
        *,
        optical_points_m: Any,
    ) -> None:
        self.depth_m = np.asarray(depth_m).copy()
        if self.depth_m.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError("Depth does not match the calibrated onboard RGB image")
        self.optical_points_m = np.asarray(optical_points_m).copy()
        if self.optical_points_m.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError("Optical points must match the calibrated depth image")
        self.kinematics = kinematics
        self.simulation_time = simulation_time
        self.wall_time = wall_time
        self.queries: list[dict[str, Any]] = []

    def query(self, arguments: str) -> dict[str, Any]:
        query = DepthQuery.model_validate_json(arguments)
        origin = np.asarray(self.kinematics["camera_optical_origin_m"])
        rotation = np.asarray(self.kinematics["camera_optical_to_heading_rotation"])
        points = []
        for pixel in query.pixels:
            depth = float(self.depth_m[pixel.y, pixel.x])
            valid = bool(np.isfinite(depth) and MINIMUM_DEPTH_M <= depth <= MAXIMUM_DEPTH_M)
            point: dict[str, Any] = {"pixel": pixel.model_dump(), "valid_samples": int(valid)}
            if valid:
                optical_point = self.optical_points_m[pixel.y, pixel.x]
                if not np.isfinite(optical_point).all() or not np.isclose(
                    optical_point[2], depth, atol=1e-6, rtol=0
                ):
                    raise ValueError("Upstream optical points disagree with selected depth")
                point.update(
                    depth_optical_z_m=depth,
                    point_optical_m=optical_point.tolist(),
                    point_heading_pelvis_m=(origin + rotation @ optical_point).tolist(),
                )
            else:
                point.update(
                    depth_optical_z_m=None,
                    point_optical_m=None,
                    point_heading_pelvis_m=None,
                )
            points.append(point)
        result = {
            "protocol": PROTOCOL,
            "coordinate_frame": FRAME_DESCRIPTION,
            "snapshot_simulation_time_seconds": self.simulation_time,
            "snapshot_wall_elapsed_seconds": self.wall_time,
            "aggregation": "raw selected pixel; invalid measurements remain unavailable",
            "points": points,
            "note": (
                "Surface measurements at selected pixels; no object recognition or motion output"
            ),
        }
        self.queries.append(result)
        return result
