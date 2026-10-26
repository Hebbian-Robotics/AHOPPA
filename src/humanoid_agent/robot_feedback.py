"""Readable robot-only feedback derived from encoders, IMU and action records."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.hand_control import upstream_hand_positions
from humanoid_agent.observation_history import ActionExecutionWindow, ImagePair
from humanoid_agent.robot_camera import calibration, wrist_calibration
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.sonic_planner_input import SonicMotionPublication


def rounded_values(values: Any, decimals: int = 6) -> Any:
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


def measured_hand_opening(record: IsaacObservation) -> dict[str, Any]:
    if record.hand_joint_sensors is None:
        return {"available": False}
    hands = {}
    for side in ("left", "right"):
        sensors = record.hand_joint_sensors[side]
        closed = np.asarray(upstream_hand_positions(side, "close"))
        measured = np.asarray(sensors.positions)
        closure = float(np.clip(np.dot(measured, closed) / np.dot(closed, closed), 0, 1))
        hands[side] = {
            "opening_percent": round(100 * (1 - closure), 1),
            "preset_fit_rms_rad": round(
                float(np.sqrt(np.mean((measured - closure * closed) ** 2))), 3
            ),
        }
    return {
        "available": True,
        "definition": "Encoder projection onto the pinned PICO preset: 100% open, 0% preset-closed. An estimate of configuration, not physical aperture, contact or grasp success.",
        **hands,
    }


def concise_robot_feedback(
    record: IsaacObservation,
    kinematics: dict[str, Any],
    now_elapsed_seconds: float,
) -> dict[str, Any]:
    positions = kinematics["tracking_points_m"]
    orientations = kinematics["tracking_orientations_wxyz"]
    copy_pose = {
        "tracking_points_m": {name: rounded_values(values) for name, values in positions.items()},
        "tracking_orientations_wxyz": {
            name: rounded_values(values) for name, values in orientations.items()
        },
    }
    return {
        "observation_id": record.observation_identity.identifier,
        "captured_elapsed_seconds": record.elapsed_wall_seconds,
        "observation_age_seconds_at_prompt": round(
            max(0.0, now_elapsed_seconds - record.elapsed_wall_seconds), 3
        ),
        "remaining_episode_seconds": round(
            max(
                0.0,
                record.observation_identity.episode_deadline_elapsed_wall_seconds
                - now_elapsed_seconds,
            ),
            3,
        ),
        "coordinate_frame": kinematics["coordinate_frame"],
        "units": "Positions: metres, quaternions: wxyz, tilt: degrees, joint angles: radians.",
        "measured_tracking": {
            name: {
                "position_m": rounded_values(positions[name], 4),
                "orientation_wxyz": rounded_values(orientations[name], 5),
            }
            for name in ("left_wrist", "right_wrist", "head")
        },
        "copy_current_pose": copy_pose,
        "body_tilt_degrees": dict(
            zip(
                ("roll", "pitch"),
                rounded_values(np.rad2deg(kinematics["base_roll_pitch_radians"]), 2),
                strict=True,
            )
        ),
        "hands": measured_hand_opening(record),
        "scope": "Robot measurements only. Base translation and room heading are not estimated. Pose copies retain six decimals for command continuity; display values are rounded.",
    }


def previous_action_feedback(
    publication: SonicMotionPublication,
    execution: ActionExecutionWindow,
    record: IsaacObservation,
    kinematics: dict[str, Any],
) -> dict[str, Any]:
    if publication.identifier != execution.action_id:
        raise ValueError("Feedback command differs from the executed action")
    measured_positions = kinematics["tracking_points_m"]
    measured_orientations = kinematics["tracking_orientations_wxyz"]
    requested_positions = dict(
        zip(
            ("left_wrist", "right_wrist", "head"),
            np.asarray(publication.poses[-1].positions_heading_frame).reshape(3, 3),
            strict=True,
        )
    )
    requested_orientations = dict(
        zip(
            ("left_wrist", "right_wrist", "head"),
            np.asarray(publication.poses[-1].orientations_heading_frame_wxyz).reshape(3, 4),
            strict=True,
        )
    )
    tracking = {}
    for name in ("left_wrist", "right_wrist", "head"):
        difference = np.asarray(measured_positions[name]) - requested_positions[name]
        tracking[name] = {
            "requested_position_m": rounded_values(requested_positions[name]),
            "measured_position_m": rounded_values(measured_positions[name]),
            "measured_minus_requested_m": rounded_values(difference, 4),
            "position_error_m": round(float(np.linalg.norm(difference)), 4),
        }
        rotation_error = cast(
            Rotation,
            Rotation.from_quat(requested_orientations[name], scalar_first=True).inv()
            * Rotation.from_quat(measured_orientations[name], scalar_first=True),
        )
        tracking[name].update(
            requested_orientation_wxyz=rounded_values(requested_orientations[name]),
            measured_orientation_wxyz=rounded_values(measured_orientations[name]),
            orientation_error_degrees=round(float(np.rad2deg(rotation_error.magnitude())), 2),
        )
    hands = measured_hand_opening(record)
    hand_feedback = {}
    for side, schedule in (
        ("left", publication.left_grasp_schedule),
        ("right", publication.right_grasp_schedule),
    ):
        measured_opening = hands.get(side, {}).get("opening_percent")
        requested_opening = (100 if schedule.command.action == "open" else 0) if schedule else None
        hand_feedback[side] = {
            "requested_action": schedule.command.action
            if schedule
            else "retain previous reference",
            "requested_opening_percent": requested_opening,
            "measured_opening_percent": measured_opening,
            "opening_error_percentage_points": round(measured_opening - requested_opening, 1)
            if measured_opening is not None and requested_opening is not None
            else None,
        }
    return {
        "action_id": publication.identifier,
        "execution_started_elapsed_seconds": execution.started_elapsed_seconds,
        "execution_endpoint_elapsed_seconds": execution.endpoint_elapsed_seconds,
        "measured_elapsed_seconds": record.elapsed_wall_seconds,
        "sample_after_execution_endpoint_seconds": round(
            record.elapsed_wall_seconds - execution.endpoint_elapsed_seconds, 3
        ),
        "requested_duration_seconds": publication.control_ticks * 0.02,
        "tracking": tracking,
        "hands": hand_feedback,
        "scope": "Requested nominal endpoint versus the current post-action robot measurement, in the action frame. Errors do not identify collision, slip, object contact or grasp success. Global locomotion displacement is unavailable.",
    }


def model_observation_payload(feedback: dict[str, Any], pairs: list[ImagePair]) -> dict[str, Any]:
    """Shared default prompt fields, intentionally excluding raw sensor arrays."""
    return {
        "robot_feedback": feedback,
        "visual_history": [
            {
                "frames": [frame.model_dump(mode="json") for frame in pair.frames],
                "view_order": [view.label for view in pair.views],
            }
            for pair in pairs
        ],
        "wrist_camera_calibration": {
            view.camera: wrist_calibration(view.camera.removesuffix("_wrist"))
            for view in pairs[-1].wrist_views
        }
        if pairs
        else {},
        "depth_queries": "query_onboard_depth reads selected pixels of the CURRENT/END onboard image only (640×480, x right/y down). Historic, external and wrist RGB have no depth tool. Surface points are measurements, not object targets. Current G1-joint and hybrid modes use point_heading_pelvis_m. Archived SMPL mode additionally returned root-local point_action_frame_m.",
        "camera_calibration": {
            name: calibration()[name]
            for name in (
                "resolution_width_height",
                "intrinsics_fx_fy_cx_cy",
                "optical_axes",
                "depth_kind",
                "depth_valid_interval_m",
                "assumptions",
            )
        },
        "sensor_assumptions": "Ideal simulated encoders/IMU and calibrated kinematics; synchronized fixed external RGB is an explicitly supplied observation. No simulator object pose, private contact channel, evaluator outcome or room localization is supplied. Images and sensor values stay frozen while physics continues.",
    }
