"""Construct neutral native planner inputs from the canonical encoder measurement."""

from __future__ import annotations

from humanoid_agent.runtime.camera import NativeVisualState
from humanoid_agent.sensor_feedback import TRACKING_NAMES, EncoderKinematics
from humanoid_agent.sonic_planner_input import NativePlannerSample


def neutral_planner_sample(
    snapshot: NativeVisualState, *, encoder: EncoderKinematics | None = None
) -> NativePlannerSample:
    """Use the same canonical initial-pose FK for study and diagnostic commands."""
    encoder = EncoderKinematics() if encoder is None else encoder
    kinematics = encoder.measure(snapshot.robot_sensors())
    return NativePlannerSample(
        locomotion_mode=0,
        movement_direction_world=[1, 0, 0],
        facing_direction_world=[1, 0, 0],
        speed_m_per_s=0,
        body_height_m=-1,
        vr_position_body_frame=[
            value for name in TRACKING_NAMES for value in kinematics["tracking_points_m"][name]
        ],
        vr_orientation_body_frame_wxyz=[
            value
            for name in TRACKING_NAMES
            for value in kinematics["tracking_orientations_wxyz"][name]
        ],
        left_hand_joints=[0.0] * 7 if len(snapshot.joint_names) == 43 else None,
        right_hand_joints=[0.0] * 7 if len(snapshot.joint_names) == 43 else None,
        endpoint_hold=True,
    )
