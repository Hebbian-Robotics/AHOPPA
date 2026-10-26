"""Keep transport parsing separate from validation of executable robot decisions."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from pydantic import BaseModel, RootModel
from scipy.spatial.transform import Rotation

from humanoid_agent.dex3 import validate_hand_joint_targets
from humanoid_agent.hand_control import validate_grasp_timing
from humanoid_agent.posture_control import PlannedIsaacChunk, PlannedTeleopIsaacChunk


@cache
def transport_payload_schema(decision_schema: type[BaseModel]) -> type[BaseModel]:
    """Send the original JSON schema, preserving raw output before domain checks."""

    class TransportPayload(RootModel[dict[str, Any]]):
        @classmethod
        def model_json_schema(cls, *arguments: Any, **keywords: Any) -> dict[str, Any]:
            return decision_schema.model_json_schema(*arguments, **keywords)

    TransportPayload.__name__ = decision_schema.__name__ + "Payload"
    return TransportPayload


def validate_initial_isaac_targets(
    decision: BaseModel,
    kinematics: dict[str, Any],
    *,
    hand_limits: dict[str, dict[str, float]] | None = None,
    maximum_duration_seconds: float | None = None,
) -> None:
    """Check sensor-relative continuity without generating or modifying targets."""
    if isinstance(decision, PlannedTeleopIsaacChunk):
        for command in (decision.left_grasp, decision.right_grasp):
            validate_grasp_timing(
                command, decision.duration_seconds, hands_available=hand_limits is not None
            )
        decision = PlannedIsaacChunk.model_validate(
            decision.model_dump(exclude={"left_grasp", "right_grasp", "turn_in_place_degrees"})
        )
    if not isinstance(decision, PlannedIsaacChunk):
        return
    if (
        maximum_duration_seconds is not None
        and decision.duration_seconds > maximum_duration_seconds
    ):
        raise ValueError("Motion duration exceeds the remaining episode budget at observation")
    validate_hand_joint_targets(decision.left_hand_joints, decision.right_hand_joints, hand_limits)
    body_names = ["left_wrist", "right_wrist", "head"]
    all_positions = np.asarray(
        [[pose.model_dump()[name] for name in body_names] for pose in decision.poses]
    )
    if np.max(np.abs(all_positions)) > 1:
        raise ValueError("Head/hand targets exceed one-metre interface bounds")
    first_positions = decision.poses[0].model_dump()
    measured = np.asarray([kinematics["tracking_points_m"][name] for name in body_names])
    targets = np.asarray([first_positions[name] for name in body_names])
    if not np.allclose(targets, measured, atol=0.02):
        raise ValueError("First keyframe must match the current measured tracking points")
    first_orientations = decision.orientations[0].model_dump()
    orientations = np.asarray([first_orientations[name] for name in body_names])
    measured_orientations = np.asarray(
        [kinematics["tracking_orientations_wxyz"][name] for name in body_names]
    )
    relative_rotation = Rotation.from_quat(
        orientations, scalar_first=True
    ).inv() * Rotation.from_quat(measured_orientations, scalar_first=True)
    if not isinstance(relative_rotation, Rotation):
        raise TypeError("Tracking orientation comparison failed")
    orientation_error = relative_rotation.magnitude()
    if np.max(orientation_error) > 0.02:
        raise ValueError("First orientation keyframe must match the current measured orientations")
