"""Translate a validated model decision into native planner inputs without inference."""

import math
from typing import Any

from humanoid_agent.command_admission import ObservationIdentity
from humanoid_agent.hand_control import resolve_grasp
from humanoid_agent.model_decision_validation import validate_initial_isaac_targets
from humanoid_agent.posture_control import (
    CONTROL_PERIOD_SECONDS,
    PlannedIsaacChunk,
    PlannedSteppedTurnTeleopIsaacChunk,
    PlannedTeleopIsaacChunk,
    command_timing,
    planner_mode_and_height,
    planner_movement_speed,
    turn_budget_seconds,
)
from humanoid_agent.sonic_planner_input import (
    CURRENT_SONIC_PLANNER_PROTOCOL,
    PlannerTrackingPose,
    SonicPlannerProtocol,
    SonicPlannerPublication,
)


def prepare_planner_publication(
    decision: PlannedIsaacChunk | PlannedTeleopIsaacChunk,
    identity: ObservationIdentity,
    kinematics: dict[str, Any],
    identifier: str,
    response_sha256: str,
    *,
    hand_limits: dict[str, dict[str, float]] | None = None,
    command_protocol: SonicPlannerProtocol = CURRENT_SONIC_PLANNER_PROTOCOL,
    leashed_facing: bool = False,
) -> SonicPlannerPublication:
    """Preserve model keyframes; never synthesize lower-body trajectories."""
    if command_protocol == "sonic-native-planner-v1" and not 0.4 <= decision.duration_seconds <= 4:
        raise ValueError("Legacy v1 motion duration must be from 0.4 to 4 seconds")
    validate_initial_isaac_targets(
        decision,
        kinematics,
        hand_limits=hand_limits,
        maximum_duration_seconds=(
            identity.episode_deadline_elapsed_wall_seconds - identity.captured_elapsed_wall_seconds
            if command_protocol != "sonic-native-planner-v1"
            else None
        ),
    )
    teleop = isinstance(decision, PlannedTeleopIsaacChunk)
    left_schedule = (
        resolve_grasp(
            decision.left_grasp,
            "left",
            duration_seconds=decision.duration_seconds,
            limits=hand_limits,
        )
        if teleop
        else None
    )
    right_schedule = (
        resolve_grasp(
            decision.right_grasp,
            "right",
            duration_seconds=decision.duration_seconds,
            limits=hand_limits,
        )
        if teleop
        else None
    )
    if teleop:
        command_protocol = "sonic-native-planner-v3"
    control_ticks, _ = command_timing(decision)
    turn_degrees = (
        decision.turn_in_place_degrees
        if isinstance(decision, PlannedSteppedTurnTeleopIsaacChunk)
        else 0
    )
    if turn_degrees != 0:
        # The turn runs until its measured heading settles, within a fixed budget.
        control_ticks = round(turn_budget_seconds(turn_degrees) / CONTROL_PERIOD_SECONDS)
    mode, height = planner_mode_and_height(decision)
    if mode not in {0, 1, 4, 5, 6}:
        raise ValueError("Unsupported native planner locomotion mode")
    body_names = ("left_wrist", "right_wrist", "head")
    return SonicPlannerPublication(
        protocol=command_protocol,
        left_grasp_schedule=left_schedule,
        right_grasp_schedule=right_schedule,
        identifier=identifier,
        source_response_sha256=response_sha256,
        episode_id=identity.episode_id,
        observation_id=identity.identifier,
        observation_elapsed_wall_seconds=identity.captured_elapsed_wall_seconds,
        valid_until_elapsed_wall_seconds=identity.episode_deadline_elapsed_wall_seconds,
        request_task_completion=decision.request_task_completion,
        inspection_report=decision.inspection_report,
        control_ticks=control_ticks,
        locomotion_mode=mode,
        movement_direction_heading_frame=decision.movement_direction,
        facing_yaw_radians=decision.facing_yaw_radians,
        speed_m_per_s=planner_movement_speed(decision),
        turn_in_place_radians=math.radians(turn_degrees) if turn_degrees != 0 else None,
        leashed_facing=leashed_facing,
        body_height_m=height,
        left_hand_joints=None
        if isinstance(decision, PlannedTeleopIsaacChunk)
        else decision.left_hand_joints,
        right_hand_joints=None
        if isinstance(decision, PlannedTeleopIsaacChunk)
        else decision.right_hand_joints,
        poses=[
            PlannerTrackingPose(
                positions_heading_frame=[
                    coordinate for name in body_names for coordinate in pose.model_dump()[name]
                ],
                orientations_heading_frame_wxyz=[
                    coordinate
                    for name in body_names
                    for coordinate in orientation.model_dump()[name]
                ],
            )
            for pose, orientation in zip(decision.poses, decision.orientations, strict=True)
        ],
    )
