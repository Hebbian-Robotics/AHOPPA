"""Explicit numerical SONIC posture commands and finite model timing."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from humanoid_agent.dex3 import HandVector
from humanoid_agent.hand_control import GraspCommand
from humanoid_agent.task_planning import PlannedHybridChunk
from humanoid_agent.validated_types import ImmutableSequence, UnitQuaternion, ValidatedModel

POSTURE_MODES = {"standing": 0, "squat": 4, "kneel_both": 5, "kneel_single": 6}
CONTROL_PERIOD_SECONDS = 0.02


class TrackingOrientations(ValidatedModel):
    """Body-to-pelvis-heading rotations, in SONIC's w, x, y, z order."""

    left_wrist: UnitQuaternion
    right_wrist: UnitQuaternion
    head: UnitQuaternion


class MotionTiming(ValidatedModel):
    duration_seconds: FiniteFloat = Field(gt=0)


class PlannedPostureChunk(PlannedHybridChunk, MotionTiming):
    locomotion_posture: Literal["standing", "squat", "kneel_both", "kneel_single"]
    body_height_m: float = Field(ge=0.2, le=0.8)

    @model_validator(mode="after")
    def reject_walking_in_static_posture(self) -> PlannedPostureChunk:
        if self.walking and self.locomotion_posture != "standing":
            raise ValueError("Squat and kneel modes are static; standing is required for walking")
        return self


class PlannedIsaacChunk(PlannedPostureChunk):
    left_hand_joints: HandVector | None = None
    right_hand_joints: HandVector | None = None
    orientations: ImmutableSequence[TrackingOrientations] = Field(min_length=4, max_length=4)


class PlannedTeleopIsaacChunk(PlannedPostureChunk):
    """Human-style grasp switches; raw joint angles remain a historical schema only."""

    left_grasp: GraspCommand | None = None
    right_grasp: GraspCommand | None = None
    orientations: ImmutableSequence[TrackingOrientations] = Field(min_length=4, max_length=4)


TurnInPlaceDegrees = Literal[0, 45, -45, 90, -90, 135, -135, 180, -180]
# Raw facing changes beyond 45 degrees preceded storage-round falls; larger turns use
# the stepped in-place turn, which advances one settled step of at most 45 degrees.
MAXIMUM_DIRECT_FACING_RADIANS = math.pi / 4
# Measured stepped turns took up to 11 s for 90 and 15 s for 180 degrees.
TURN_BUDGET_SECONDS_PER_STEP = 5.5
TURN_BUDGET_MARGIN_SECONDS = 1.0


def turn_budget_seconds(turn_degrees: float) -> float:
    steps = math.ceil(abs(turn_degrees) / 45 - 1e-9)
    return steps * TURN_BUDGET_SECONDS_PER_STEP + TURN_BUDGET_MARGIN_SECONDS


class PlannedSteppedTurnTeleopIsaacChunk(PlannedTeleopIsaacChunk):
    """Teleop chunk whose large turns go through the stepped in-place turn tool."""

    facing_yaw_radians: float = Field(
        ge=-MAXIMUM_DIRECT_FACING_RADIANS, le=MAXIMUM_DIRECT_FACING_RADIANS
    )
    turn_in_place_degrees: TurnInPlaceDegrees

    @model_validator(mode="after")
    def require_pure_turn(self) -> PlannedSteppedTurnTeleopIsaacChunk:
        if self.turn_in_place_degrees != 0 and (
            self.walking or self.facing_yaw_radians != 0 or self.locomotion_posture != "standing"
        ):
            raise ValueError(
                "turn_in_place_degrees requires standing, walking false and facing_yaw_radians 0"
            )
        return self


def command_timing(timing: MotionTiming) -> tuple[int, Any]:
    ticks = max(1, round(timing.duration_seconds / CONTROL_PERIOD_SECONDS))
    return ticks, np.linspace(0, ticks * CONTROL_PERIOD_SECONDS, 4)


def planner_mode_and_height(plan: PlannedHybridChunk) -> tuple[int, float]:
    if not isinstance(plan, PlannedPostureChunk) or plan.locomotion_posture == "standing":
        return (1 if plan.walking else 0), -1.0
    return POSTURE_MODES[plan.locomotion_posture], plan.body_height_m


def planner_movement_speed(plan: PlannedHybridChunk) -> float:
    """Keep static posture speed at zero, matching a released gamepad stick."""
    return plan.speed_m_per_s if plan.walking else 0.0
