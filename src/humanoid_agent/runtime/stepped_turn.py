"""Privileged in-place check of the stepped turn primitive; no model requests."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.runtime.evaluation import GroundTaskMeasurements
from humanoid_agent.shutdown_task import MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
from humanoid_agent.sonic_planner_input import NativePlannerSample, heading_rotation
from humanoid_agent.stepped_turn import SteppedTurn


def measured_heading_radians(root_state_wxyz: list[float]) -> float:
    rotation = Rotation.from_quat(np.asarray(root_state_wxyz[3:7]), scalar_first=True)
    forward = heading_rotation(rotation).apply([1.0, 0.0, 0.0])
    return math.atan2(forward[1], forward[0])


def stable(measured: GroundTaskMeasurements) -> bool:
    return (
        measured.root_height_m >= 0.70
        and measured.gravity_z < -0.90
        and measured.horizontal_speed_m_per_s <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
        and abs(measured.root_vertical_speed_m_per_s) <= MAXIMUM_STABLE_PELVIS_SPEED_M_PER_S
    )


TURN_CHECK_PROGRAM_DEGREES = (90.0, 90.0, 180.0, -90.0, -180.0, -90.0)
TURN_CHECK_DURATION_SECONDS = 180.0


class TurnCheckControl:
    """Stand, then run a fixed program of stepped turns in place at home.

    A privileged physics check of the turn primitive; no model requests.
    """

    def __init__(self, neutral: NativePlannerSample) -> None:
        self.neutral = neutral
        self.latest_measurement: GroundTaskMeasurements | None = None
        self.failure_reason: str | None = None
        self.turn_index = 0
        self.turn: SteppedTurn | None = None
        self.completed_turns: list[dict[str, object]] = []
        self.settle_until_seconds = 3.0
        self._last_sample = neutral

    @property
    def duration_seconds(self) -> float:
        return TURN_CHECK_DURATION_SECONDS

    def phase(self, elapsed_seconds: float) -> str:
        del elapsed_seconds
        if self.failure_reason is not None:
            return f"halted-{self.failure_reason}"
        if self.turn_index >= len(TURN_CHECK_PROGRAM_DEGREES):
            return "done"
        return f"turn-{self.turn_index}-{TURN_CHECK_PROGRAM_DEGREES[self.turn_index]:+.0f}"

    def sample(
        self, elapsed_seconds: float, root_state_wxyz: list[float], hand_points_m: list[list[float]]
    ) -> NativePlannerSample:
        del hand_points_m
        measured = self.latest_measurement
        if (
            measured is None
            or self.failure_reason is not None
            or self.turn_index >= len(TURN_CHECK_PROGRAM_DEGREES)
            or elapsed_seconds < self.settle_until_seconds
        ):
            return self._last_sample
        heading = measured_heading_radians(root_state_wxyz)
        if self.turn is None:
            self.turn = SteppedTurn(
                heading, math.radians(TURN_CHECK_PROGRAM_DEGREES[self.turn_index]), elapsed_seconds
            )
        status = self.turn.update(elapsed_seconds, heading, stable(measured))
        if status != "turning":
            self.completed_turns.append(
                {
                    "turn_degrees": TURN_CHECK_PROGRAM_DEGREES[self.turn_index],
                    "status": status,
                    "seconds": elapsed_seconds - self.turn.started_seconds,
                    "step_settle_seconds": self.turn.step_settle_seconds,
                    "final_heading_error_degrees": math.degrees(
                        self.turn.remaining_radians(heading)
                    ),
                }
            )
            if status == "timed_out":
                self.failure_reason = "turn-step-timeout"
                return self._last_sample
            self.turn_index += 1
            self.turn = None
            self.settle_until_seconds = elapsed_seconds + 1.0
            return self._last_sample
        self._last_sample = self.neutral.model_copy(
            update={
                "locomotion_mode": 0,
                "speed_m_per_s": 0.0,
                "facing_direction_world": self.turn.facing_xy(),
            }
        )
        return self._last_sample

    def summary(self) -> dict[str, object]:
        return {
            "phase": self.phase(0.0),
            "failure_reason": self.failure_reason,
            "program_degrees": list(TURN_CHECK_PROGRAM_DEGREES),
            "completed_turns": self.completed_turns,
            "model_requests": 0,
        }
