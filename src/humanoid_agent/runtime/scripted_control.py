"""Deterministic planner inputs for the no-interaction standing controller check."""

from __future__ import annotations

import math

from humanoid_agent.sonic_planner_input import NativePlannerSample


class StandingControlSource:
    """Hold the released neutral targets for one phase, then hold the endpoint.

    Sampling after completion keeps the original neutral targets; nothing walks,
    reaches or closes a hand.
    """

    def __init__(self, neutral: NativePlannerSample, phase_seconds: float = 5.0) -> None:
        if not math.isfinite(phase_seconds) or not 1 <= phase_seconds <= 30:
            raise ValueError("Scripted phase duration must be finite and between 1 and 30 seconds")
        self._neutral = NativePlannerSample.model_validate(neutral)
        self._phase_seconds = phase_seconds

    @property
    def duration_seconds(self) -> float:
        return self._phase_seconds

    def phase(self, elapsed_seconds: float) -> str:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Scripted elapsed time must be finite and nonnegative")
        return "stand"

    def sample(self, elapsed_seconds: float) -> NativePlannerSample:
        self.phase(elapsed_seconds)
        return self._neutral.model_copy(
            update={
                "locomotion_mode": 0,
                "speed_m_per_s": -1,
                "movement_direction_world": self._neutral.movement_direction_world,
                "vr_position_body_frame": list(self._neutral.vr_position_body_frame),
                "right_hand_joints": self._neutral.right_hand_joints,
                "endpoint_hold": elapsed_seconds >= self.duration_seconds,
            }
        )
