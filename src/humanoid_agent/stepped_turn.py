"""Turn the released planner in upstream-style 45-degree facing steps.

The released gamepad driver turns by adding plus or minus pi/4 to the facing angle
(L1/R1). Continuous facing sweeps, snaps and walking arcs all preceded storage-round
falls. A stepped turn commands one facing step of at most 45 degrees while standing,
waits until the measured heading stops changing, then steps again from the measured
heading. It never commands walking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

FACING_STEP_RADIANS = math.pi / 4
# The released planner settled about 10 degrees short of each 45-degree step, so a
# step ends when the heading stops changing and the next step starts from it.
STILL_HEADING_TOLERANCE_RADIANS = math.radians(1.5)
SETTLED_HOLD_SECONDS = 1.0
FINAL_HEADING_TOLERANCE_RADIANS = math.radians(12.0)
STEP_TIMEOUT_SECONDS = 10.0
EXTRA_CORRECTION_STEPS = 3

SteppedTurnStatus = Literal["turning", "done", "timed_out"]


def wrapped(angle: float) -> float:
    return math.remainder(angle, math.tau)


@dataclass
class SteppedTurn:
    """One turn by a signed angle, executed as settled facing steps of at most 45 degrees."""

    start_heading_radians: float
    turn_radians: float
    started_seconds: float
    status: SteppedTurnStatus = "turning"
    step_settle_seconds: list[float] = field(default_factory=list)
    step_target_radians: float = field(init=False)
    step_started_seconds: float = field(init=False)
    still_reference_radians: float | None = None
    still_since_seconds: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.turn_radians) or abs(self.turn_radians) > math.pi + 1e-9:
            raise ValueError("A stepped turn covers at most 180 degrees")
        self.start_step(self.started_seconds, self.start_heading_radians)

    @property
    def final_target_radians(self) -> float:
        return wrapped(self.start_heading_radians + self.turn_radians)

    @property
    def step_limit(self) -> int:
        return (
            math.ceil(abs(self.turn_radians) / FACING_STEP_RADIANS - 1e-9) + EXTRA_CORRECTION_STEPS
        )

    def remaining_radians(self, heading_radians: float) -> float:
        # Keep the commanded turn direction for a half turn instead of the shorter wrap.
        remaining = wrapped(self.final_target_radians - heading_radians)
        if abs(abs(remaining) - math.pi) < 1e-6:
            return math.copysign(abs(remaining), self.turn_radians)
        return remaining

    def start_step(self, elapsed_seconds: float, heading_radians: float) -> None:
        remaining = self.remaining_radians(heading_radians)
        step = max(-FACING_STEP_RADIANS, min(FACING_STEP_RADIANS, remaining))
        self.step_target_radians = wrapped(heading_radians + step)
        self.step_started_seconds = elapsed_seconds
        self.still_reference_radians = None
        self.still_since_seconds = None

    def facing_xy(self) -> list[float]:
        return [math.cos(self.step_target_radians), math.sin(self.step_target_radians), 0.0]

    def update(
        self, elapsed_seconds: float, heading_radians: float, pelvis_stable: bool = True
    ) -> SteppedTurnStatus:
        """Advance on the measured heading; an unstable pelvis keeps the step open."""
        if self.status != "turning":
            return self.status
        moving = (
            self.still_reference_radians is None
            or abs(wrapped(heading_radians - self.still_reference_radians))
            > STILL_HEADING_TOLERANCE_RADIANS
            or not pelvis_stable
        )
        if moving:
            self.still_reference_radians = heading_radians
            self.still_since_seconds = elapsed_seconds
        assert self.still_since_seconds is not None
        # Ignore stillness before the step has begun to turn the robot.
        began = elapsed_seconds - self.step_started_seconds >= SETTLED_HOLD_SECONDS
        if began and elapsed_seconds - self.still_since_seconds >= SETTLED_HOLD_SECONDS:
            self.step_settle_seconds.append(elapsed_seconds - self.step_started_seconds)
            if abs(self.remaining_radians(heading_radians)) <= FINAL_HEADING_TOLERANCE_RADIANS:
                self.status = "done"
            elif len(self.step_settle_seconds) >= self.step_limit:
                self.status = "timed_out"
            else:
                self.start_step(elapsed_seconds, heading_radians)
        elif elapsed_seconds - self.step_started_seconds >= STEP_TIMEOUT_SECONDS:
            self.status = "timed_out"
        return self.status
