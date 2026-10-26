"""The storage round's in-place turn tool replaces large raw facing changes."""

import math
import unittest

from pydantic import ValidationError

from humanoid_agent.command_admission import ObservationIdentity
from humanoid_agent.posture_control import (
    MAXIMUM_DIRECT_FACING_RADIANS,
    PlannedSteppedTurnTeleopIsaacChunk,
    turn_budget_seconds,
)
from humanoid_agent.sonic_model_command import prepare_planner_publication
from humanoid_agent.sonic_planner_input import NativePlannerSample, motion_timeline
from humanoid_agent.stepped_turn import FACING_STEP_RADIANS, wrapped

POSITIONS = {"left_wrist": [0.1, 0.2, -0.1], "right_wrist": [0.1, -0.2, -0.1], "head": [0, 0, 0.4]}
ORIENTATIONS = {name: [1, 0, 0, 0] for name in POSITIONS}
DECISION = {
    "subtasks": [
        {
            "identifier": "turn",
            "description": "Turn",
            "completion_condition": "Facing C",
            "status": "active",
        }
    ],
    "active_subtask_id": "turn",
    "observed_progress": "Ready",
    "request_task_completion": False,
    "duration_seconds": 2.0,
    "walking": False,
    "speed_m_per_s": 0.3,
    "movement_direction": [1, 0, 0],
    "facing_yaw_radians": 0,
    "turn_in_place_degrees": 0,
    "poses": [POSITIONS] * 4,
    "orientations": [ORIENTATIONS] * 4,
    "locomotion_posture": "standing",
    "body_height_m": 0.75,
}
IDENTITY = ObservationIdentity(
    episode_id="episode",
    identifier="initial",
    captured_elapsed_wall_seconds=0,
    episode_deadline_elapsed_wall_seconds=120,
)
KINEMATICS = {"tracking_points_m": POSITIONS, "tracking_orientations_wxyz": ORIENTATIONS}


def yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class TurnInPlaceToolTests(unittest.TestCase):
    def test_raw_facing_beyond_45_degrees_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PlannedSteppedTurnTeleopIsaacChunk.model_validate(
                {**DECISION, "facing_yaw_radians": MAXIMUM_DIRECT_FACING_RADIANS + 0.05}
            )

    def test_turn_must_stand_still_without_extra_facing(self) -> None:
        for change in ({"walking": True}, {"facing_yaw_radians": 0.2}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                PlannedSteppedTurnTeleopIsaacChunk.model_validate(
                    {**DECISION, "turn_in_place_degrees": 90, **change}
                )

    def test_turn_runs_within_its_budget_regardless_of_model_duration(self) -> None:
        decision = PlannedSteppedTurnTeleopIsaacChunk.model_validate(
            {**DECISION, "turn_in_place_degrees": -90}
        )
        publication = prepare_planner_publication(decision, IDENTITY, KINEMATICS, "turn", "a" * 64)
        self.assertAlmostEqual(publication.turn_in_place_radians or 0, -math.pi / 2)
        self.assertAlmostEqual(publication.control_ticks * 0.02, turn_budget_seconds(-90))
        plain = prepare_planner_publication(
            PlannedSteppedTurnTeleopIsaacChunk.model_validate(DECISION),
            IDENTITY,
            KINEMATICS,
            "stand",
            "a" * 64,
        )
        self.assertIsNone(plain.turn_in_place_radians)
        self.assertNotIn("turn_in_place_radians", plain.model_dump())

    def test_timeline_steps_facing_from_the_measured_heading(self) -> None:
        decision = PlannedSteppedTurnTeleopIsaacChunk.model_validate(
            {**DECISION, "turn_in_place_degrees": 180}
        )
        publication = prepare_planner_publication(decision, IDENTITY, KINEMATICS, "turn", "a" * 64)
        start_heading = 0.3
        timeline = motion_timeline(publication, yaw_quaternion(start_heading))
        heading = start_heading
        commanded = []
        elapsed = 0.0
        while elapsed < timeline.execution_duration_seconds:
            sample = timeline.sample(elapsed, yaw_quaternion(heading))
            assert isinstance(sample, NativePlannerSample)
            self.assertEqual(sample.locomotion_mode, 0)
            facing = math.atan2(sample.facing_direction_world[1], sample.facing_direction_world[0])
            self.assertLessEqual(abs(wrapped(facing - heading)), FACING_STEP_RADIANS + 1e-6)
            if not commanded or abs(wrapped(facing - commanded[-1])) > 1e-9:
                commanded.append(facing)
                # A robot that settles 10 degrees short of every step.
                step = wrapped(facing - heading)
                heading = wrapped(facing - math.copysign(min(abs(step), math.radians(10)), step))
            elapsed += 0.02
        self.assertLessEqual(abs(wrapped(start_heading + math.pi - heading)), math.radians(12))


if __name__ == "__main__":
    unittest.main()
