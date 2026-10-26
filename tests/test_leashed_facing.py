"""Leashed facing approaches a raw facing change without a large planner jump."""

import math
import unittest

from humanoid_agent.command_admission import ObservationIdentity
from humanoid_agent.posture_control import PlannedTeleopIsaacChunk
from humanoid_agent.sonic_model_command import prepare_planner_publication
from humanoid_agent.sonic_planner_input import (
    FACING_LEAD_RADIANS,
    NativePlannerSample,
    motion_timeline,
)

POSITIONS = {"left_wrist": [0.1, 0.2, -0.1], "right_wrist": [0.1, -0.2, -0.1], "head": [0, 0, 0.4]}
ORIENTATIONS = {name: [1, 0, 0, 0] for name in POSITIONS}
IDENTITY = ObservationIdentity(
    episode_id="episode",
    identifier="initial",
    captured_elapsed_wall_seconds=0,
    episode_deadline_elapsed_wall_seconds=120,
)
KINEMATICS = {"tracking_points_m": POSITIONS, "tracking_orientations_wxyz": ORIENTATIONS}


def decision(facing_yaw_radians: float) -> PlannedTeleopIsaacChunk:
    return PlannedTeleopIsaacChunk.model_validate(
        {
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
            "duration_seconds": 3.0,
            "walking": False,
            "speed_m_per_s": 0.3,
            "movement_direction": [1, 0, 0],
            "facing_yaw_radians": facing_yaw_radians,
            "poses": [POSITIONS] * 4,
            "orientations": [ORIENTATIONS] * 4,
            "locomotion_posture": "standing",
            "body_height_m": 0.75,
        }
    )


def yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def facing_of(sample: NativePlannerSample) -> float:
    return math.atan2(sample.facing_direction_world[1], sample.facing_direction_world[0])


class LeashedFacingTests(unittest.TestCase):
    def test_large_facing_change_stays_within_the_lead_and_converges(self) -> None:
        publication = prepare_planner_publication(
            decision(math.radians(170)), IDENTITY, KINEMATICS, "turn", "a" * 64, leashed_facing=True
        )
        timeline = motion_timeline(publication, yaw_quaternion(0.0))
        heading = 0.0
        elapsed = 0.0
        while elapsed < 30:
            sample = timeline.sample(elapsed, yaw_quaternion(heading))
            assert isinstance(sample, NativePlannerSample)
            lead = math.remainder(facing_of(sample) - heading, math.tau)
            self.assertLessEqual(abs(lead), FACING_LEAD_RADIANS + 1e-9)
            # A robot that turns at 40 degrees per second toward the command.
            heading += max(-math.radians(0.8), min(math.radians(0.8), lead))
            elapsed += 0.02
        self.assertAlmostEqual(math.remainder(heading - math.radians(170), math.tau), 0, places=2)

    def test_unleashed_publication_keeps_the_raw_facing(self) -> None:
        publication = prepare_planner_publication(
            decision(math.radians(170)), IDENTITY, KINEMATICS, "turn", "a" * 64
        )
        self.assertNotIn("leashed_facing", publication.model_dump())
        sample = motion_timeline(publication, yaw_quaternion(0.0)).sample(0, yaw_quaternion(0.0))
        assert isinstance(sample, NativePlannerSample)
        self.assertAlmostEqual(facing_of(sample), math.radians(170))


if __name__ == "__main__":
    unittest.main()
