"""Isaac prompts must retain sensor evidence and exclude private evaluator fields."""

import copy
import json
import unittest
from typing import Any

from pydantic import ValidationError

from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.runtime.observations import IsaacObservation


class IsaacObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record: dict[str, Any] = {
            "simulation_seconds": 1.0,
            "elapsed_wall_seconds": 1.5,
            "robot_sensors": {
                "joint_names": ["example_joint"],
                "joint_positions_rad": [0.2],
                "joint_velocities_rad_per_s": [0.1],
                "imu_angular_velocity_body_rad_per_s": [0, 0, 0],
                "imu_gravity_direction_body": [0, 0, -1],
            },
            "depth_tiles_m": [[None] * 16 for _ in range(12)],
            "camera_intrinsic_matrix": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
            "observation_identity": {
                "episode_id": "episode_a",
                "identifier": "feedback_001",
                "captured_elapsed_wall_seconds": 1.5,
                "episode_deadline_elapsed_wall_seconds": 600,
            },
        }

    def test_private_fields_are_dropped_while_sensor_evidence_is_preserved(self) -> None:
        enriched_record = {
            **self.record,
            "root_position": [10, 20, 0.7],
            "button_center_m": [1.6, -0.2, 0.06],
            "physical_criterion_met": True,
            "new_evaluator_measurement": {"distance": 0.01},
        }
        observation = IsaacObservation.model_validate(enriched_record).model_dump()
        self.assertEqual(observation, self.record)
        external_record = {**enriched_record, "external_camera_available": True}
        self.assertEqual(
            IsaacObservation.model_validate(external_record).model_dump(),
            {**self.record, "external_camera_available": True},
        )

    def test_missing_or_malformed_sensor_fields_fail_at_the_boundary(self) -> None:
        for missing_field in ["robot_sensors", "observation_identity", "elapsed_wall_seconds"]:
            incomplete = {
                name: value for name, value in self.record.items() if name != missing_field
            }
            with self.subTest(missing_field=missing_field), self.assertRaises(ValidationError):
                IsaacObservation.model_validate(incomplete)
        for field, invalid_value in [
            ("elapsed_wall_seconds", float("nan")),
            ("simulation_seconds", -1),
            ("depth_tiles_m", [[1] * 16]),
            ("camera_intrinsic_matrix", [[float("inf"), 0, 320], [0, 400, 240], [0, 0, 1]]),
        ]:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                IsaacObservation.model_validate({**self.record, field: invalid_value})
        record = copy.deepcopy(self.record)
        record["robot_sensors"]["target_world_position"] = [1, 2, 3]
        with self.assertRaises(ValidationError):
            IsaacObservation.model_validate(record)

    def test_camera_feedback_must_match_the_requested_episode_observation_and_clock(self) -> None:
        clock = EpisodeClock(
            episode_id="episode_a",
            deadline_monotonic_seconds=1000,
            elapsed_wall_seconds=2,
            deadline_elapsed_wall_seconds=600,
        )
        observation = IsaacObservation.parse_requested_snapshot(
            json.dumps(self.record), identifier="feedback_001", clock=clock
        )
        self.assertEqual(observation.model_dump(), self.record)
        for changes in (
            {"identifier": "other_observation"},
            {"episode_id": "other_episode"},
            {"captured_elapsed_wall_seconds": 1.6},
            {"episode_deadline_elapsed_wall_seconds": 601},
        ):
            with self.subTest(changes=changes):
                record = copy.deepcopy(self.record)
                record["observation_identity"].update(changes)
                with self.assertRaisesRegex(ValueError, "requested observation"):
                    IsaacObservation.parse_requested_snapshot(
                        json.dumps(record), identifier="feedback_001", clock=clock
                    )


if __name__ == "__main__":
    unittest.main()
