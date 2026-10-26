"""Private render transforms must become hardware-observable, body-frame feedback."""

import unittest

import numpy as np

from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.runtime.camera import (
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    NativeVisualBatch,
    NativeVisualState,
)


class NativeCameraFeedbackTests(unittest.TestCase):
    def test_rotated_imu_feedback_omits_world_position_and_preserves_encoders(self) -> None:
        state = NativeVisualState(
            identifier="state_000001",
            episode_id="native_camera_check",
            episode_deadline_elapsed_wall_seconds=25,
            simulation_seconds=1,
            elapsed_wall_seconds=1.1,
            source_monotonic_seconds=100,
            joint_names=list(JOINT_NAMES),
            joint_positions=[index / 100 for index in range(29)],
            joint_velocities=[0.01] * 29,
            root_state_world=[3, 4, 0.76, 2**-0.5, 2**-0.5, 0, 0, 0, 0, 0, 0, 0, 2],
        )
        sensors = state.robot_sensors()
        np.testing.assert_allclose(
            sensors.imu_angular_velocity_body_rad_per_s, [0, 2, 0], atol=1e-12
        )
        np.testing.assert_allclose(sensors.imu_gravity_direction_body, [0, -1, 0], atol=1e-12)
        self.assertEqual(list(sensors.joint_positions_rad), state.joint_positions)
        self.assertEqual(list(sensors.joint_velocities_rad_per_s), state.joint_velocities)
        shifted = state.model_copy(
            update={"root_state_world": [300, -40, 20, *state.root_state_world[3:]]}
        )
        self.assertEqual(shifted.robot_sensors(), sensors)
        self.assertNotIn("root_state_world", sensors.model_dump())

    def test_tiled_capture_rejects_cross_robot_and_cross_batch_acknowledgments(self) -> None:
        states = [
            NativeVisualState(
                identifier=f"frame_{index}",
                episode_id=f"episode_{index}",
                episode_deadline_elapsed_wall_seconds=20,
                simulation_seconds=1,
                elapsed_wall_seconds=1,
                source_monotonic_seconds=100,
                joint_names=list(JOINT_NAMES),
                joint_positions=[0.0] * 29,
                joint_velocities=[0.0] * 29,
                root_state_world=[index * 12, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            for index in range(2)
        ]
        batch = NativeVisualBatch(identifier="batch_a", states=states)
        records = [
            {
                "environment_index": index,
                "episode_id": state.episode_id,
                "identifier": state.identifier,
            }
            for index, state in enumerate(states)
        ]
        batch.require_matching_receipt(
            CameraCaptureCompleted(
                identifier="batch_a",
                captures=[CameraCaptureIdentity.model_validate(record) for record in records],
            )
        )
        for receipt in (
            {"identifier": "batch_b", "records": records},
            {"identifier": "batch_a", "records": list(reversed(records))},
            {"identifier": "batch_a", "records": records[:1]},
        ):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                batch.require_matching_receipt(
                    CameraCaptureCompleted.model_validate(
                        {"identifier": receipt["identifier"], "captures": receipt["records"]}
                    )
                )


if __name__ == "__main__":
    unittest.main()
