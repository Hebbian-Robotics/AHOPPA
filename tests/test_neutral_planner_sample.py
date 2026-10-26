"""Neutral targets share canonical ordering without rebuilding encoder models."""

import unittest
from unittest.mock import patch


class NeutralPlannerSampleTests(unittest.TestCase):
    def test_reuses_encoder_and_preserves_canonical_target_order(self) -> None:
        from unittest.mock import MagicMock

        from humanoid_agent.runtime.camera import NativeVisualState
        from humanoid_agent.runtime.native_control import neutral_planner_sample
        from humanoid_agent.sensor_feedback import EncoderKinematics

        snapshot = MagicMock(spec=NativeVisualState)
        snapshot.joint_names = [f"body_joint_{index}" for index in range(29)]
        encoder = MagicMock(spec=EncoderKinematics)
        encoder.measure.return_value = {
            "tracking_points_m": {
                "head": [7, 8, 9],
                "right_wrist": [4, 5, 6],
                "left_wrist": [1, 2, 3],
            },
            "tracking_orientations_wxyz": {
                "head": [0, 0, 0, 1],
                "right_wrist": [0, 1, 0, 0],
                "left_wrist": [1, 0, 0, 0],
            },
        }
        with patch("humanoid_agent.runtime.native_control.EncoderKinematics") as encoder_factory:
            sample = neutral_planner_sample(snapshot, encoder=encoder)
        encoder_factory.assert_not_called()
        encoder.measure.assert_called_once_with(snapshot.robot_sensors.return_value)
        self.assertEqual(list(sample.vr_position_body_frame), list(range(1, 10)))
        self.assertEqual(
            list(sample.vr_orientation_body_frame_wxyz), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        )
        self.assertEqual((sample.locomotion_mode, sample.speed_m_per_s), (0, 0))
        self.assertEqual(sample.body_height_m, -1)
        self.assertTrue(sample.endpoint_hold)
