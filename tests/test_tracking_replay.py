"""Recorded SDK references use their actual coordinate convention in 3D replays."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.presentation.recorded_motion import recorded_targets_world


class TrackingReplayTests(unittest.TestCase):
    def test_hybrid_reference_uses_current_body_tilt_and_translation(self) -> None:
        quaternion = Rotation.from_euler("xyz", [0.2, 0.3, 0.7]).as_quat(scalar_first=True)
        root = [1.0, 2.0, 0.8, *quaternion]
        points = np.asarray([[0.3, 0.2, -0.1], [0.3, -0.2, -0.1], [0, 0, 0.4]])
        result = recorded_targets_world(
            {"vr_position_body_frame": points.reshape(-1).tolist()}, root
        )
        expected = Rotation.from_quat(quaternion, scalar_first=True).apply(points) + root[:3]
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_nonfinite_or_wrong_size_target_records_fail_closed(self) -> None:
        root = [0, 0, 1, 1, 0, 0, 0]
        with self.assertRaises(ValueError):
            recorded_targets_world({"vr_position_body_frame": [float("nan")] * 9}, root)
        with self.assertRaises(ValueError):
            recorded_targets_world({"vr_position_body_frame": [0.0] * 6}, root)
