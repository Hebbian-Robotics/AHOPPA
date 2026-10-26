"""The Newton boundary must preserve the robot's measured orientation and motion."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.runtime.state import clock_validity, recorded_state_from_newton


class NewtonStateTests(unittest.TestCase):
    def test_simulation_lag_is_classified_independently_at_the_prespecified_limit(self) -> None:
        for lag, expected in (
            (0.0, "valid"),
            (0.1, "valid"),
            (0.100001, "invalid_physics_lag"),
            (2.0, "invalid_physics_lag"),
        ):
            with self.subTest(lag=lag):
                self.assertEqual(clock_validity(lag)["status"], expected)
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.subTest(lag=invalid), self.assertRaises(ValueError):
                clock_validity(invalid)

    def test_preserves_physical_orientation_and_world_velocities(self) -> None:
        rotation = Rotation.from_euler("xyz", [0.3, -0.2, 1.1])
        source = np.r_[[1, 2, 3], rotation.as_quat(), [2, -1, 0.2], [0.4, 0.5, 0.6]]
        recorded = recorded_state_from_newton(source)
        np.testing.assert_allclose(
            Rotation.from_quat(recorded[3:7], scalar_first=True).apply([0, 0, -1]),
            rotation.apply([0, 0, -1]),
        )
        np.testing.assert_allclose(recorded[:3], [1, 2, 3])
        np.testing.assert_allclose(recorded[7:10], [2, -1, 0.2])
        np.testing.assert_allclose(recorded[10:13], [0.4, 0.5, 0.6])
        for invalid in [np.zeros(13), np.full(13, np.nan), np.zeros(12)]:
            with self.assertRaises(ValueError):
                recorded_state_from_newton(invalid)
