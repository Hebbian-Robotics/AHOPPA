"""Check the scripted standing command independently of simulation and inference."""

import math
import unittest

from humanoid_agent.runtime.scripted_control import StandingControlSource
from humanoid_agent.sonic_planner_input import NativePlannerSample


def neutral_sample() -> NativePlannerSample:
    return NativePlannerSample(
        locomotion_mode=0,
        movement_direction_world=[1, 0, 0],
        facing_direction_world=[1, 0, 0],
        speed_m_per_s=-1,
        body_height_m=-1,
        vr_position_body_frame=[0.2, 0.2, 0.1, 0.2, -0.2, 0.1, 0, 0, 0.4],
        vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
        endpoint_hold=False,
    )


class StandingControlTests(unittest.TestCase):
    def test_standing_check_holds_neutral_without_walking_reaching_or_closing_hands(self) -> None:
        neutral = neutral_sample()
        source = StandingControlSource(neutral, 5)
        self.assertEqual(source.duration_seconds, 5)
        for elapsed in (0, 1, 4.9, 5, 25):
            self.assertEqual(source.phase(elapsed), "stand")
            sample = source.sample(elapsed)
            self.assertEqual(sample.locomotion_mode, 0)
            self.assertEqual(sample.speed_m_per_s, -1)
            self.assertEqual(sample.vr_position_body_frame, neutral.vr_position_body_frame)
            self.assertEqual(
                sample.vr_orientation_body_frame_wxyz, neutral.vr_orientation_body_frame_wxyz
            )
            self.assertEqual(sample.right_hand_joints, neutral.right_hand_joints)
            self.assertEqual(sample.endpoint_hold, elapsed >= 5)

    def test_samples_and_constructor_do_not_share_mutable_input(self) -> None:
        neutral = neutral_sample()
        original = neutral.model_dump()
        source = StandingControlSource(neutral)
        held = source.sample(3)
        self.assertEqual(neutral.model_dump(), original)
        changed = held.model_copy(update={"vr_position_body_frame": [100.0] * 9})
        self.assertEqual(changed.vr_position_body_frame[0], 100)
        self.assertNotEqual(held.vr_position_body_frame[0], 100)
        self.assertEqual(
            source.sample(0).vr_position_body_frame[0], original["vr_position_body_frame"][0]
        )

    def test_invalid_durations_and_elapsed_times_fail_before_sampling(self) -> None:
        for duration in (0, 0.99, 30.01, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                StandingControlSource(neutral_sample(), duration)
        source = StandingControlSource(neutral_sample(), 30)
        self.assertEqual(source.duration_seconds, 30)
        for elapsed in (-0.01, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                source.sample(elapsed)
            with self.assertRaises(ValueError):
                source.phase(elapsed)
