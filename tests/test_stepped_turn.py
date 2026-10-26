"""A stepped turn advances in settled facing steps of at most 45 degrees and never walks."""

import math
import unittest

from humanoid_agent.stepped_turn import (
    FACING_STEP_RADIANS,
    FINAL_HEADING_TOLERANCE_RADIANS,
    SETTLED_HOLD_SECONDS,
    SteppedTurn,
    wrapped,
)


def run_turn(
    turn_degrees: float, *, shortfall_degrees: float | None
) -> tuple[SteppedTurn, list[float], float]:
    """Simulate a robot that settles short of each commanded step, or never turns."""
    turn = SteppedTurn(0.0, math.radians(turn_degrees), 0.0)
    heading = 0.0
    commanded: list[float] = []
    elapsed = 0.0
    while turn.status == "turning" and elapsed < 120:
        target = turn.step_target_radians
        if not commanded or abs(wrapped(target - commanded[-1])) > 1e-9:
            commanded.append(target)
            if shortfall_degrees is not None:
                step = wrapped(target - heading)
                shortfall = min(abs(step), math.radians(shortfall_degrees))
                heading = wrapped(target - math.copysign(shortfall, step))
        turn.update(elapsed, heading, pelvis_stable=True)
        elapsed += 0.02
    return turn, commanded, heading


class SteppedTurnTests(unittest.TestCase):
    def test_half_turn_steps_at_most_45_degrees_and_waits_to_settle(self) -> None:
        turn, commanded, heading = run_turn(180, shortfall_degrees=0)
        self.assertEqual(turn.status, "done")
        self.assertEqual(len(commanded), 4)
        previous = 0.0
        for target in commanded:
            self.assertLessEqual(abs(wrapped(target - previous)), FACING_STEP_RADIANS + 1e-9)
            self.assertGreater(wrapped(target - previous), 0)
            previous = target
        for seconds in turn.step_settle_seconds:
            self.assertGreaterEqual(seconds, SETTLED_HOLD_SECONDS)
        self.assertLessEqual(abs(wrapped(math.pi - heading)), FINAL_HEADING_TOLERANCE_RADIANS)

    def test_turn_completes_when_each_step_settles_short(self) -> None:
        turn, commanded, heading = run_turn(-90, shortfall_degrees=10)
        self.assertEqual(turn.status, "done")
        self.assertGreater(len(commanded), 2)
        self.assertLessEqual(abs(wrapped(-math.pi / 2 - heading)), FINAL_HEADING_TOLERANCE_RADIANS)

    def test_turn_gives_up_when_the_heading_never_moves(self) -> None:
        turn, commanded, heading = run_turn(90, shortfall_degrees=None)
        self.assertEqual(turn.status, "timed_out")
        self.assertEqual(len(commanded), 1)
        self.assertEqual(heading, 0.0)

    def test_turn_larger_than_half_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SteppedTurn(0.0, math.radians(200), 0.0)


if __name__ == "__main__":
    unittest.main()
