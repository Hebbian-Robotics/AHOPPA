"""The privileged reference visits every drawer, pushes only open ones and halts safely."""

import itertools
import math
import unittest
from dataclasses import replace

import numpy as np

from humanoid_agent.runtime.evaluation import GroundTaskMeasurements
from humanoid_agent.runtime.shutdown_reference import (
    HOME_CORRECTION_RADIUS_M,
    ShutdownReferenceControl,
)
from humanoid_agent.shutdown_task import ShutdownCondition, make_shutdown_task
from humanoid_agent.sonic_planner_input import NativePlannerSample

NEUTRAL = NativePlannerSample(
    locomotion_mode=0,
    movement_direction_world=[1, 0, 0],
    facing_direction_world=[1, 0, 0],
    speed_m_per_s=0,
    body_height_m=-1,
    vr_position_body_frame=[0.2, 0.2, -0.1, 0.2, -0.2, -0.1, 0, 0, 0.4],
    vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
    endpoint_hold=True,
)


class ToyRound:
    """Turn to the commanded facing and translate by commanded walking; a push closes
    the drawer it faces."""

    def __init__(self, condition: ShutdownCondition, *, reopen_once: str | None = None) -> None:
        self.task = make_shutdown_task(condition)
        self.reference = ShutdownReferenceControl(NEUTRAL, self.task, [0.0, 0.0])
        self.root = np.zeros(2)
        self.yaw = 0.0
        self.drawers = {
            cabinet.identifier: self.task.initial_drawer_position_m(cabinet)
            for cabinet in self.task.cabinets
        }
        self.pushed: list[str] = []
        self.reopen_once = reopen_once
        self.last_direction = np.zeros(2)
        self.carry_ticks = 0

    def run(self, seconds: float) -> None:
        for tick in range(int(seconds / 0.02)):
            elapsed = tick * 0.02
            self.reference.latest_measurement = GroundTaskMeasurements(
                root_height_m=0.78,
                root_distance_from_start_m=float(np.linalg.norm(self.root)),
                horizontal_speed_m_per_s=0,
                gravity_z=-1,
                pelvis_floor_force_n=0,
                torso_floor_force_n=0,
                drawer_a_position_m=self.drawers.get("A", 0.0),
                drawer_b_position_m=self.drawers.get("B", 0.0),
            )
            sample = self.reference.sample(
                elapsed,
                [
                    *self.root,
                    0.78,
                    math.cos(self.yaw / 2),
                    0,
                    0,
                    math.sin(self.yaw / 2),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [[*np.add(self.root, offset), 0.7] for offset in ((0.2, 0.2), (0.2, -0.2))],
            )
            self.yaw = math.atan2(
                sample.facing_direction_world[1], sample.facing_direction_world[0]
            )
            if sample.locomotion_mode == 1:
                self.last_direction = np.asarray(sample.movement_direction_world[:2])
                self.root += self.last_direction * 0.3 * 0.02
                self.carry_ticks = 42
            elif self.carry_ticks > 0:
                # Like the released stop transition, keep moving about 0.25 m after a stop.
                self.root += self.last_direction * 0.3 * 0.02
                self.carry_ticks -= 1
            stage = self.reference.current_stage
            if stage == "push" and self.reference.visit_index < len(self.task.cabinets):
                identifier = self.task.cabinets[self.reference.visit_index].identifier
                self.drawers[identifier] = 0.0
                if identifier not in self.pushed or identifier == self.reopen_once:
                    self.pushed.append(identifier)
            if (
                stage == "lower_hands"
                and self.reopen_once in self.drawers
                and (self.pushed.count(self.reopen_once) == 1)
            ):
                self.drawers[self.reopen_once] = 0.06
                self.reopen_once = None


class ShutdownReferenceTests(unittest.TestCase):
    def test_round_pushes_only_open_drawers_and_returns_home(self) -> None:
        round_ = ToyRound("second-open")
        round_.run(200)
        self.assertEqual(round_.reference.current_stage, "done")
        self.assertIsNone(round_.reference.failure_reason)
        self.assertEqual(round_.pushed, ["B"])
        self.assertEqual(round_.reference.closing_attempts, {"A": 0, "B": 1})
        self.assertLess(float(np.linalg.norm(round_.root)), HOME_CORRECTION_RADIUS_M)

    def test_reopened_drawer_is_pushed_again_before_moving_on(self) -> None:
        round_ = ToyRound("first-open", reopen_once="A")
        round_.run(200)
        self.assertEqual(round_.reference.closing_attempts["A"], 2)
        self.assertEqual(round_.reference.reopened_after_release_m["A"], [0.06])
        self.assertEqual(round_.reference.current_stage, "done")

    def test_timeout_halts_locomotion_instead_of_repeating_a_walk(self) -> None:
        reference = ShutdownReferenceControl(NEUTRAL, make_shutdown_task("both-open"), [0, 0])
        reference.latest_measurement = replace(
            GroundTaskMeasurements(0.78, 0, 0, -1, 5, 0, 0), drawer_a_position_m=0.2
        )
        root = [0, 0, 0.78, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hands = [[0.2, 0.2, 0.7], [0.2, -0.2, 0.7]]
        sample = reference.sample(0, root, hands)
        for tick in range(1, int(90 / 0.02)):
            sample = reference.sample(tick * 0.02, root, hands)
        self.assertEqual(reference.failure_reason, "progress-timeout")
        self.assertEqual(sample.locomotion_mode, 0)
        self.assertTrue(math.isclose(sample.speed_m_per_s, 0))


if __name__ == "__main__":
    unittest.main()


class StorageRouteTests(unittest.TestCase):
    def test_routes_go_around_the_locker_block_and_visit_order_is_greedy(self) -> None:
        from humanoid_agent.runtime.shutdown_reference import plan_route, route_length
        from humanoid_agent.shutdown_visibility import PlanOccluder, segment_crosses_rectangle

        block = PlanOccluder((1.9, 0.0), (0.415, 1.92), 2.09)
        route = plan_route(np.asarray([0.0, 0.0]), np.asarray([3.8, 0.0]), [block])
        points = [np.asarray([0.0, 0.0]), *route]
        self.assertGreater(len(route), 1)
        self.assertFalse(
            any(segment_crosses_rectangle(a, b, block) for a, b in itertools.pairwise(points))
        )
        self.assertGreater(
            route_length(np.asarray([0.0, 0.0]), np.asarray([3.8, 0.0]), [block]), 3.8
        )
        self.assertEqual(
            len(plan_route(np.asarray([0.0, 0.0]), np.asarray([-1.0, 2.0]), [block])), 1
        )
