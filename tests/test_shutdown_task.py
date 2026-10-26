"""Shutdown scoring needs contact-caused closure, untouched closed drawers and a real return."""

import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from humanoid_agent.analysis.summarize_shutdown_round import latex_table
from humanoid_agent.analysis.summarize_shutdown_round import summarize as summarize_shutdown
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.experiment_records import TASK_OUTCOME_ADAPTER, ShutdownTaskOutcome
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.recording import read_record_bytes
from humanoid_agent.runtime.actor import RobotEpisode
from humanoid_agent.runtime.camera import NativeVisualState
from humanoid_agent.runtime.evaluation import GroundTaskEvaluation, GroundTaskMeasurements
from humanoid_agent.runtime.measurements import shutdown_measurements
from humanoid_agent.shutdown_task import (
    SHUTDOWN_CONDITIONS,
    STORAGE_CONDITIONS,
    balanced_condition_schedule,
    make_shutdown_task,
    make_storage_round_task,
    shutdown_instruction,
)
from humanoid_agent.sonic_planner_input import NativePlannerSample
from humanoid_agent.task_outcome import EpisodeEndReason

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def standing_at(horizontal: float, lateral: float, **drawers: float) -> GroundTaskMeasurements:
    return GroundTaskMeasurements(
        root_height_m=0.76,
        root_distance_from_start_m=math.hypot(horizontal, lateral),
        horizontal_speed_m_per_s=0,
        gravity_z=-1,
        pelvis_floor_force_n=0,
        torso_floor_force_n=0,
        left_foot_floor_force_n=30,
        right_foot_floor_force_n=30,
        left_knee_angle_rad=0.2,
        right_knee_angle_rad=0.2,
        **drawers,
    )


def push_drawer_a(
    evaluation: GroundTaskEvaluation, *, contact_field: str
) -> GroundTaskMeasurements:
    """Close drawer A from 0.2 m in 1 cm steps while the named contact is active."""
    pose = standing_at(2.1, 1.0, drawer_a_position_m=0.2)
    evaluation.update(pose)
    for step in range(1, 21):
        pose = standing_at(
            2.1,
            1.0,
            drawer_a_position_m=0.2 - 0.01 * step,
            drawer_a_velocity_m_per_s=-0.5,
            **{contact_field: 3.0},
        )
        evaluation.update(pose)
    return standing_at(2.1, 1.0)


def run_round(evaluation: GroundTaskEvaluation, *, contact_field: str) -> None:
    settled = push_drawer_a(evaluation, contact_field=contact_field)
    for _ in range(60):
        evaluation.update(settled)
    for _ in range(60):
        evaluation.update(standing_at(0, 0))


class ShutdownTaskTests(unittest.TestCase):
    def test_conditions_change_only_private_drawer_states(self) -> None:
        tasks = [make_shutdown_task(condition) for condition in SHUTDOWN_CONDITIONS]
        self.assertEqual(len({shutdown_instruction(task) for task in tasks}), 1)
        self.assertEqual(
            {tuple(c.center_xy_m for c in task.cabinets) for task in tasks},
            {((2.9, 1.0), (2.9, -1.0))},
        )
        self.assertEqual(
            [tuple(c.initially_open for c in task.cabinets) for task in tasks],
            [(False, False), (True, False), (False, True), (True, True)],
        )
        with self.assertRaisesRegex(ValidationError, "not in the scene"):
            make_shutdown_task("both-open", cabinet_count=1)

    def test_schedule_blocks_balance_every_condition(self) -> None:
        schedule = balanced_condition_schedule(3, seed=7)
        self.assertEqual(Counter(schedule), dict.fromkeys(SHUTDOWN_CONDITIONS, 3))
        for block in range(3):
            self.assertEqual(set(schedule[4 * block : 4 * block + 4]), set(SHUTDOWN_CONDITIONS))
        self.assertEqual(schedule, balanced_condition_schedule(3, seed=7))

    def test_scene_files_share_geometry_and_differ_only_in_drawer_states(self) -> None:
        scenes = {
            condition: GroundWorkcellScene.model_validate_json(
                (PROJECT_ROOT / f"configs/shutdown-storage-{condition}.json").read_text()
            )
            for condition in STORAGE_CONDITIONS
        }
        for condition, scene in scenes.items():
            self.assertEqual(scene.shutdown.condition, condition)
            self.assertEqual(scene.shutdown, make_storage_round_task(condition))
        geometry = {
            scene.model_copy(update={"shutdown": None}).model_dump_json()
            for scene in scenes.values()
        }
        self.assertEqual(len(geometry), 1)

    def test_hand_push_then_supported_return_meets_the_physical_criterion(self) -> None:
        evaluation = GroundTaskEvaluation(make_shutdown_task("first-open"))
        run_round(evaluation, contact_field="drawer_a_hand_normal_force_n")
        self.assertTrue(evaluation.activated)
        self.assertTrue(evaluation.physical_criterion_met)
        self.assertAlmostEqual(
            evaluation.drawers["A"].closing_displacement_m["hand"], 0.2, places=6
        )
        self.assertFalse(evaluation.drawer_disturbed("B"))

    def test_return_before_closure_and_open_drawers_never_count(self) -> None:
        evaluation = GroundTaskEvaluation(make_shutdown_task("first-open"))
        for _ in range(100):
            evaluation.update(standing_at(0, 0, drawer_a_position_m=0.2))
        self.assertFalse(evaluation.activated)
        self.assertFalse(evaluation.physical_criterion_met)

    def test_disturbed_closed_drawer_is_recorded_even_if_reclosed(self) -> None:
        evaluation = GroundTaskEvaluation(make_shutdown_task("none-open"))
        evaluation.update(standing_at(2.1, -1.0))
        evaluation.update(standing_at(2.1, -1.0, drawer_b_position_m=0.05))
        # One second to settle closed, then one second held at home.
        for _ in range(120):
            evaluation.update(standing_at(0, 0))
        self.assertTrue(evaluation.drawer_disturbed("B"))
        self.assertTrue(evaluation.physical_criterion_met)

    def test_production_record_separates_hand_and_torso_closure(self) -> None:
        for contact_field, expected_by_hand in (
            ("drawer_a_hand_normal_force_n", True),
            ("drawer_a_body_normal_force_n", False),
        ):
            outcome = self.finish_round(contact_field, declare=False)
            assert outcome.shutdown is not None
            drawer_a, drawer_b = outcome.shutdown.drawers
            self.assertEqual(drawer_a.closed_by_hand, expected_by_hand)
            self.assertTrue(drawer_a.closed_and_settled)
            self.assertFalse(drawer_b.initially_open)
            self.assertTrue(outcome.physical_criterion_met)
            self.assertTrue(outcome.shutdown.requires_manipulation)
            # A physically complete round without a declaration is preserved, not a success.
            self.assertEqual(outcome.end_reason, EpisodeEndReason.WALL_TIME_LIMIT)
            self.assertFalse(outcome.agent_requested_completion)
            self.assertFalse(outcome.task_success)
            self.assertIsNotNone(outcome.shutdown.first_physical_completion_seconds)
            self.assertIsNone(outcome.shutdown.completion_declaration_elapsed_seconds)

    def test_declared_complete_round_records_declaration_time(self) -> None:
        outcome = self.finish_round("drawer_a_hand_normal_force_n", declare=True)
        assert outcome.shutdown is not None
        self.assertTrue(outcome.task_success)
        self.assertEqual(outcome.task_protocol, "drawer-shutdown-v1")
        self.assertEqual(outcome.shutdown.completion_declaration_elapsed_seconds, 20)
        inconsistent = outcome.model_dump(mode="json")
        inconsistent["shutdown"]["returned_home_upright"] = False
        with self.assertRaisesRegex(ValidationError, "closure and return home"):
            TASK_OUTCOME_ADAPTER.validate_python(inconsistent)

    def finish_round(self, contact_field: str, *, declare: bool) -> ShutdownTaskOutcome:
        initial = NativeVisualState(
            identifier="initial",
            episode_id="shutdown_record",
            episode_deadline_elapsed_wall_seconds=60,
            simulation_seconds=0,
            elapsed_wall_seconds=0,
            source_monotonic_seconds=100,
            joint_names=JOINT_NAMES,
            joint_positions=[0.0] * 29,
            joint_velocities=[0.0] * 29,
            root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        neutral = NativePlannerSample(
            locomotion_mode=0,
            movement_direction_world=[1, 0, 0],
            facing_direction_world=[1, 0, 0],
            speed_m_per_s=0,
            body_height_m=-1,
            vr_position_body_frame=[0.0] * 9,
            vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
            endpoint_hold=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "physics").mkdir()
            (directory / "commands").mkdir()
            actor = RobotEpisode(
                directory, initial, neutral, [0, 0, 0], make_shutdown_task("first-open")
            )
            try:
                run_round(actor.evaluation, contact_field=contact_field)
                actor.finish_reason = actor.evaluation.terminal_reason(
                    standing_at(0, 0), completion_requested=declare, deadline_reached=not declare
                )
                actor.finish(20, 4000)
            finally:
                actor.io.close()
            outcome = TASK_OUTCOME_ADAPTER.validate_json(
                read_record_bytes(directory / "physics/task-outcome.json")
            )
            assert isinstance(outcome, ShutdownTaskOutcome)
            return outcome


class ShutdownSummaryTests(unittest.TestCase):
    def test_controls_stay_separate_and_false_claims_are_counted(self) -> None:
        def row(condition: str, **values: object) -> dict[str, object]:
            return {
                "provider": "sol",
                "condition": condition,
                "requires_manipulation": condition != "none-open",
                "execution_status": "completed",
                "timing_valid": False,
                "maximum_physics_lag_seconds": 0.12,
                "end_reason": "agent_declared_completion",
                "declared_completion": True,
                "physical_criterion_met": False,
                "task_success": False,
                "physical_completion_observed": False,
                "declaration_seconds": 300.0,
                "open_drawers": 0 if condition == "none-open" else 1,
                "open_drawers_closed": 0,
                "open_drawers_closed_by_hand": 0,
                "closed_drawers_disturbed": 0,
                **values,
            }

        summary = summarize_shutdown(
            [
                row("none-open", physical_criterion_met=True, task_success=True),
                row("first-open"),
                row(
                    "second-open",
                    physical_criterion_met=True,
                    task_success=True,
                    physical_completion_observed=True,
                    open_drawers_closed=1,
                    timing_valid=True,
                ),
            ]
        )["sol"]
        self.assertEqual(summary["control"]["task_success"], 1)
        self.assertEqual(summary["manipulation"]["completed"], 2)
        self.assertEqual(summary["manipulation"]["task_success"], 1)
        self.assertEqual(summary["manipulation"]["task_success_timing_valid"], 1)
        self.assertEqual(summary["manipulation"]["declared_without_physical_completion"], 1)
        self.assertEqual(summary["manipulation"]["open_drawers_closed"], 1)
        self.assertIn("1/2 & 1 & 1 & 0 & 1 & 1/1", latex_table({"sol": summary}))


class ShutdownMeasurementTests(unittest.TestCase):
    def test_drawer_forces_separate_hand_bodies_and_keep_the_peak_step(self) -> None:
        task = make_shutdown_task("first-open", cabinet_count=1)
        names = ["pelvis", "torso_link", "right_wrist_yaw_link", "right_hand_index_1_link"]
        history = np.zeros((1, 4, len(names), 2, 3))
        history[0, 1, 2, 0] = [3.0, 4.0, 0.0]  # wrist: 5 N in step 1
        history[0, 1, 3, 1] = [0.0, 0.0, 2.0]  # fingertip: 2 N in step 1
        history[0, 3, 1, 0] = [0.0, 1.0, 0.0]  # torso: 1 N in step 3
        fields = {
            "cabinet_A_joint_positions": np.asarray([[0.0, 0.0, 0.0, 0.12]]),
            "cabinet_A_joint_velocities": np.asarray([[0.0, 0.0, 0.0, -0.3]]),
            "cabinet_A_drawer_normal_force_history": history,
        }
        joints = ["door_left_joint", "door_right_joint", "drawer_bottom_joint", "drawer_top_joint"]
        measured = shutdown_measurements(fields, task, joints, names, 0)
        self.assertEqual(measured["drawer_a_position_m"], 0.12)
        self.assertEqual(measured["drawer_a_velocity_m_per_s"], -0.3)
        self.assertAlmostEqual(measured["drawer_a_hand_normal_force_n"], 7.0)
        self.assertAlmostEqual(measured["drawer_a_body_normal_force_n"], 1.0)
        with self.assertRaisesRegex(ValueError, "layout"):
            shutdown_measurements(fields, task, joints, names[:3], 0)

    def test_measurements_reject_nonfinite_drawer_state(self) -> None:
        evaluation = GroundTaskEvaluation(make_shutdown_task("first-open"))
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluation.update(replace(standing_at(0, 0), drawer_a_position_m=math.nan))


if __name__ == "__main__":
    unittest.main()


class StorageVisibilityTests(unittest.TestCase):
    def test_lockers_hide_far_fronts_and_nearby_fronts_are_seen(self) -> None:
        from humanoid_agent.runtime.measurements import (
            TOP_DRAWER_FRONT_CENTER_HEIGHT_M,
            storage_occluders,
        )
        from humanoid_agent.shutdown_visibility import front_in_view

        scene = GroundWorkcellScene.model_validate_json(
            (PROJECT_ROOT / "configs/shutdown-storage-open-A.json").read_text()
        )
        assert scene.shutdown is not None
        occluders = storage_occluders(scene)
        self.assertEqual(len(occluders), 3)
        cabinet = scene.shutdown.cabinet("A")
        front = np.asarray([*cabinet.drawer_front_xy_m(0.2), TOP_DRAWER_FRONT_CENTER_HEIGHT_M])
        normal = np.asarray(cabinet.outward_direction_xy)
        facing_east = np.asarray([0.0, 0.0, 0.0, 1.0])
        # Standing in the east aisle about 1.1 m from the open drawer, facing it.
        self.assertTrue(
            front_in_view(np.asarray([3.4, 1.5, 0.95]), facing_east, front, normal, occluders)
        )
        # From home the lockers block the line of sight and the front is out of range.
        self.assertFalse(
            front_in_view(np.asarray([0.0, 0.0, 0.95]), facing_east, front, normal, occluders)
        )
        # Behind the cabinet (wrong side) the front is never counted.
        facing_west = np.asarray([0.0, 0.0, 1.0, 0.0])
        self.assertFalse(
            front_in_view(np.asarray([6.2, 1.5, 0.95]), facing_west, front, normal, occluders)
        )

    def test_storage_conditions_cover_zero_to_two_open_drawers(self) -> None:
        from humanoid_agent.shutdown_task import STORAGE_CONDITIONS, make_storage_round_task

        self.assertEqual(len(STORAGE_CONDITIONS), 11)
        open_counts = Counter(
            sum(c.initially_open for c in make_storage_round_task(condition).cabinets)
            for condition in STORAGE_CONDITIONS
        )
        self.assertEqual(open_counts, {0: 1, 1: 4, 2: 6})
        instructions = {
            shutdown_instruction(make_storage_round_task(condition))
            for condition in STORAGE_CONDITIONS
        }
        self.assertEqual(len(instructions), 1)
        self.assertIn("storage lockers", next(iter(instructions)))
