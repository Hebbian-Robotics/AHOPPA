"""Protect completion claims and independently verified outcomes without model calls."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.runtime.evaluation import GroundTaskEvaluation, GroundTaskMeasurements
from humanoid_agent.shutdown_task import make_storage_round_task
from humanoid_agent.task_balance import (
    GROUND_REACHING_BALANCE,
    STANDING_DIAGNOSTIC_BALANCE,
    forbidden_floor_contact,
)
from humanoid_agent.task_outcome import EpisodeEndReason, TaskOutcome
from humanoid_agent.task_planning import PlanningDecision, Subtask, TaskPlan


class TaskOutcomeTests(unittest.TestCase):
    def test_forbidden_floor_contacts_reject_supported_torso_but_permit_kneeling(self) -> None:
        self.assertFalse(GROUND_REACHING_BALANCE.is_fallen(0.25, -1.0))
        self.assertFalse(forbidden_floor_contact(0.0, 0.0))
        for pelvis_force, torso_force in [(1.0, 0.0), (0.0, 1.0), (0.01, 0.01)]:
            with self.subTest(pelvis=pelvis_force, torso=torso_force):
                self.assertTrue(forbidden_floor_contact(pelvis_force, torso_force))
        for invalid_force in [-1.0, float("nan"), float("inf")]:
            with self.subTest(force=invalid_force), self.assertRaises(ValueError):
                forbidden_floor_contact(invalid_force, 0.0)

    def test_fall_precedes_completion_and_timeout(self) -> None:
        evaluation = GroundTaskEvaluation(make_storage_round_task("none-open"))
        fallen = GroundTaskMeasurements(0.43, 0.1, 0, -1, 1.0, 0)
        reason = evaluation.terminal_reason(
            fallen, completion_requested=True, deadline_reached=True
        )
        self.assertEqual(reason, EpisodeEndReason.ROBOT_FELL)
        assert reason is not None
        outcome = TaskOutcome(reason, evaluation.physical_criterion_met)
        self.assertFalse(outcome.task_success)
        self.assertFalse(outcome.agent_requested_completion)

    def test_ground_balance_allows_deliberate_lowering_but_rejects_collapse_and_tipping(
        self,
    ) -> None:
        self.assertTrue(STANDING_DIAGNOSTIC_BALANCE.is_fallen(0.25, -1.0))
        for height_m in [0.2, 0.25, 0.35, 0.75]:
            with self.subTest(height_m=height_m):
                self.assertFalse(GROUND_REACHING_BALANCE.is_fallen(height_m, -1.0))
        self.assertTrue(GROUND_REACHING_BALANCE.is_fallen(0.1, -1.0))
        self.assertTrue(GROUND_REACHING_BALANCE.is_fallen(0.75, 0.0))
        for height_m, gravity_z in [(float("nan"), -1.0), (0.75, float("inf"))]:
            with (
                self.subTest(height_m=height_m, gravity_z=gravity_z),
                self.assertRaises(ValueError),
            ):
                GROUND_REACHING_BALANCE.is_fallen(height_m, gravity_z)

    def test_success_requires_a_completion_declaration_and_the_physical_criterion(self) -> None:
        cases = [
            (EpisodeEndReason.AGENT_DECLARED_COMPLETION, True, True),
            (EpisodeEndReason.AGENT_DECLARED_COMPLETION, False, False),
            (EpisodeEndReason.WALL_TIME_LIMIT, True, False),
            (EpisodeEndReason.MOTION_CHUNK_LIMIT, True, False),
            (EpisodeEndReason.ROBOT_FELL, True, False),
            (EpisodeEndReason.EXECUTION_ERROR, True, False),
        ]
        for end_reason, physical_criterion_met, expected_success in cases:
            with self.subTest(reason=end_reason, physical=physical_criterion_met):
                outcome = TaskOutcome(end_reason, physical_criterion_met)
                self.assertEqual(outcome.task_success, expected_success)

    def test_invalid_subtask_references_are_rejected_before_execution(self) -> None:
        subtask = Subtask(
            identifier="approach",
            description="Approach the visible target",
            completion_condition="Target is within reach",
            status="active",
        )
        with self.assertRaisesRegex(ValidationError, "unique"):
            TaskPlan(subtasks=[subtask, subtask])
        with self.assertRaisesRegex(ValidationError, "Active subtask"):
            PlanningDecision(
                subtasks=[subtask],
                active_subtask_id="missing",
                observed_progress="Target is visible",
                request_task_completion=False,
            )

    def test_nonfinite_record_does_not_replace_the_last_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.json"
            atomic_json(destination, {"task_success": False})
            with self.assertRaises(ValueError):
                atomic_json(destination, {"distance": float("nan")})
            self.assertEqual(json.loads(destination.read_text()), {"task_success": False})


if __name__ == "__main__":
    unittest.main()
