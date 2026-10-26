"""Shutdown outcome records accept only drawer-shutdown evidence."""

import unittest

from pydantic import ValidationError

from humanoid_agent.experiment_records import TASK_OUTCOME_ADAPTER, ShutdownTaskOutcome
from humanoid_agent.task_balance import ISAAC_CONTACT_BALANCE_PROTOCOL
from humanoid_agent.task_outcome import COMPLETION_PROTOCOL


class TaskOutcomeVariantTests(unittest.TestCase):
    def completion_evidence(self) -> dict[str, object]:
        return {
            "end_reason": "wall_time_limit",
            "physical_criterion_met": False,
            "agent_requested_completion": False,
            "task_success": False,
            "completion_protocol": COMPLETION_PROTOCOL,
            "balance_contact_protocol": ISAAC_CONTACT_BALANCE_PROTOCOL,
        }

    def shutdown_outcome(self) -> dict[str, object]:
        return {
            "condition": "none-open",
            "requires_manipulation": False,
            "drawers": [
                {
                    "cabinet": "A",
                    "initially_open": False,
                    "initial_position_m": 0.0,
                    "final_position_m": 0.0,
                    "maximum_position_m": 0.0,
                    "closed_and_settled": True,
                    "disturbed": False,
                    "closed_by_hand": False,
                    "closing_displacement_by_latest_contact_m": {
                        "hand": 0.0,
                        "other_body": 0.0,
                        "none": 0.0,
                    },
                    "hand_contact_seconds": 0.0,
                    "other_body_contact_seconds": 0.0,
                }
            ],
            "all_drawers_closed_and_settled": True,
            "returned_home_upright": False,
            "first_physical_completion_seconds": None,
            "completion_declaration_elapsed_seconds": None,
        }

    def test_shutdown_outcome_rejects_evidence_from_earlier_tasks(self) -> None:
        record = {
            **self.completion_evidence(),
            "task_protocol": "drawer-shutdown-v1",
            "shutdown": self.shutdown_outcome(),
        }
        for foreign_evidence in (
            {"transfer_condition": "same-table"},
            {"object_lifted": True},
            {"maximum_carried_distance_m": 0.1},
            {"mobility_condition": "walk-and-stop"},
            {"mobility_stage_achieved": True},
            {"inspection_report_correct": False},
            {"inspection_visited_stations": ["A"]},
        ):
            with self.subTest(evidence=foreign_evidence), self.assertRaises(ValidationError):
                TASK_OUTCOME_ADAPTER.validate_python({**record, **foreign_evidence})

    def test_recorded_null_fields_round_trip_and_unknown_protocols_fail_closed(self) -> None:
        record = {
            **self.completion_evidence(),
            "task_protocol": "drawer-shutdown-v1",
            "shutdown": self.shutdown_outcome(),
            "transfer_condition": None,
            "mobility_condition": None,
            "inspection_report": None,
        }
        outcome = TASK_OUTCOME_ADAPTER.validate_python(record)
        serialized = outcome.model_dump(mode="json")
        self.assertIsNone(serialized["transfer_condition"])
        self.assertNotIn("inspection_report_correct", serialized)
        self.assertEqual(TASK_OUTCOME_ADAPTER.validate_json(outcome.model_dump_json()), outcome)
        for protocol in ("marker-return-v1", "future-task"):
            with self.subTest(protocol=protocol), self.assertRaises(ValidationError):
                TASK_OUTCOME_ADAPTER.validate_python({**record, "task_protocol": protocol})
        with self.assertRaises(ValidationError):
            TASK_OUTCOME_ADAPTER.validate_python({**record, "shutdown": None})

    def test_both_shutdown_protocols_preserve_four_cabinet_visibility_evidence(self) -> None:
        drawers = [
            {
                "cabinet": cabinet,
                "initially_open": False,
                "initial_position_m": 0.0,
                "final_position_m": 0.0,
                "maximum_position_m": 0.0,
                "closed_and_settled": True,
                "disturbed": False,
                "closed_by_hand": False,
                "closing_displacement_by_latest_contact_m": {
                    "hand": 0.0,
                    "other_body": 0.0,
                    "none": 0.0,
                },
                "hand_contact_seconds": 0.0,
                "other_body_contact_seconds": 0.0,
                "front_viewed": True,
                "first_viewed_seconds": 1.0,
            }
            for cabinet in ("A", "B", "C", "D")
        ]
        for protocol in ("drawer-shutdown-v1", "drawer-shutdown-v2"):
            outcome = TASK_OUTCOME_ADAPTER.validate_python(
                {
                    **self.completion_evidence(),
                    "task_protocol": protocol,
                    "shutdown": {
                        "condition": "none-open",
                        "requires_manipulation": False,
                        "drawers": drawers,
                        "all_drawers_closed_and_settled": True,
                        "returned_home_upright": False,
                        "first_physical_completion_seconds": None,
                        "completion_declaration_elapsed_seconds": None,
                    },
                }
            )
            assert isinstance(outcome, ShutdownTaskOutcome)
            self.assertEqual(outcome.task_protocol, protocol)
            self.assertEqual(len(outcome.shutdown.drawers), 4)
            self.assertTrue(outcome.shutdown.drawers[0].front_viewed)
            self.assertEqual(TASK_OUTCOME_ADAPTER.validate_json(outcome.model_dump_json()), outcome)
