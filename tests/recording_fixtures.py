"""Seed historical JSON fixtures for read-only compatibility tests."""

from pathlib import Path

from pydantic import BaseModel

from humanoid_agent.artifact_io import atomic_json


def write_test_record(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload.model_dump(mode="json"))


def shutdown_task_outcome_record(
    *,
    end_reason: str,
    physical_criterion_met: bool,
    agent_requested_completion: bool,
    task_success: bool,
) -> dict[str, object]:
    """A one-cabinet, nothing-open shutdown outcome with the given completion evidence."""
    return {
        "end_reason": end_reason,
        "physical_criterion_met": physical_criterion_met,
        "agent_requested_completion": agent_requested_completion,
        "task_success": task_success,
        "completion_protocol": "agent-declaration-then-independent-verification-v1",
        "balance_contact_protocol": "ground-reaching-filtered-floor-contact-v2",
        "task_protocol": "drawer-shutdown-v1",
        "shutdown": {
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
            "returned_home_upright": physical_criterion_met,
            "first_physical_completion_seconds": 1.0 if physical_criterion_met else None,
            "completion_declaration_elapsed_seconds": 20.0 if agent_requested_completion else None,
        },
    }
