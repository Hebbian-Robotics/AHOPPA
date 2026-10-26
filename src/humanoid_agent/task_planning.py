"""Typed task plans and short-horizon motion decisions for the robot operator."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from humanoid_agent.inspection_report import InspectionReport
from humanoid_agent.motion_types import HybridChunk
from humanoid_agent.validated_types import ImmutableSequence, ValidatedModel


class Subtask(ValidatedModel):
    identifier: str
    description: str
    completion_condition: str
    status: Literal["pending", "active", "complete", "blocked"]


class TaskPlan(ValidatedModel):
    subtasks: ImmutableSequence[Subtask] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_unique_subtask_identifiers(self) -> TaskPlan:
        identifiers = [subtask.identifier for subtask in self.subtasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Subtask identifiers must be unique")
        return self


class PlanningDecision(TaskPlan):
    active_subtask_id: str
    observed_progress: str
    request_task_completion: bool
    inspection_report: InspectionReport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def require_active_subtask_in_plan(self) -> PlanningDecision:
        if self.active_subtask_id not in {subtask.identifier for subtask in self.subtasks}:
            raise ValueError("Active subtask must refer to an identifier in the plan")
        return self


class PlannedHybridChunk(HybridChunk, PlanningDecision):
    facing_yaw_radians: float = Field(ge=-3.141593, le=3.141593)
