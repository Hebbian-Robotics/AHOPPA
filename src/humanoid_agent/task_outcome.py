"""Keep an agent's completion declaration separate from verified task success."""

from dataclasses import dataclass
from enum import StrEnum

COMPLETION_PROTOCOL = "agent-declaration-then-independent-verification-v1"


class EpisodeEndReason(StrEnum):
    AGENT_DECLARED_COMPLETION = "agent_declared_completion"
    WALL_TIME_LIMIT = "wall_time_limit"
    MOTION_CHUNK_LIMIT = "motion_chunk_limit"
    ROBOT_FELL = "robot_fell"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class TaskOutcome:
    end_reason: EpisodeEndReason
    physical_criterion_met: bool

    @property
    def agent_requested_completion(self) -> bool:
        return self.end_reason is EpisodeEndReason.AGENT_DECLARED_COMPLETION

    @property
    def task_success(self) -> bool:
        return self.agent_requested_completion and self.physical_criterion_met
