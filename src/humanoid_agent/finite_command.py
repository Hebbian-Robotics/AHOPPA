"""Episode and observation identities shared by finite motion transports."""

from typing import Annotated

from pydantic import Field, FiniteFloat

from humanoid_agent.inspection_report import InspectionReport
from humanoid_agent.validated_types import ValidatedModel

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9_-]+$")]


class FiniteCommandIdentity(ValidatedModel):
    """Admission deadline does not shorten an already admitted command."""

    identifier: Identifier
    source_response_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episode_id: Identifier
    observation_id: Identifier
    observation_elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    valid_until_elapsed_wall_seconds: FiniteFloat = Field(gt=0)
    request_task_completion: bool
    inspection_report: InspectionReport | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
