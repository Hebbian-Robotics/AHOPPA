"""Validated acknowledgments exchanged by the local operator and physics owner."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from humanoid_agent.command_admission import AdmissionReason, CommandAdmission
from humanoid_agent.finite_command import Identifier
from humanoid_agent.validated_types import ValidatedModel


class IdentifierReceipt(ValidatedModel):
    identifier: Identifier


class CommandStatus(IdentifierReceipt):
    state: Literal["admitted", "duplicate", "rejected"]
    admission: CommandAdmission

    @model_validator(mode="after")
    def require_matching_admission(self) -> CommandStatus:
        expected_state = (
            "admitted"
            if self.admission.reason == AdmissionReason.ACCEPTED
            else "duplicate"
            if self.admission.reason == AdmissionReason.DUPLICATE
            else "rejected"
        )
        if self.identifier != self.admission.identifier or self.state != expected_state:
            raise ValueError("Command status disagrees with its admission")
        return self
