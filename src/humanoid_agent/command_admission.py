"""Episode-bound admission and durable identities for finite robot commands."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, FiniteFloat

from humanoid_agent.artifact_io import canonical_json_sha256
from humanoid_agent.finite_command import FiniteCommandIdentity, Identifier
from humanoid_agent.recording import read_record_bytes, record_paths
from humanoid_agent.validated_types import ValidatedModel


class ObservationIdentity(ValidatedModel):
    episode_id: Identifier
    identifier: Identifier
    captured_elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    episode_deadline_elapsed_wall_seconds: FiniteFloat = Field(gt=0)


class EpisodeClock(ValidatedModel):
    deadline_monotonic_seconds: FiniteFloat = Field(gt=0)
    episode_id: Identifier
    elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    deadline_elapsed_wall_seconds: FiniteFloat = Field(gt=0)


class AdmissionReason(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    IDENTIFIER_CONFLICT = "identifier_conflict"
    WRONG_EPISODE = "wrong_episode"
    UNKNOWN_OBSERVATION = "unknown_observation"
    OBSERVATION_MISMATCH = "observation_mismatch"
    EXPIRED = "expired"


class CommandAdmission(ValidatedModel):
    identifier: str
    reason: AdmissionReason
    payload_sha256: str
    observed_elapsed_wall_seconds: FiniteFloat = Field(ge=0)
    observation_age_seconds: FiniteFloat | None


class CommandAdmissionLedger:
    """Owned by the simulator thread; duplicate IDs survive broker reconnects.

    This preserves admission evidence, not simulator checkpoint/restart support.
    An admitted command must never be retried blindly after a process crash.
    """

    def __init__(
        self,
        directory: Path,
        episode_id: str,
        *,
        record: Callable[[Path, BaseModel], None],
        restore_records: bool = True,
    ) -> None:
        self.directory = directory
        self.episode_id = episode_id
        self.observation_directory = directory / "observations"
        self.admission_directory = directory / "admissions"
        self.record = record
        # Disk reconstruction is setup-only. Active admission never touches storage.
        self.observations: dict[str, ObservationIdentity] = {}
        for path in record_paths(self.observation_directory) if restore_records else ():
            observation = ObservationIdentity.model_validate_json(read_record_bytes(path))
            if observation.identifier != path.stem or observation.episode_id != episode_id:
                raise ValueError("Stored observation differs from its ledger identity or episode")
            self.observations[observation.identifier] = observation
        self.admissions: dict[str, CommandAdmission] = {}
        for path in record_paths(self.admission_directory) if restore_records else ():
            admission = CommandAdmission.model_validate_json(read_record_bytes(path))
            if admission.identifier != path.stem:
                raise ValueError("Stored admission differs from its ledger identity")
            self.admissions[admission.identifier] = admission

    def register_observation(self, observation: ObservationIdentity) -> None:
        if observation.episode_id != self.episode_id:
            raise ValueError("Cannot register another episode's observation")
        path = self.observation_directory / f"{observation.identifier}.json"
        if observation.identifier in self.observations:
            if self.observations[observation.identifier] != observation:
                raise ValueError("Observation identity cannot be reassigned")
            return
        self.record(path, observation)
        self.observations[observation.identifier] = observation

    def admit(self, command: FiniteCommandIdentity, elapsed_seconds: float) -> CommandAdmission:
        admission = self.assess(command, elapsed_seconds)
        if command.identifier not in self.admissions:
            path = self.admission_directory / f"{command.identifier}.json"
            self.record(path, admission)
            self.admissions[command.identifier] = admission
        return admission

    def assess(self, command: FiniteCommandIdentity, elapsed_seconds: float) -> CommandAdmission:
        """Check admission before preparing motion, without consuming its identity."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Command admission time must be finite and nonnegative")
        digest = canonical_json_sha256(command.model_dump(mode="json"))
        if command.identifier in self.admissions:
            previous = self.admissions[command.identifier]
            if previous.payload_sha256 != digest:
                reason = AdmissionReason.IDENTIFIER_CONFLICT
            elif previous.reason == AdmissionReason.ACCEPTED:
                reason = AdmissionReason.DUPLICATE
            else:
                reason = previous.reason
            return previous.model_copy(
                update={
                    "reason": reason,
                    "observed_elapsed_wall_seconds": elapsed_seconds,
                    "observation_age_seconds": (
                        previous.observation_age_seconds
                        + elapsed_seconds
                        - previous.observed_elapsed_wall_seconds
                        if previous.observation_age_seconds is not None
                        else None
                    ),
                }
            )

        age = None
        if command.episode_id != self.episode_id:
            reason = AdmissionReason.WRONG_EPISODE
        elif command.observation_id not in self.observations:
            reason = AdmissionReason.UNKNOWN_OBSERVATION
        else:
            observation = self.observations[command.observation_id]
            age = elapsed_seconds - observation.captured_elapsed_wall_seconds
            if (
                command.observation_elapsed_wall_seconds
                != observation.captured_elapsed_wall_seconds
                or age < 0
            ):
                reason = AdmissionReason.OBSERVATION_MISMATCH
            elif (
                command.valid_until_elapsed_wall_seconds
                > observation.episode_deadline_elapsed_wall_seconds
                or elapsed_seconds >= command.valid_until_elapsed_wall_seconds
            ):
                reason = AdmissionReason.EXPIRED
            else:
                reason = AdmissionReason.ACCEPTED
        return CommandAdmission(
            identifier=command.identifier,
            reason=reason,
            payload_sha256=digest,
            observed_elapsed_wall_seconds=elapsed_seconds,
            observation_age_seconds=age,
        )
