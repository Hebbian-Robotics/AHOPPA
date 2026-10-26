"""Versioned evidence accepted by the active episode and comparison paths."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    TypeAdapter,
    model_validator,
)

from humanoid_agent.artifact_io import canonical_json_sha256
from humanoid_agent.recording import read_record_bytes, recording_sha256, validate_recording
from humanoid_agent.shutdown_task import CabinetIdentifier, ShutdownCondition
from humanoid_agent.task_outcome import EpisodeEndReason, TaskOutcome
from humanoid_agent.validated_types import ValidatedModel

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[FiniteFloat, Field(ge=0, strict=True)]
Provider = Literal["sol", "astra", "opus", "gemini"]


class CompletionEvidence(BaseModel):
    end_reason: EpisodeEndReason
    physical_criterion_met: StrictBool
    agent_requested_completion: StrictBool
    task_success: StrictBool

    @model_validator(mode="after")
    def require_consistent_completion_evidence(self) -> CompletionEvidence:
        outcome = TaskOutcome(self.end_reason, self.physical_criterion_met)
        if (
            self.agent_requested_completion != outcome.agent_requested_completion
            or self.task_success != outcome.task_success
        ):
            raise ValueError("Recorded success disagrees with completion evidence")
        return self


class ShutdownDrawerOutcome(BaseModel):
    """Private simulator evidence for one designated drawer, scored after the episode."""

    model_config = ConfigDict(extra="forbid")
    cabinet: CabinetIdentifier
    initially_open: StrictBool
    initial_position_m: FiniteFloat
    final_position_m: FiniteFloat
    maximum_position_m: FiniteFloat
    closed_and_settled: StrictBool
    disturbed: StrictBool
    closed_by_hand: StrictBool
    closing_displacement_by_latest_contact_m: dict[
        Literal["hand", "other_body", "none"], Nonnegative
    ]
    hand_contact_seconds: Nonnegative
    other_body_contact_seconds: Nonnegative
    # v2 only: the front entered the onboard camera's unoccluded view within range.
    front_viewed: StrictBool | None = Field(default=None, exclude_if=lambda value: value is None)
    first_viewed_seconds: Nonnegative | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def require_consistent_drawer_evidence(self) -> ShutdownDrawerOutcome:
        if self.disturbed and self.initially_open:
            raise ValueError("Only initially closed drawers can be disturbed")
        if self.closed_by_hand and not (self.initially_open and self.closed_and_settled):
            raise ValueError("Hand closure requires an initially open drawer that ended closed")
        return self


class ShutdownOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition: ShutdownCondition
    requires_manipulation: StrictBool
    drawers: list[ShutdownDrawerOutcome] = Field(min_length=1, max_length=4)
    all_drawers_closed_and_settled: StrictBool
    returned_home_upright: StrictBool
    first_physical_completion_seconds: Nonnegative | None
    completion_declaration_elapsed_seconds: Nonnegative | None

    @model_validator(mode="after")
    def require_consistent_round(self) -> ShutdownOutcome:
        if self.all_drawers_closed_and_settled != all(
            drawer.closed_and_settled for drawer in self.drawers
        ):
            raise ValueError("Round closure disagrees with drawer evidence")
        if self.requires_manipulation != any(drawer.initially_open for drawer in self.drawers):
            raise ValueError("Manipulation requirement disagrees with initial drawer states")
        if self.returned_home_upright and not self.all_drawers_closed_and_settled:
            raise ValueError("Return is scored only after every drawer is closed")
        return self


class TaskOutcomeRecord(CompletionEvidence, ValidatedModel):
    # Historical JSON permits enum strings; copies and trusted instances still
    # revalidate, so an update cannot bypass a task variant's invariants.
    model_config = ConfigDict(strict=False)
    completion_protocol: Literal["agent-declaration-then-independent-verification-v1"]
    balance_contact_protocol: Literal["ground-reaching-filtered-floor-contact-v2"]


# Outcome records keep the null fields that earlier tasks introduced, so new
# records stay identical to the recorded ones and those remain readable.
class NoTransferEvidence(BaseModel):
    transfer_condition: None = None
    object_lifted: None = None
    maximum_carried_distance_m: None = None


class NoMobilityEvidence(BaseModel):
    mobility_condition: None = None
    mobility_stage_achieved: None = None


class NoInspectionEvidence(BaseModel):
    inspection_report_correct: None = Field(default=None, exclude_if=lambda value: value is None)
    inspection_condition: None = None
    inspection_visited_stations: None = None
    inspection_report: None = None
    inspection_expected_report: None = None


class ShutdownTaskOutcome(
    TaskOutcomeRecord, NoTransferEvidence, NoMobilityEvidence, NoInspectionEvidence
):
    """The drawer-shutdown round's independent physical outcome."""

    task_protocol: Literal["drawer-shutdown-v1", "drawer-shutdown-v2"] = "drawer-shutdown-v1"
    shutdown: ShutdownOutcome

    @model_validator(mode="after")
    def require_shutdown_completion(self) -> ShutdownTaskOutcome:
        if self.physical_criterion_met and not self.shutdown.returned_home_upright:
            raise ValueError("Physical shutdown completion requires closure and return home")
        return self


TASK_OUTCOME_ADAPTER = TypeAdapter(ShutdownTaskOutcome)


class MeasuredValidity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["valid", "invalid_physics_lag"]
    maximum_physics_lag_seconds: Nonnegative
    limit_seconds: Annotated[FiniteFloat, Field(ge=0.1, le=0.1, strict=True)]
    policy: Literal["wall-clock-lag-v1"]

    @model_validator(mode="after")
    def require_matching_threshold_result(self) -> MeasuredValidity:
        if (self.maximum_physics_lag_seconds <= self.limit_seconds) != (self.status == "valid"):
            raise ValueError("Simulation validity disagrees with measured physics lag")
        return self


class UnknownValidity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_established"]


Validity = Annotated[MeasuredValidity | UnknownValidity, Field(discriminator="status")]


class EpisodeProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_manifest_sha256: Digest
    scene_sha256: Digest
    appearance_manifest_sha256: Digest | None
    task_instruction_sha256: Digest
    facility_map_sha256: Digest | None = Field(default=None, exclude_if=lambda value: value is None)
    wall_seconds: Annotated[FiniteFloat, Field(gt=0, le=3600)]
    execution_topology: Literal["single-robot", "shared-physics"]
    gpu_scheduling: Literal["driver-default", "cuda-mps"]
    sensor_protocol: Literal["egocentric-rgbd-wrist-rgb-kinematics-depth-tools-v4"]
    observation_interface_protocol: (
        Literal[
            "paired-action-history-robot-feedback-v1",
            "paired-action-history-initial-reference-v2",
        ]
        | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    recording_protocol: Literal["webolt-recording-v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recording_runtime_sha256: Digest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    command_protocol: Literal[
        "sonic-native-planner-v1",
        "sonic-native-planner-v2",
        "sonic-native-planner-v3",
    ]
    completion_protocol: Literal["agent-declaration-then-independent-verification-v1"]
    balance_contact_protocol: Literal["ground-reaching-filtered-floor-contact-v2"]
    clock_policy: Literal["wall-clock-lag-v1"]
    sonic_constants_sha256: Digest
    robot_urdf_sha256: Digest
    # Retired combined-task field; current and recorded storage protocols hold null.
    exploratory_composite_rationale: None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    model_max_output_tokens: Annotated[int, Field(gt=0, le=65536)] | None = None
    api_service_tier: Literal["standard", "flex"]
    api_request_timeout_seconds: Annotated[FiniteFloat, Field(gt=0)]
    api_retry_policy: Literal["none", "bounded-api-retry-v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    project_lock_sha256: Digest
    physics_lock_sha256: Digest


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["webolt-episode-v3"]
    provider: Provider | None
    episode_kind: Literal["model", "controller-check"] | None
    protocol: EpisodeProtocol | None
    status: Literal["completed", "cancelled", "infrastructure_error"]
    error: str | None
    cleanup_errors: list[str]
    component_exit_codes: dict[
        Literal["physics", "operator", "gpu_scheduling"], Annotated[int, Field(strict=True)] | None
    ]
    task_outcome: ShutdownTaskOutcome | None
    simulation_validity: Validity
    analysis: Literal["not_run"]
    rendering: Literal["not_run"]
    publication: Literal["not_run"]

    @model_validator(mode="after")
    def require_completed_execution_evidence(self) -> EpisodeRecord:
        if self.status == "completed" and (
            self.episode_kind is None
            or (self.episode_kind == "model" and self.provider is None)
            or self.protocol is None
            or self.task_outcome is None
            or self.error is not None
            or self.cleanup_errors
            or self.component_exit_codes.get("physics") != 0
            or (
                self.protocol is not None
                and self.protocol.gpu_scheduling == "cuda-mps"
                and self.component_exit_codes.get("gpu_scheduling") != 0
            )
            or isinstance(self.simulation_validity, UnknownValidity)
            or (
                self.episode_kind == "model"
                and self.component_exit_codes.get("operator") not in (0, -2, 130)
            )
        ):
            raise ValueError("Completed episode lacks consistent terminal execution evidence")
        if self.cleanup_errors and self.status != "infrastructure_error":
            raise ValueError("Cleanup failures must remain infrastructure errors")
        return self


class PhysicsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finish_reason: EpisodeEndReason
    status: Literal["completed"]
    elapsed_wall_seconds: Nonnegative
    task_success: StrictBool
    physics_steps: Annotated[int, Field(ge=0, strict=True)]
    trace_rows: Annotated[int, Field(ge=0, strict=True)]
    distinct_native_commands_applied: Annotated[int, Field(ge=0, strict=True)]
    control_started: StrictBool
    physics_started_after_controller_ready: StrictBool
    maximum_physics_clock_lag_seconds: Nonnegative
    simulation_validity: MeasuredValidity
    maximum_command_age_seconds: Nonnegative
    physics_process_model_requests: Literal[0]
    agent_requested_completion: StrictBool

    @model_validator(mode="after")
    def require_matching_clock_evidence(self) -> PhysicsResult:
        if (
            self.maximum_physics_clock_lag_seconds
            != self.simulation_validity.maximum_physics_lag_seconds
        ):
            raise ValueError("Physics result contains contradictory lag evidence")
        return self


class EpisodeIORecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["webolt-episode-io-v1", "webolt-episode-io-v2", "webolt-episode-io-v3"]
    status: Literal["completed", "failed"]
    error: str | None
    operations_processed: Annotated[int, Field(ge=0, strict=True)]
    operations_submitted: Annotated[int, Field(ge=0, strict=True)] | None
    interrupted_capture: str | None
    queued_captures_at_shutdown: Annotated[int, Field(ge=0, strict=True)] = 0
    recording_sha256: Digest | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def require_complete_recording(self) -> EpisodeIORecord:
        if self.status == "completed" and (
            self.error is not None
            or self.operations_submitted != self.operations_processed
            or (
                self.schema_version in {"webolt-episode-io-v2", "webolt-episode-io-v3"}
                and (self.interrupted_capture is not None or self.queued_captures_at_shutdown != 0)
            )
        ):
            raise ValueError("Recording completion does not account for accepted operations")
        if self.schema_version == "webolt-episode-io-v3" and self.recording_sha256 is None:
            raise ValueError("MCAP recording lacks its finalized digest")
        return self


class CohortEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: Path
    condition: Annotated[str, Field(min_length=1)]
    provider: Provider


class CohortManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["webolt-cohort-v3"]
    protocol: EpisodeProtocol
    episodes: Annotated[list[CohortEpisode], Field(min_length=1)]

    @model_validator(mode="after")
    def require_consistent_condition_assignments(self) -> CohortManifest:
        providers: dict[str, Provider] = {}
        for episode in self.episodes:
            if episode.condition in providers and providers[episode.condition] != episode.provider:
                raise ValueError("A comparison condition cannot pool different providers")
            providers[episode.condition] = episode.provider
        return self


def load_episode_record(directory: Path) -> EpisodeRecord:
    return EpisodeRecord.model_validate_json((directory / "episode.json").read_text())


def validate_physical_record(directory: Path, episode: EpisodeRecord) -> PhysicsResult:
    physics = directory / "physics"
    recording = EpisodeIORecord.model_validate_json((physics / "io-summary.json").read_text())
    if recording.status != "completed":
        raise ValueError("Episode recording did not complete")
    if recording.schema_version == "webolt-episode-io-v3":
        archive = physics / "records.mcap"
        if recording_sha256(archive) != recording.recording_sha256:
            raise ValueError("MCAP recording changed after completion")
        validate_recording(archive)
    elif episode.protocol is not None and episode.protocol.recording_protocol is not None:
        raise ValueError("MCAP episode lacks its recording drain evidence")
    result = PhysicsResult.model_validate_json(read_record_bytes(physics / "result.json"))
    outcome = TASK_OUTCOME_ADAPTER.validate_json(read_record_bytes(physics / "task-outcome.json"))
    if (
        episode.task_outcome != outcome
        or episode.simulation_validity != result.simulation_validity
        or result.finish_reason != outcome.end_reason
        or result.task_success != outcome.task_success
        or result.agent_requested_completion != outcome.agent_requested_completion
    ):
        raise ValueError("Episode, physics and task-outcome records disagree")
    return result


def validate_frozen_source(episode: Path, expected_manifest_sha256: Digest) -> None:
    """Verify the retained manifest and every source file within its source root."""
    source_manifest = TypeAdapter(dict[str, Digest]).validate_json(
        (episode / "source-manifest.json").read_text()
    )
    if canonical_json_sha256(source_manifest) != expected_manifest_sha256:
        raise ValueError("Episode source manifest differs from its protocol")
    source_root = (episode / "source").resolve(strict=True)
    for relative_path, expected_digest in source_manifest.items():
        source_path = (source_root / relative_path).resolve(strict=True)
        if not source_path.is_relative_to(source_root):
            raise ValueError("Source manifest path escapes the frozen source directory")
        with source_path.open("rb") as source_stream:
            actual_digest = hashlib.file_digest(source_stream, "sha256").hexdigest()
        if actual_digest != expected_digest:
            raise ValueError("Frozen episode source changed after execution")


class FrozenAppearanceManifest(BaseModel):
    """Only file identities are consumed; appearance metadata may contain extra fields."""

    files: dict[str, Digest]


def validate_frozen_inputs(directory: Path, protocol: EpisodeProtocol) -> None:
    """Use the same evidence integrity checks for cohorts and component readiness."""
    if protocol.recording_protocol is not None:
        runtime_requirements = directory / "inputs/recording-runtime-requirements.txt"
        if (
            protocol.recording_runtime_sha256
            != hashlib.sha256(runtime_requirements.read_bytes()).hexdigest()
        ):
            raise ValueError("Frozen recording runtime changed after execution")
    for relative_path, expected_digest in (
        ("inputs/scene.json", protocol.scene_sha256),
        ("inputs/sonic_constants.json", protocol.sonic_constants_sha256),
        ("inputs/robot.urdf", protocol.robot_urdf_sha256),
        ("inputs/uv.lock", protocol.project_lock_sha256),
        ("inputs/physics-uv.lock", protocol.physics_lock_sha256),
    ):
        if hashlib.sha256((directory / relative_path).read_bytes()).hexdigest() != expected_digest:
            raise ValueError("Frozen episode input changed after execution")
    if protocol.appearance_manifest_sha256 is None:
        return
    appearance = (directory / "inputs/appearance").resolve(strict=True)
    manifest_bytes = (appearance / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != protocol.appearance_manifest_sha256:
        raise ValueError("Frozen appearance manifest changed after execution")
    manifest = FrozenAppearanceManifest.model_validate_json(manifest_bytes)
    for filename, expected_digest in manifest.files.items():
        path = (appearance / filename).resolve(strict=True)
        if (
            not path.is_relative_to(appearance)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest
        ):
            raise ValueError("Frozen appearance asset changed after execution")
