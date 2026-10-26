"""One robot's command admission, task state and evidence inside the production owner."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_agent.canonical_robot import canonical_urdf_path
from humanoid_agent.command_admission import AdmissionReason, EpisodeClock, ObservationIdentity
from humanoid_agent.dex3 import hand_joint_limits, validate_hand_joint_targets
from humanoid_agent.experiment_records import (
    CompletionEvidence,
    PhysicsResult,
    ShutdownDrawerOutcome,
    ShutdownOutcome,
    ShutdownTaskOutcome,
)
from humanoid_agent.hand_control import HandTransitionInitials
from humanoid_agent.observation_history import ActionExecutionWindow, ActionFrameContext
from humanoid_agent.runtime.camera import NativeVisualState
from humanoid_agent.runtime.episode_io import (
    AppendRecord,
    BeginEpisode,
    CaptureSnapshot,
    CommandReceived,
    EpisodeIO,
    ObservationReady,
    ObservationRequested,
    WriteRecord,
)
from humanoid_agent.runtime.evaluation import GroundTaskEvaluation, GroundTaskMeasurements
from humanoid_agent.runtime.records import CommandStatus
from humanoid_agent.runtime.sdk import PlannerPublicationReceipt, SonicMotorCommand
from humanoid_agent.runtime.state import clock_validity
from humanoid_agent.runtime.timing import CommandTimingEvidence
from humanoid_agent.shutdown_task import ShutdownTask
from humanoid_agent.sonic_planner_input import (
    ControllerObservation,
    NativePlannerSample,
    SonicPlannerCommandStream,
)
from humanoid_agent.task_balance import ISAAC_CONTACT_BALANCE_PROTOCOL
from humanoid_agent.task_outcome import COMPLETION_PROTOCOL, EpisodeEndReason, TaskOutcome


def shutdown_outcome(
    evaluation: GroundTaskEvaluation, *, completion_declaration_elapsed_seconds: float | None
) -> ShutdownOutcome:
    task = evaluation.shutdown
    drawers = []
    for cabinet in task.cabinets:
        progress = evaluation.drawers[cabinet.identifier]
        initial = (
            progress.initial_position_m
            if progress.initial_position_m is not None
            else task.initial_drawer_position_m(cabinet)
        )
        closed = progress.closed_hold_ticks * 0.02 >= task.hold_seconds
        closed_by_hand = (
            cabinet.initially_open
            and closed
            and progress.closing_displacement_m["hand"]
            >= initial - progress.final_position_m - task.closed_tolerance_m
        )
        drawers.append(
            ShutdownDrawerOutcome(
                cabinet=cabinet.identifier,
                initially_open=cabinet.initially_open,
                initial_position_m=initial,
                final_position_m=progress.final_position_m,
                maximum_position_m=progress.maximum_position_m,
                closed_and_settled=closed,
                disturbed=evaluation.drawer_disturbed(cabinet.identifier),
                closed_by_hand=closed_by_hand,
                closing_displacement_by_latest_contact_m=dict(progress.closing_displacement_m),
                hand_contact_seconds=progress.hand_contact_ticks * 0.02,
                other_body_contact_seconds=progress.other_body_contact_ticks * 0.02,
                front_viewed=progress.viewed_ticks > 0
                if task.protocol == "drawer-shutdown-v2"
                else None,
                first_viewed_seconds=progress.first_viewed_tick * 0.02
                if task.protocol == "drawer-shutdown-v2" and progress.first_viewed_tick is not None
                else None,
            )
        )
    return ShutdownOutcome(
        condition=task.condition,
        requires_manipulation=task.requires_manipulation,
        drawers=drawers,
        all_drawers_closed_and_settled=all(drawer.closed_and_settled for drawer in drawers),
        returned_home_upright=evaluation.physical_criterion_met,
        first_physical_completion_seconds=evaluation.first_physical_completion_seconds,
        completion_declaration_elapsed_seconds=completion_declaration_elapsed_seconds,
    )


class RobotEpisode:
    """Command admission and scoring are private to this episode, not to the group."""

    def __init__(
        self,
        directory: Path,
        initial: NativeVisualState,
        neutral: NativePlannerSample,
        environment_origin_m: list[float],
        shutdown: ShutdownTask,
        *,
        observation_history_enabled: bool = False,
    ) -> None:
        self.directory = directory
        self.output = directory / "physics"
        self.initial = initial
        self.neutral = neutral
        self.observation_history_enabled = observation_history_enabled
        self.execution_window: ActionExecutionWindow | None = None
        self.captured_history_roles: set[str] = set()
        self.hand_limits = (
            hand_joint_limits(canonical_urdf_path()) if len(initial.joint_names) == 43 else None
        )
        self.origin = np.asarray(environment_origin_m)
        self.io = EpisodeIO(self.output, directory / "commands")
        self.evaluation = GroundTaskEvaluation(shutdown)
        self.requested_capture: str | None = None
        self.pending_capture: str | None = None
        self.last_snapshot = "initial"
        self.finished_command: str | None = None
        self.finish_reason: EpisodeEndReason | None = None
        self.final_elapsed = 0.0
        self.final_tick = 0
        self.maximum_lag = 0.0
        self.maximum_command_age = 0.0
        self.trace_rows = 0
        self.native_commands = 0
        self.timed_commands: set[str] = set()
        self.last_motor_command: SonicMotorCommand | None = None
        self.command_timings: dict[str, tuple[float, CommandTimingEvidence | None, str]] = {}
        self.active_admitted_at: float | None = None
        self.release_at: float | None = None
        try:
            self.stream = SonicPlannerCommandStream(
                self.output / "command-ledger",
                initial.episode_id,
                record=self.io.record,
                restore_records=False,
            )
            self.register_observation(initial)
        except BaseException:
            self.io.close()
            raise

    def register_observation(self, snapshot: NativeVisualState) -> None:
        self.stream.register_observation(
            ControllerObservation(
                identity=ObservationIdentity(
                    episode_id=snapshot.episode_id,
                    identifier=snapshot.identifier,
                    captured_elapsed_wall_seconds=snapshot.elapsed_wall_seconds,
                    episode_deadline_elapsed_wall_seconds=snapshot.episode_deadline_elapsed_wall_seconds,
                ),
                pelvis_quaternion_wxyz=snapshot.root_state_world[3:7],
            )
        )

    def start(self, release_at: float, *, capture_initial_observation: bool = True) -> None:
        self.release_at = release_at
        self.io.submit(
            BeginEpisode(
                EpisodeClock(
                    deadline_monotonic_seconds=release_at
                    + self.initial.episode_deadline_elapsed_wall_seconds,
                    episode_id=self.initial.episode_id,
                    elapsed_wall_seconds=0,
                    deadline_elapsed_wall_seconds=self.initial.episode_deadline_elapsed_wall_seconds,
                ),
                {"wall_seconds": 0, "simulation_seconds": 0},
                capture_initial_observation=capture_initial_observation,
            )
        )

    def poll(
        self,
        elapsed: float,
        measure_current_state: Callable[[], GroundTaskMeasurements],
        *,
        accept_model_commands: bool = True,
        hand_initials: HandTransitionInitials | None = None,
    ) -> None:
        if self.finish_reason is not None:
            return
        for event in self.io.events():
            if isinstance(event, CommandReceived):
                if not accept_model_commands:
                    raise ValueError("Scripted controller checks cannot admit model commands")
                publication = event.publication
                validate_hand_joint_targets(
                    publication.left_hand_joints, publication.right_hand_joints, self.hand_limits
                )
                for schedule, side in (
                    (publication.left_grasp_schedule, "left"),
                    (publication.right_grasp_schedule, "right"),
                ):
                    if schedule is not None:
                        validate_hand_joint_targets(
                            schedule.target_positions if side == "left" else None,
                            schedule.target_positions if side == "right" else None,
                            self.hand_limits,
                        )
                admission = self.stream.activate(publication, elapsed, hand_initials=hand_initials)
                self.io.submit(
                    WriteRecord(
                        self.output / "command-status.json",
                        CommandStatus(
                            identifier=publication.identifier,
                            state="admitted"
                            if admission.reason == AdmissionReason.ACCEPTED
                            else "duplicate"
                            if admission.reason == AdmissionReason.DUPLICATE
                            else "rejected",
                            admission=admission,
                        ),
                    )
                )
                if admission.reason == AdmissionReason.ACCEPTED:
                    self.active_admitted_at = time.monotonic()
                    self.command_timings[publication.identifier] = (
                        self.active_admitted_at,
                        event.timing,
                        event.timing_status,
                    )
                    self.io.submit(
                        WriteRecord(
                            self.output / (publication.identifier + ".executed-command.json"),
                            publication,
                        )
                    )
            elif isinstance(event, ObservationReady):
                self.last_snapshot = event.identifier
                if self.pending_capture == event.identifier:
                    self.pending_capture = None
            elif isinstance(event, ObservationRequested):
                if not accept_model_commands:
                    raise ValueError("Scripted controller checks own their observation schedule")
                if event.identifier in (
                    self.last_snapshot,
                    self.pending_capture,
                    self.requested_capture,
                ):
                    continue
                if self.pending_capture is not None or self.requested_capture is not None:
                    raise RuntimeError("A request replaced this robot's unfinished capture")
                self.requested_capture = event.identifier
            else:
                raise TypeError("Unexpected model episode I/O event")
        if self.stream.active_timeline is not None and self.stream.endpoint_reached(elapsed):
            active = self.stream.active_timeline.publication
            if self.finished_command != active.identifier:
                self.finished_command = active.identifier
                self.io.submit(
                    WriteRecord(
                        self.output / "motion-finished.json", {"identifier": active.identifier}
                    )
                )
                if active.request_task_completion:
                    self.finish_reason = self.evaluation.terminal_reason(
                        measure_current_state(),
                        completion_requested=True,
                        deadline_reached=elapsed
                        >= self.initial.episode_deadline_elapsed_wall_seconds,
                    )
        if elapsed >= self.initial.episode_deadline_elapsed_wall_seconds:
            self.finish_reason = self.evaluation.terminal_reason(
                measure_current_state(), deadline_reached=True
            )
        elif self.evaluation.fallen:
            self.finish_reason = EpisodeEndReason.ROBOT_FELL

    def current_capture_ready(self) -> bool:
        if not self.observation_history_enabled or self.stream.active_timeline is None:
            return True
        return (
            self.execution_window is not None
            and self.execution_window.action_id
            == self.stream.active_timeline.publication.identifier
            and self.captured_history_roles == {"start", "midpoint"}
        )

    def due_history_frame(self, elapsed_seconds: float) -> ActionFrameContext | None:
        if (
            not self.observation_history_enabled
            or self.execution_window is None
            or self.finish_reason is not None
        ):
            return None
        for role in ("start", "midpoint"):
            if (
                role not in self.captured_history_roles
                and elapsed_seconds >= self.execution_window.capture_time(role)
            ):
                return ActionFrameContext(
                    execution=self.execution_window,
                    role=role,
                    requested_elapsed_seconds=self.execution_window.capture_time(role),
                )
        return None

    def begin_history_capture(
        self, state: NativeVisualState, context: ActionFrameContext, requested_at: float
    ) -> NativeVisualState:
        if context.role not in {"start", "midpoint"}:
            raise ValueError("History capture must be an action start or midpoint")
        updated = state.model_copy(
            update={
                "identifier": f"{context.execution.action_id}_history_{context.role}",
                "action_frame": context,
            }
        )
        self.io.submit(
            CaptureSnapshot(
                updated.identifier,
                updated.model_dump(mode="json"),
                requested_at,
                publish_state=False,
                publish_ready=False,
            )
        )
        self.captured_history_roles.add(context.role)
        return updated

    def begin_capture(self, state: NativeVisualState, requested_at: float) -> NativeVisualState:
        if not self.current_capture_ready():
            raise ValueError(
                "Post-action capture requires this action's native receipt and history frames"
            )
        if state.identifier != self.requested_capture:
            raise ValueError("Capture does not match this robot's queued observation request")
        if self.execution_window is not None:
            state = state.model_copy(
                update={
                    "action_frame": ActionFrameContext(
                        execution=self.execution_window,
                        role="current/end",
                        requested_elapsed_seconds=self.execution_window.endpoint_elapsed_seconds,
                    )
                }
            )
        self.register_observation(state)
        self.io.submit(
            CaptureSnapshot(
                state.identifier, state.model_dump(mode="json"), requested_at, publish_state=False
            )
        )
        self.pending_capture = state.identifier
        self.requested_capture = None
        return state

    def planner_sample(self, elapsed: float, root_wxyz: list[float]) -> NativePlannerSample:
        if self.finish_reason is not None:
            # Retain the endpoint tracking reference, but do not continue locomotion.
            sample = self.stream.sample(elapsed, root_wxyz)
            if sample is not None:
                return sample.model_copy(update={"locomotion_mode": 0, "speed_m_per_s": 0})
            return self.neutral
        sample = self.stream.sample(elapsed, root_wxyz)
        return sample if sample is not None else self.neutral

    def record_planner_publication(self, publication: PlannerPublicationReceipt) -> None:
        if self.release_at is None:
            return
        elapsed = publication.published_monotonic_seconds - self.release_at
        if elapsed < 0 or (self.finish_reason is not None and elapsed > self.final_elapsed):
            return
        identifier = publication.command_id
        timing: dict[str, Any] | None = None
        if identifier is not None and identifier not in self.timed_commands:
            if identifier not in self.command_timings:
                raise ValueError("Native planner publication has no admitted source command")
            admitted_at, evidence, timing_status = self.command_timings[identifier]
            command = self.stream.ledger.admissions[identifier]
            observation_at = self.release_at + (
                command.observed_elapsed_wall_seconds - command.observation_age_seconds
                if command.observation_age_seconds is not None
                else 0
            )
            now = publication.published_monotonic_seconds
            timing = {
                "status": timing_status,
                "observation_age_at_first_planner_send_seconds": now - observation_at,
            }
            if evidence is not None:
                timing = evidence.execution_breakdown(
                    observation_monotonic_seconds=observation_at,
                    execution_monotonic_seconds=now,
                )
            timing["first_planner_send_monotonic_seconds"] = now
            timing["admission_to_first_planner_send_seconds"] = now - admitted_at
            self.timed_commands.add(identifier)
            if (
                self.observation_history_enabled
                and self.stream.active_timeline is not None
                and self.stream.active_started_elapsed_seconds is not None
            ):
                self.execution_window = ActionExecutionWindow(
                    action_id=identifier,
                    started_elapsed_seconds=elapsed,
                    endpoint_elapsed_seconds=max(
                        elapsed,
                        self.stream.active_started_elapsed_seconds
                        + self.stream.active_timeline.execution_duration_seconds,
                    ),
                )
                self.captured_history_roles.clear()
                timing["execution_window"] = self.execution_window.model_dump(mode="json")
        self.io.submit(
            AppendRecord(
                "executed-inputs.jsonl",
                {
                    "elapsed_wall_seconds": elapsed,
                    "command_id": identifier,
                    "sample": publication.sample.model_dump(mode="json"),
                    "transport_stage": "native_zmq_publication",
                    "sequence_number": publication.sequence_number,
                    "published_monotonic_seconds": publication.published_monotonic_seconds,
                    **({"observation_timing": timing} if timing is not None else {}),
                },
            )
        )

    def observe_motor_command(self, command: SonicMotorCommand, age: float) -> None:
        if self.finish_reason is not None:
            return
        self.maximum_command_age = max(self.maximum_command_age, age)
        if command != self.last_motor_command:
            self.native_commands += 1
            self.last_motor_command = command

    def record_trace(
        self,
        state: NativeVisualState,
        evidence: dict[str, Any],
        native_command: SonicMotorCommand | None,
        *,
        phase: str = "native-task",
    ) -> None:
        if self.finish_reason is not None:
            return
        root = np.asarray(state.root_state_world).copy()
        root[:3] -= self.origin
        self.io.submit(
            AppendRecord(
                "physical-trace.jsonl",
                {
                    "simulation_seconds": state.simulation_seconds,
                    "wall_seconds": state.elapsed_wall_seconds,
                    "free_phase_seconds": state.elapsed_wall_seconds,
                    "phase": phase,
                    "root_state": root.tolist(),
                    "root_position": root[:3].tolist(),
                    "root_quaternion_wxyz": root[3:7].tolist(),
                    "joint_names": state.joint_names,
                    "command_id": self.stream.active_timeline.publication.identifier
                    if self.stream.active_timeline is not None
                    else None,
                    "contact_evidence": evidence,
                    "activated": self.evaluation.activated,
                    "physical_criterion_met": self.evaluation.physical_criterion_met,
                    "cabinet_joint_positions": state.cabinet_joint_positions,
                    "drawer_latest_contact": {
                        identifier: progress.latest_contact
                        for identifier, progress in self.evaluation.drawers.items()
                    },
                    "drawer_closed_hold_ticks": {
                        identifier: progress.closed_hold_ticks
                        for identifier, progress in self.evaluation.drawers.items()
                    },
                    # Latest 20 ms evaluator sample: slide speed and peak normal forces.
                    "drawer_evaluation_sample": {
                        identifier: {
                            "velocity_m_per_s": progress.latest_velocity_m_per_s,
                            "hand_normal_force_n": progress.latest_hand_normal_force_n,
                            "other_body_normal_force_n": progress.latest_other_body_normal_force_n,
                        }
                        for identifier, progress in self.evaluation.drawers.items()
                    },
                    "joint_positions": state.joint_positions,
                    "joint_velocities": state.joint_velocities,
                    "native_command": native_command.model_dump(mode="json")
                    if native_command is not None
                    else None,
                },
            )
        )
        self.trace_rows += 1

    def finish(
        self, elapsed: float, tick: int, *, timing_summary: dict[str, Any] | None = None
    ) -> None:
        if self.finish_reason is None:
            raise RuntimeError("Cannot finish an episode without a terminal reason")
        self.requested_capture = None
        self.final_elapsed = elapsed
        self.final_tick = tick
        if timing_summary is not None:
            self.io.submit(WriteRecord(self.output / "timing.json", timing_summary))
        outcome = TaskOutcome(self.finish_reason, self.evaluation.physical_criterion_met)
        common_outcome = CompletionEvidence(
            end_reason=self.finish_reason,
            physical_criterion_met=outcome.physical_criterion_met,
            agent_requested_completion=outcome.agent_requested_completion,
            task_success=outcome.task_success,
        ).model_dump()
        common_outcome.update(
            completion_protocol=COMPLETION_PROTOCOL,
            balance_contact_protocol=ISAAC_CONTACT_BALANCE_PROTOCOL,
        )
        record = ShutdownTaskOutcome(
            **common_outcome,
            task_protocol=self.evaluation.shutdown.protocol,
            shutdown=shutdown_outcome(
                self.evaluation,
                completion_declaration_elapsed_seconds=elapsed
                if self.finish_reason == EpisodeEndReason.AGENT_DECLARED_COMPLETION
                else None,
            ),
        )
        self.io.submit(WriteRecord(self.output / "task-outcome.json", record, durable=True))
        result = PhysicsResult(
            finish_reason=self.finish_reason,
            status="completed",
            elapsed_wall_seconds=elapsed,
            task_success=outcome.task_success,
            physics_steps=tick,
            trace_rows=self.trace_rows,
            distinct_native_commands_applied=self.native_commands,
            control_started=True,
            physics_started_after_controller_ready=True,
            maximum_physics_clock_lag_seconds=self.maximum_lag,
            simulation_validity=clock_validity(self.maximum_lag),
            maximum_command_age_seconds=self.maximum_command_age,
            physics_process_model_requests=0,
            agent_requested_completion=outcome.agent_requested_completion,
        )
        self.io.submit(WriteRecord(self.output / "result.json", result, durable=True))
