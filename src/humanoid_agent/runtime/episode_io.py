"""One owned background process for local messaging and loss-intolerant MCAP evidence.

Physics only enqueues immutable CPU snapshots and consumes validated messages.
It never waits for a disk operation. A full queue or failed worker invalidates
execution; evidence is not silently dropped to keep a timing score valid.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.experiment_records import EpisodeIORecord
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.process_lifecycle import process_record
from humanoid_agent.recording import EpisodeRecorder, recording_sha256, validate_recording
from humanoid_agent.runtime.camera import (
    CAMERA_EVENT_ADAPTER,
    CameraCaptureCompleted,
    CameraCaptureRequest,
    NativeVisualBatch,
    NativeVisualState,
)
from humanoid_agent.runtime.cpu import CpuAllocation, parse_cpu_set
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.runtime.operator_transport import (
    OperatorRequest,
    OperatorServer,
    OperatorState,
    RequestObservation,
    SubmitCommand,
)
from humanoid_agent.runtime.records import IdentifierReceipt
from humanoid_agent.runtime.timing import CommandTimingEvidence
from humanoid_agent.sonic_planner_input import SonicMotionPublication


@dataclass(frozen=True)
class WriteRecord:
    path: Path
    payload: Any
    durable: bool = False


@dataclass(frozen=True)
class AppendRecord:
    filename: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BeginEpisode:
    clock: EpisodeClock
    started: dict[str, float]
    capture_initial_observation: bool = True


@dataclass(frozen=True)
class CaptureSnapshot:
    identifier: str
    state: dict[str, Any]
    requested_monotonic_seconds: float
    publish_state: bool = True
    publish_ready: bool = True


@dataclass(frozen=True)
class CaptureBatchSnapshot:
    identifier: str
    state: dict[str, Any]
    requested_monotonic_seconds: float
    observation_replies: tuple[Path | None, ...] = ()


@dataclass(frozen=True)
class StopIO:
    expected_operations: int


@dataclass(frozen=True)
class CommandReceived:
    publication: SonicMotionPublication
    timing: CommandTimingEvidence | None
    timing_status: str


class ObservationRequested(IdentifierReceipt):
    pass


@dataclass(frozen=True)
class ObservationReady:
    identifier: str


IOOperation = (
    WriteRecord | AppendRecord | BeginEpisode | CaptureSnapshot | CaptureBatchSnapshot | StopIO
)
IOEvent = CommandReceived | ObservationRequested | ObservationReady


def _io_worker(output: Path, commands: Path, incoming: Any, outgoing: Any, ready: Any) -> None:
    configure_logging()
    if sys.platform != "linux":
        raise RuntimeError("Episode I/O workers require Linux CPU affinity")
    os.sched_setaffinity(0, parse_cpu_set(CpuAllocation.from_environment().io))
    recorder = EpisodeRecorder(output, live_port=0)
    try:
        operator_server = OperatorServer(commands)
    except BaseException:
        recorder.close()
        raise
    if recorder.server is None:
        recorder.close()
        raise RuntimeError("Episode recording requires a live telemetry server")
    state = OperatorState()
    atomic_json(
        output / "telemetry-endpoint.json",
        {
            "url": f"ws://127.0.0.1:{recorder.server.port}",
            "profile": "webolt-recording-v1",
        },
    )
    operations = 0
    clock: EpisodeClock | None = None
    pending: CaptureSnapshot | None = None
    pending_batch: CaptureBatchSnapshot | None = None
    capture_queue: deque[CaptureSnapshot] = deque()
    batch_queue: deque[CaptureBatchSnapshot] = deque()
    next_ingress_poll = 0.0
    last_requested = "initial"
    expected_operations = None
    camera_sender: LocalPushSender | None = None
    camera_events_path = output / "camera-events"
    camera_events = LocalPullReceiver(camera_events_path)
    completed_captures: dict[str, CameraCaptureCompleted] = {}
    status = "completed"
    error = None
    ready.set()

    def handle_operator_request(request: OperatorRequest) -> OperatorState:
        nonlocal last_requested
        if isinstance(request, SubmitCommand):
            if state.finished:
                raise ValueError("Episode has already finished")
            timing = request.timing
            timing_status = "missing_operator_timing" if timing is None else "provided"
            if timing is not None and any(
                getattr(timing, field) != getattr(request.publication, field)
                for field in (
                    "identifier",
                    "episode_id",
                    "observation_id",
                    "source_response_sha256",
                )
            ):
                timing = None
                timing_status = "operator_timing_identity_mismatch"
            outgoing.put_nowait(CommandReceived(request.publication, timing, timing_status))
            recorder.write("operator/commands.jsonl", request)
            source_directory = output.parent / "operator" / request.publication.identifier
            if source_directory.is_dir():
                # Preserve exact prompt/response bytes behind source-response hashes.
                # The model operator continues to own these immutable artifacts.
                for source in sorted(source_directory.glob("*.json")):
                    recorder.attach(
                        f"operator/{request.publication.identifier}/{source.name}",
                        source,
                        "application/json",
                    )
        elif isinstance(request, RequestObservation):
            if not state.finished and request.identifier != last_requested:
                outgoing.put_nowait(ObservationRequested(identifier=request.identifier))
                last_requested = request.identifier
                recorder.write("operator/observation-requests.jsonl", request)
        return state

    try:
        while True:
            try:
                ingress_wait_seconds = max(0.0, min(0.01, next_ingress_poll - time.monotonic()))
                operation = incoming.get(timeout=ingress_wait_seconds)
            except queue.Empty:
                operation = None
            if isinstance(operation, StopIO):
                expected_operations = operation.expected_operations
                if operations != expected_operations:
                    raise RuntimeError("Not all accepted episode I/O operations were received")
                operation = None
            if (
                expected_operations is not None
                and pending is None
                and pending_batch is None
                and not capture_queue
                and not batch_queue
            ):
                break
            if operation is not None:
                operations += 1
            if isinstance(operation, WriteRecord):
                payload = operation.payload
                recorder.write(operation.path.relative_to(output).as_posix(), payload)
                if operation.path.name == "command-status.json":
                    state = state.model_copy(update={"command_status": payload})
                elif operation.path.name == "motion-finished.json":
                    state = state.model_copy(update={"motion_finished": payload})
                elif operation.path.name in {"result.json", "task-outcome.json"}:
                    if operation.path.name == "result.json":
                        state = state.model_copy(update={"finished": True})
                    # These are live supervisor projections. Final evidence is read
                    # from MCAP and bound to its finalized digest at shutdown.
                    atomic_json(
                        operation.path,
                        payload.model_dump(mode="json")
                        if isinstance(payload, BaseModel)
                        else payload,
                        durable=operation.durable,
                    )
                if operation.durable:
                    recorder.flush(durable=True)
            elif isinstance(operation, AppendRecord):
                if operation.filename not in {"physical-trace.jsonl", "executed-inputs.jsonl"}:
                    raise ValueError("Unexpected evidence stream")
                recorder.write(operation.filename, operation.payload)
            elif isinstance(operation, BeginEpisode):
                clock = operation.clock
                recorder.write("physics-start.json", operation.started)
                recorder.write("episode-clock.json", clock)
                state = state.model_copy(update={"clock": clock})
                pending = (
                    CaptureSnapshot("initial", {}, time.monotonic(), publish_state=False)
                    if operation.capture_initial_observation
                    else None
                )
                # The owner waits for the initial batch's typed completion before
                # starting episode I/O. BeginEpisode transfers that readiness.
            elif isinstance(operation, CaptureBatchSnapshot):
                if len(batch_queue) >= 8:
                    raise RuntimeError("Camera batch queue exceeded its bounded history budget")
                batch_queue.append(operation)
            elif isinstance(operation, CaptureSnapshot):
                if len(capture_queue) >= 8:
                    raise RuntimeError("Camera snapshot queue exceeded its bounded history budget")
                capture_queue.append(operation)
            elif operation is not None:
                raise TypeError("Unknown episode I/O operation")

            if pending_batch is None and batch_queue:
                pending_batch = batch_queue.popleft()
                if camera_sender is None:
                    camera_sender = LocalPushSender(output / "camera-state.json")
                camera_sender.send(
                    CameraCaptureRequest(
                        state=NativeVisualBatch.model_validate(pending_batch.state),
                        reply_to=camera_events_path,
                        observation_replies=list(pending_batch.observation_replies),
                    )
                    .model_dump_json()
                    .encode()
                )
                recorder.write("camera-state.json", pending_batch.state)
            if pending is None and capture_queue:
                pending = capture_queue.popleft()
                if pending.publish_state:
                    if camera_sender is None:
                        camera_sender = LocalPushSender(output / "camera-state.json")
                    camera_sender.send(
                        CameraCaptureRequest(
                            state=NativeVisualState.model_validate(pending.state),
                            reply_to=camera_events_path,
                        )
                        .model_dump_json()
                        .encode()
                    )
                    recorder.write("camera-state.json", pending.state)
            now = time.monotonic()
            if now < next_ingress_poll:
                continue
            next_ingress_poll = now + 0.01
            operator_server.poll(handle_operator_request)

            while (camera_payload := camera_events.receive(timeout_ms=0)) is not None:
                completion = CAMERA_EVENT_ADAPTER.validate_json(camera_payload)
                if not isinstance(completion, CameraCaptureCompleted):
                    raise ValueError("Unexpected camera startup event in episode I/O")
                recorder.write("camera-completions.jsonl", completion)
                if clock is not None and (
                    len(completion.captures) != 1
                    or completion.captures[0].episode_id != clock.episode_id
                    or completion.captures[0].identifier != completion.identifier
                ):
                    raise ValueError("Camera completion does not identify this robot's observation")
                if len(completed_captures) >= 16:
                    raise RuntimeError(
                        "Camera completion queue exceeded its bounded history budget"
                    )
                completed_captures[completion.identifier] = completion

            if pending_batch is not None:
                completion = completed_captures.pop(pending_batch.identifier, None)
                if completion is not None:
                    batch = NativeVisualBatch.model_validate(pending_batch.state)
                    if batch.identifier != pending_batch.identifier:
                        raise ValueError("Queued capture identifier differs from its payload")
                    batch.require_matching_receipt(completion)
                    outgoing.put_nowait(ObservationReady(pending_batch.identifier))
                    pending_batch = None
                elif time.monotonic() - pending_batch.requested_monotonic_seconds > 30:
                    raise TimeoutError("Camera observation batch exceeded 30-second capture budget")
            if clock is None:
                continue
            if pending is not None:
                capture = output / "camera-output" / pending.identifier
                completion = completed_captures.pop(pending.identifier, None)
                if pending.identifier == "initial" or completion is not None:
                    observation = IsaacObservation.parse_requested_snapshot(
                        (capture / "observation.json").read_text(),
                        identifier=pending.identifier,
                        clock=clock,
                    )
                    destination = output / "observations" / pending.identifier
                    destination.mkdir(parents=True, exist_ok=False)
                    for source, target in (
                        ("onboard.png", "rgb.png"),
                        ("depth-m.npy", "depth.npy"),
                        ("optical-points-m.npy", "optical-points-m.npy"),
                        ("observation.json", "observation.json"),
                    ):
                        shutil.copyfile(capture / source, destination / target)
                    if observation.external_camera_available:
                        shutil.copyfile(capture / "observer.png", destination / "external.png")
                    for camera_name in observation.wrist_camera_names:
                        shutil.copyfile(
                            capture / f"{camera_name}.png", destination / f"{camera_name}.png"
                        )
                    recorder.write(
                        f"observations/{pending.identifier}/observation.json", observation
                    )
                    for artifact in sorted(destination.iterdir()):
                        recorder.attach(
                            f"observations/{pending.identifier}/{artifact.name}",
                            artifact,
                            "image/png"
                            if artifact.suffix == ".png"
                            else "application/json"
                            if artifact.suffix == ".json"
                            else "application/octet-stream",
                            source_timestamp_ns=round(
                                (
                                    clock.deadline_monotonic_seconds
                                    - clock.deadline_elapsed_wall_seconds
                                    + observation.elapsed_wall_seconds
                                )
                                * 1_000_000_000
                            ),
                        )
                    release_at = (
                        clock.deadline_monotonic_seconds - clock.deadline_elapsed_wall_seconds
                    )
                    clock = clock.model_copy(
                        update={"elapsed_wall_seconds": max(0, time.monotonic() - release_at)}
                    )
                    recorder.write("episode-clock.json", clock)
                    state = state.model_copy(update={"clock": clock})
                    if pending.publish_ready:
                        receipt = IdentifierReceipt(identifier=pending.identifier)
                        recorder.write("observation-ready.json", receipt)
                        state = state.model_copy(update={"observation_ready": receipt})
                        outgoing.put_nowait(ObservationReady(pending.identifier))
                    pending = None
                elif time.monotonic() - pending.requested_monotonic_seconds > 30:
                    raise TimeoutError("Camera snapshot exceeded 30-second capture budget")
    except BaseException as exception:
        status = "failed"
        error = f"{type(exception).__name__}: {exception}"
        logging.getLogger(__name__).exception("Episode I/O worker failed")
        raise
    finally:
        operator_server.close()
        camera_events.close()
        if camera_sender is not None:
            camera_sender.close()
        summary = {
            "status": status,
            "error": error,
            "schema_version": "webolt-episode-io-v3",
            "operations_processed": operations,
            "operations_submitted": expected_operations,
            "queued_captures_at_shutdown": len(capture_queue) + len(batch_queue),
            "interrupted_capture": pending.identifier
            if pending
            else pending_batch.identifier
            if pending_batch
            else None,
        }
        recorder.write("io-summary.json", summary)
        recorder.close()
        if validate_recording(recorder.path) != recorder.message_count:
            raise RuntimeError("MCAP did not retain every submitted evidence message")
        atomic_json(
            output / "io-summary.json",
            {**summary, "recording_sha256": recording_sha256(recorder.path)},
            durable=True,
        )


class EpisodeIO:
    """A bounded local queue, not a second scheduler or supervision service."""

    def __init__(self, output: Path, commands: Path, *, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("Episode I/O capacity must be positive")
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        self.incoming = context.Queue(maxsize=capacity)
        self.outgoing = context.Queue(maxsize=capacity)
        self.process = context.Process(
            target=_io_worker,
            args=(output, commands, self.incoming, self.outgoing, ready),
            name="episode-io",
        )
        self.output = output
        self.submitted = 0
        self.failed_submission = False
        self.closed = False
        self.process.start()
        try:
            if self.process.pid is None:
                raise RuntimeError("Episode I/O worker did not start")
            atomic_json(output / "io-worker-process.json", process_record(self.process.pid))
            deadline = time.monotonic() + 60
            while not ready.wait(timeout=0.05):
                if self.process.exitcode is not None:
                    raise RuntimeError("Episode I/O worker failed startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Episode I/O worker did not finish startup")
        except BaseException:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=5)
            self.incoming.close()
            self.outgoing.close()
            raise

    def submit(self, operation: IOOperation) -> None:
        if self.closed or self.process.exitcode is not None:
            raise RuntimeError("Episode I/O worker is not running")
        try:
            self.incoming.put_nowait(operation)
        except queue.Full as error:
            self.failed_submission = True
            raise RuntimeError("Episode I/O queue is full; recording cannot keep pace") from error
        self.submitted += 1

    def record(self, path: Path, payload: BaseModel) -> None:
        self.submit(WriteRecord(path, payload, durable=True))

    def events(self) -> list[IOEvent]:
        if self.process.exitcode is not None:
            raise RuntimeError("Episode I/O worker failed; inspect io-summary.json")
        events = []
        while True:
            try:
                events.append(self.outgoing.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.exitcode is None:
                self.incoming.put(StopIO(self.submitted), timeout=5)
            self.process.join(timeout=30)
            if self.process.is_alive():
                raise TimeoutError("Episode I/O worker did not drain within 30 seconds")
            if self.process.exitcode != 0 or self.failed_submission:
                raise RuntimeError("Episode recording failed or could not accept all evidence")
            summary = EpisodeIORecord.model_validate_json(
                (self.output / "io-summary.json").read_text()
            )
            if summary.status != "completed" or summary.operations_processed != self.submitted:
                raise RuntimeError("Episode I/O drain did not account for all accepted operations")
        finally:
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            # A failed worker can leave feeder bytes undelivered; never wait forever.
            self.incoming.cancel_join_thread()
            self.incoming.close()
            self.outgoing.cancel_join_thread()
            self.outgoing.close()
