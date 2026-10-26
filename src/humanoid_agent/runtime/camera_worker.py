"""Render private pose snapshots independently of the task's physics process."""

from __future__ import annotations

import fcntl
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.runtime.camera import (
    CAMERA_REQUEST_ADAPTER,
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    CameraReady,
    CameraStopRequest,
    NativeCameraRenderer,
    NativeVisualBatch,
)
from humanoid_agent.runtime.environment import (
    create_newton_simulation_configuration,
    launch_newton_runtime,
    register_newton_scene_schemas,
)
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender

logger = logging.getLogger(__name__)


def main(
    *,
    cli_isaac_source: Annotated[Path, Parameter(name="--isaac-source")],
    cli_scene: Annotated[Path, Parameter(name="--scene")],
    cli_decorative_assets: Annotated[Path | None, Parameter(name="--decorative-assets")] = None,
    cli_state: Annotated[Path, Parameter(name="--state")],
    cli_output: Annotated[Path, Parameter(name="--output")],
    cli_ready: Annotated[Path, Parameter(name="--ready")],
    num_environments: int = 1,
    batch_captures: bool = False,
    external_camera: bool = False,
    cli_watchdog_seconds: Annotated[float, Parameter(name="--watchdog-seconds")] = 900,
) -> None:
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    register_newton_scene_schemas(rendering=True)
    configuration = create_newton_simulation_configuration()
    runtime = launch_newton_runtime(configuration)
    renderer = None
    runtime_started = False
    receiver = LocalPullReceiver(cli_state)
    reply_senders: dict[Path, LocalPushSender] = {}

    def reply(destination: Path, event: CameraReady | CameraCaptureCompleted) -> None:
        sender = reply_senders.get(destination)
        if sender is None:
            sender = LocalPushSender(destination)
            reply_senders[destination] = sender
        sender.send(event.model_dump_json().encode())

    try:
        # Kit shader initialization can stall when multiple applications start
        # together. Serialize setup only; capture and physics remain concurrent.
        with Path("/tmp/humanoid-agent-isaac-startup.lock").open("a") as startup_lock:
            fcntl.flock(startup_lock.fileno(), fcntl.LOCK_EX)
            runtime.__enter__()
            runtime_started = True
            renderer = NativeCameraRenderer(
                cli_isaac_source,
                cli_scene,
                cli_decorative_assets,
                cli_output,
                simulation_configuration=configuration,
                num_environments=num_environments,
                render_observer=external_camera,
            )
            reply(cli_ready, CameraReady())

        deadline = time.monotonic() + cli_watchdog_seconds
        previous_identifier = None
        while renderer.simulation.is_running():
            if time.monotonic() >= deadline:
                raise TimeoutError("Native camera worker reached its watchdog")
            payload = receiver.receive()
            if payload is None:
                continue
            request = CAMERA_REQUEST_ADAPTER.validate_json(payload)
            if isinstance(request, CameraStopRequest):
                break
            state = request.state
            if state.identifier == previous_identifier:
                continue
            if isinstance(state, NativeVisualBatch):
                if not batch_captures:
                    raise ValueError("Unexpected batch for a single-state camera worker")
                records = renderer.capture_batch(state.states)
            else:
                if num_environments != 1 or batch_captures:
                    raise ValueError("Camera requires a complete environment batch")
                records = [renderer.capture(state)]
            completion = CameraCaptureCompleted(
                identifier=state.identifier,
                captures=[
                    CameraCaptureIdentity(
                        environment_index=record["environment_index"],
                        episode_id=record["episode_id"],
                        identifier=record["identifier"],
                    )
                    for record in records
                ],
            )
            if isinstance(state, NativeVisualBatch):
                state.require_matching_receipt(completion)
            elif completion.captures != [
                CameraCaptureIdentity(
                    environment_index=0, episode_id=state.episode_id, identifier=state.identifier
                )
            ]:
                raise ValueError("Camera result does not identify its requested observation")
            reply(request.reply_to, completion)
            if request.observation_replies:
                for destination, capture in zip(
                    request.observation_replies, completion.captures, strict=True
                ):
                    if destination is not None:
                        reply(
                            destination,
                            CameraCaptureCompleted(
                                identifier=capture.identifier, captures=[capture]
                            ),
                        )
            previous_identifier = state.identifier
        atomic_json(
            cli_output / "result.json",
            {
                "rendered_states": renderer.rendered_states,
                "task_physics_steps": 0,
                "model_requests": 0,
            },
        )
    except Exception as error:
        recording_error = None
        if renderer is not None:
            try:
                # This closes evidence only. Keep Isaac's failure teardown out of
                # the hard-exit path while retaining earlier successful captures.
                renderer.close()
            except Exception as finalization_error:
                recording_error = f"{type(finalization_error).__name__}: {finalization_error}"
        receiver.close()
        for sender in reply_senders.values():
            sender.close()
        atomic_json(
            cli_ready.with_suffix(".error.json"),
            {
                "type": type(error).__name__,
                "message": str(error),
                "recording_finalization_error": recording_error,
            },
        )
        logger.exception("Worker execution failed")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        receiver.close()
        for sender in reply_senders.values():
            sender.close()
        if renderer is not None:
            renderer.close()
        if runtime_started:
            runtime.__exit__(None, None, None)


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
