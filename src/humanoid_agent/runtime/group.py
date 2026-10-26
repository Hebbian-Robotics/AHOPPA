"""Own a shared simulator and independent model operators on one GPU host."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event
from types import FrameType
from typing import TYPE_CHECKING, Any

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.process_lifecycle import process_record
from humanoid_agent.runtime.cpu import CpuAllocation
from humanoid_agent.runtime.diagnostic import scripted_episode_options
from humanoid_agent.runtime.episode import (
    create_host_process_environment,
    interrupt_pending_attempts,
    model_operator_command,
    preflight_episode,
    prepare_episode_inputs,
    reconcile_execution,
    stop_process,
    write_terminal_episode,
)
from humanoid_agent.runtime.gpu import CudaMultiprocessService
from humanoid_agent.runtime.layout import EpisodeLayout
from humanoid_agent.runtime.operator_transport import OperatorClient, QueryOperatorState
from humanoid_agent.task_instructions import scene_task_instruction

if TYPE_CHECKING:
    from humanoid_agent.cli import EpisodeOptions

logger = logging.getLogger(__name__)


def run_episode_group(arguments: EpisodeOptions, *, single_episode: bool = False) -> None:
    """Freeze once, supervise all components, and always record every assigned episode."""
    if arguments.episode_kind == "controller-check":
        arguments = scripted_episode_options(arguments)
    cpu_allocation = CpuAllocation.from_options(arguments)
    providers = (arguments.provider,) if single_episode else arguments.providers
    output: Path = arguments.output.resolve()
    layout = EpisodeLayout(output, len(providers), single_episode)
    gpu_scheduling = "driver-default" if layout.num_environments == 1 else "cuda-mps"
    execution_topology = "single-robot" if layout.num_environments == 1 else "shared-physics"
    output.mkdir(parents=True, exist_ok=False)
    logger.info("Starting shared episode group in %s", output)
    directories = layout.episode_directories
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs = []
    interrupted: set[str] = set()
    previous_handlers: dict[Any, Any] = {}
    status = "infrastructure_error"
    error = None
    prepared = None
    episode_clock: EpisodeClock | None = None
    cleanup_errors: list[str] = []
    stopping = Event()
    gpu_service = CudaMultiprocessService(output / "gpu-scheduling")

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stopping.set()

    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "commands").mkdir()
        atomic_json(
            output / "group-configuration.json",
            {
                "schema_version": "webolt-episode-group-v1",
                "execution_topology": execution_topology,
                "gpu_scheduling": gpu_scheduling,
                "cpu_allocation": asdict(cpu_allocation),
                "episode_kind": arguments.episode_kind,
                "providers": providers,
                "episode_directories": [str(path.relative_to(output)) for path in directories],
                "wall_seconds": arguments.wall_seconds,
                "clock": "One absolute host-monotonic release/deadline after all controllers and cameras are warm",
                "failure_policy": "Component infrastructure failure stops the group; retain each physical outcome separately",
                "presentation_required_for_valid_record": False,
            },
            durable=True,
        )
        if arguments.task_scene is None:
            raise ValueError("An episode requires --task-scene")
        task_scene = GroundWorkcellScene.model_validate_json(arguments.task_scene.read_text())
        arguments = replace(
            arguments,
            task_instruction=scene_task_instruction(
                arguments.task_instruction, task_scene.shutdown
            ),
        )
        prepared = prepare_episode_inputs(
            arguments, output, execution_topology=execution_topology, gpu_scheduling=gpu_scheduling
        )
        environment = create_host_process_environment(arguments, prepared.constants)
        environment["SONIC_ROBOT_URDF"] = str(prepared.robot_urdf)
        environment["PYTHONPATH"] = str(prepared.source) + os.pathsep + environment["PYTHONPATH"]
        for directory in directories:
            if directory == output:
                continue
            for name in ("inputs", "source", "source-manifest.json"):
                (directory / name).symlink_to(Path("../..") / name)
        physics_command = [
            "taskset",
            "-c",
            cpu_allocation.physics,
            str(arguments.isaac_workspace / "IsaacLab/.venv/bin/python"),
            "-m",
            "humanoid_agent",
            "--role",
            "physics",
            "--output",
            str(output / "shared-physics"),
            "--task-scene",
            str(prepared.scene),
            "--native-workspace",
            str(arguments.native_workspace),
            "--isaac-workspace",
            str(arguments.isaac_workspace),
            "--wall-seconds",
            str(arguments.wall_seconds),
            "--num-environments",
            str(layout.num_environments),
        ]
        if arguments.external_camera:
            physics_command.append("--external-camera")
        physics_command.extend(["--episode-kind", arguments.episode_kind, "--mode", arguments.mode])
        if arguments.episode_kind == "controller-check":
            physics_command.extend(
                [
                    "--diagnostic-camera",
                    arguments.diagnostic_camera,
                    "--diagnostic-motion",
                    arguments.diagnostic_motion,
                    "--diagnostic-phase-seconds",
                    str(arguments.diagnostic_phase_seconds),
                    "--capture-interval-seconds",
                    str(arguments.capture_interval_seconds),
                ]
            )
        if single_episode:
            physics_command.extend(["--episode-directory", str(output)])
        if prepared.decorations is not None:
            physics_command.extend(["--decorative-assets", str(prepared.decorations)])
        components = [("physics", physics_command, output / "shared-physics.log")]
        for index, directory in enumerate(directories):
            operator_arguments = replace(arguments, provider=providers[index], output=directory)
            preflight_episode(operator_arguments, directory, prepared.source, environment)
            operator_command = (
                model_operator_command(operator_arguments, directory)
                if arguments.episode_kind == "model"
                else None
            )
            atomic_json(
                directory / "episode-configuration.json",
                {
                    "kind": arguments.episode_kind,
                    "motion_mode": arguments.mode,
                    "provider": providers[index],
                    "environment_index": index,
                    "shared_owner": "." if single_episode else "../..",
                    "physics_command": physics_command,
                    "operator_command": operator_command,
                    "execution_topology": execution_topology,
                    "gpu_scheduling": gpu_scheduling,
                    "cpu_allocation": asdict(cpu_allocation),
                    "communication": "Host-local queues and atomic files; private native controller transport per robot",
                },
            )
            if operator_command is not None:
                components.append(
                    (f"operator_{index}", operator_command, directory / "operator.log")
                )
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        if layout.num_environments > 1:
            environment = gpu_service.start(environment)
        for name, command, log_path in components:
            log = log_path.open("wb")
            logs.append(log)
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            processes[name] = process
            atomic_json(
                output / "processes.json",
                {role: process_record(child.pid) for role, child in processes.items()},
                durable=True,
            )
        startup_deadline = time.monotonic() + 900
        while processes["physics"].poll() is None:
            if gpu_service.process is not None:
                gpu_service.check()
            if stopping.is_set():
                status = "cancelled"
                break
            if episode_clock is None:
                operator_state = OperatorClient(directories[0] / "commands").request(
                    QueryOperatorState(), time.monotonic() + 0.05
                )
                if operator_state is not None:
                    episode_clock = operator_state.clock
            deadline = (
                episode_clock.deadline_monotonic_seconds + 120
                if episode_clock is not None
                else startup_deadline
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Shared group exceeded initialization or absolute deadline plus cleanup grace"
                )
            for index, directory in enumerate(directories):
                name = f"operator_{index}"
                operator = processes.get(name)
                if operator is None:
                    continue
                outcome_exists = (directory / "physics/task-outcome.json").exists()
                if outcome_exists and operator.poll() is None and name not in interrupted:
                    interrupted.add(name)
                    operator.send_signal(signal.SIGINT)
                if operator.poll() not in (None, 0):
                    error_path = directory / "operator/operator-error.json"
                    operator_error = (
                        json.loads(error_path.read_text()) if error_path.exists() else None
                    )
                    failure = reconcile_execution(
                        0, operator.returncode, operator_error, name in interrupted, outcome_exists
                    )
                    if failure is not None:
                        raise RuntimeError(f"Robot {index}: {failure}")
            time.sleep(0.05)
        else:
            if processes["physics"].returncode != 0:
                raise RuntimeError(
                    f"Shared physics exited {processes['physics'].returncode}; inspect shared-physics.log"
                )
            shared_result = json.loads((output / "shared-physics/result.json").read_text())
            if shared_result["status"] != "completed":
                raise RuntimeError(
                    "Shared physics failed; inspect shared-physics/result.json and component logs"
                )
            for index, directory in enumerate(directories):
                name = f"operator_{index}"
                operator = processes.get(name)
                if operator is not None:
                    if operator.poll() is None:
                        interrupted.add(name)
                    stop_process(operator)
                error_path = directory / "operator/operator-error.json"
                operator_error = json.loads(error_path.read_text()) if error_path.exists() else None
                failure = reconcile_execution(
                    processes["physics"].returncode,
                    operator.returncode if operator is not None else None,
                    operator_error,
                    name in interrupted,
                    (directory / "physics/task-outcome.json").is_file(),
                )
                if failure is not None:
                    raise RuntimeError(f"Robot {index}: {failure}")
            status = "completed"
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        logger.exception("Shared episode group failed")
        raise
    finally:
        for name in [f"operator_{index}" for index in range(layout.num_environments)] + ["physics"]:
            process = processes.get(name)
            if process is not None:
                try:
                    stop_process(process, output / "shared-physics" if name == "physics" else None)
                    if name == "physics":
                        for directory in directories:
                            stop_process(process, directory / "physics")
                except Exception as exception:
                    cleanup_errors.append(f"{name}: {type(exception).__name__}: {exception}")
        try:
            gpu_service.close()
        except Exception as exception:
            cleanup_errors.append(f"GPU scheduling: {type(exception).__name__}: {exception}")
        for log in logs:
            log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        terminals = []
        for index, directory in enumerate(directories):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            episode_cleanup = list(cleanup_errors)
            try:
                interrupt_pending_attempts(directory / "operator")
            except Exception as exception:
                episode_cleanup.append(
                    f"API attempt evidence: {type(exception).__name__}: {exception}"
                )
            component_codes = (
                {"physics": processes["physics"].returncode} if "physics" in processes else {}
            )
            operator = processes.get(f"operator_{index}")
            if gpu_service.process is not None:
                component_codes["gpu_scheduling"] = gpu_service.process.returncode
            if operator is not None:
                component_codes["operator"] = operator.returncode
            terminal = write_terminal_episode(
                directory,
                provider=providers[index],
                episode_kind=arguments.episode_kind,
                protocol=prepared.protocol if prepared is not None else None,
                status="infrastructure_error" if episode_cleanup else status,
                error=error,
                cleanup_errors=episode_cleanup,
                component_exit_codes=component_codes,
            )
            terminals.append(terminal)
        group_status = (
            "infrastructure_error"
            if cleanup_errors
            or any(terminal.status == "infrastructure_error" for terminal in terminals)
            else status
        )
        atomic_json(
            output / "group.json",
            {
                "schema_version": "webolt-episode-group-v1",
                "status": group_status,
                "error": error,
                "cleanup_errors": cleanup_errors,
                "component_exit_codes": {
                    **{name: process.returncode for name, process in processes.items()},
                    **(
                        {"gpu_scheduling": gpu_service.process.returncode}
                        if gpu_service.process is not None
                        else {}
                    ),
                },
                "episodes": [str(directory.relative_to(output)) for directory in directories],
            },
            durable=True,
        )
        logger.info("Shared episode group ended: %s", group_status)
        if group_status == "infrastructure_error" and error is None:
            raise RuntimeError("Group infrastructure failed; inspect retained episode records")
