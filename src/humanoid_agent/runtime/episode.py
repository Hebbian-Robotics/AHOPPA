"""Own one GPU-host episode and retain its independent component records."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

from humanoid_agent.artifact_io import atomic_json, canonical_json_sha256
from humanoid_agent.experiment_records import (
    TASK_OUTCOME_ADAPTER,
    EpisodeProtocol,
    EpisodeRecord,
    PhysicsResult,
    validate_physical_record,
)
from humanoid_agent.facility_map import FACILITY_MAP_FILENAME, render_facility_map
from humanoid_agent.isaac_workcell import GroundWorkcellScene
from humanoid_agent.observation_history import OBSERVATION_INTERFACE_PROTOCOL
from humanoid_agent.paths import PROJECT_ROOT
from humanoid_agent.process_lifecycle import ProcessIdentity, process_identity
from humanoid_agent.recording import read_record_bytes, record_exists
from humanoid_agent.runtime.cpu import CpuAllocation
from humanoid_agent.runtime.diagnostic import episode_command_protocol
from humanoid_agent.sensor_feedback import PROTOCOL
from humanoid_agent.task_balance import ISAAC_CONTACT_BALANCE_PROTOCOL
from humanoid_agent.task_outcome import COMPLETION_PROTOCOL

logger = logging.getLogger(__name__)


def stop_process(
    process: subprocess.Popen[bytes],
    records: Path | None = None,
    *,
    extra_owned: Sequence[ProcessIdentity] = (),
) -> None:
    """Stop the owner and identified descendants, including recorded orphan workers."""
    owned = {identity.pid: identity for identity in extra_owned}
    owner = process_identity(process.pid) if process.poll() is None else None
    if owner is not None:
        owned[owner.pid] = owner
    if records is not None:
        record_directories = [records, *records.glob("robot_[0-9]*")]
        for name in (
            "controller-process.json",
            "sdk-worker-process.json",
            "camera-worker-process.json",
            "io-worker-process.json",
        ):
            for directory in record_directories:
                record_path = directory / name
                if record_path.exists():
                    identity = ProcessIdentity.model_validate_json(record_path.read_text())
                    if identity.matching_process() is not None:
                        owned[identity.pid] = identity
    # Discover before stopping parents; recorded roots also cover reparented workers.
    for identity in list(owned.values()):
        root = identity.matching_process()
        if root is not None:
            with suppress(psutil.NoSuchProcess):
                for child in root.children(recursive=True):
                    child_identity = process_identity(child.pid)
                    if child_identity is not None:
                        owned[child.pid] = child_identity
    if process.poll() is None and owner is not None:
        current_owner = owner.matching_process()
        if current_owner is not None:
            with suppress(psutil.NoSuchProcess):
                current_owner.send_signal(signal.SIGINT)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=90)
    remaining = []
    for identity in reversed(list(owned.values())):
        child = identity.matching_process()
        if child is not None:
            with suppress(psutil.NoSuchProcess):
                child.kill()
                remaining.append(child)
    if process.poll() is None:
        process.wait(timeout=5)
    _, alive = psutil.wait_procs(remaining, timeout=5)
    if any(owned[child.pid].matching_process() is not None for child in alive):
        raise TimeoutError("Owned process descendants did not terminate after forced shutdown")


def interrupt_pending_attempts(operator_directory: Path) -> None:
    for path in operator_directory.rglob("*.timing.json"):
        record = json.loads(path.read_text())
        if record.get("state") == "started":
            atomic_json(
                path,
                {
                    **record,
                    "state": "interrupted",
                    "recorded_by": "episode_owner",
                    "ended_monotonic_seconds": time.monotonic(),
                    "provider_completion_known": False,
                },
            )


def create_host_process_environment(arguments: Any, constants: Path) -> dict[str, str]:
    """Reuse installed SONIC transport and renderer dependencies on this host."""
    return {
        **os.environ,
        **CpuAllocation.from_options(arguments).environment(),
        "HUMANOID_AGENT_PROJECT_ROOT": str(PROJECT_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "SONIC_SOURCE": str(arguments.native_workspace / "SONIC"),
        "SONIC_CONSTANTS": str(constants.resolve()),
        "MUJOCO_GL": "egl",
        "PYTHONPATH": os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[2]),
                str(arguments.native_workspace / "SONIC"),
                str(arguments.native_workspace / "SONIC/external_dependencies/unitree_sdk2_python"),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [
                str(arguments.native_workspace / "cyclonedds/lib"),
                str(arguments.system_libraries),
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        ),
    }


def reconcile_execution(
    physics_exit_code: int | None,
    operator_exit_code: int | None,
    operator_error: dict[str, Any] | None,
    operator_interrupted_by_owner: bool,
    task_outcome_exists: bool,
) -> str | None:
    if physics_exit_code != 0:
        return "Physics failed; inspect physics.log and physics/execution-error.json"
    expected_interruption = (
        operator_interrupted_by_owner
        and operator_exit_code in {0, -signal.SIGINT, 128 + signal.SIGINT}
        and (
            operator_error is None
            or operator_error.get("exception_type") in {"KeyboardInterrupt", "SystemExit"}
        )
    )
    if (
        operator_exit_code not in (None, 0) or operator_error is not None
    ) and not expected_interruption:
        return "Operator failed; physical task outcome retained independently"
    if not task_outcome_exists:
        return "Physics exited without an independent task outcome"
    return None


def require_episode_storage(output: Path, minimum_free_bytes: int = 5 * 1024**3) -> None:
    """Reserve an engineering margin for evidence before starting components.

    This check cannot reserve disk space or prevent another job filling it later.
    Five GiB is a startup floor, not a prediction of an episode's final size.
    """
    if minimum_free_bytes < 1:
        raise ValueError("Episode storage margin must be positive")
    available_bytes = shutil.disk_usage(output).free
    passed = available_bytes >= minimum_free_bytes
    atomic_json(
        output / "storage-preflight.json",
        {
            "passed": passed,
            "available_bytes": available_bytes,
            "minimum_free_bytes": minimum_free_bytes,
            "scope": "Startup evidence margin; space is not reserved",
        },
    )
    if not passed:
        raise OSError(
            f"Insufficient episode evidence storage: {available_bytes} free bytes; "
            f"requires at least {minimum_free_bytes} before starting components"
        )


def preflight_episode(
    arguments: Any, output: Path, source: Path, environment: dict[str, str]
) -> None:
    from humanoid_agent.canonical_robot import ControllerConstants, require_canonical_urdf
    from humanoid_agent.runtime.robot import read_sonic_robot_parameters

    require_episode_storage(output)
    ControllerConstants.model_validate_json(Path(environment["SONIC_CONSTANTS"]).read_text())
    asset_root = arguments.isaac_workspace / "SONIC/gear_sonic/data/assets"
    robot = ElementTree.fromstring(require_canonical_urdf(Path(environment["SONIC_ROBOT_URDF"])))
    mesh_paths = set()
    for mesh in robot.iter("mesh"):
        filename = mesh.attrib["filename"]
        if not filename.startswith("package://robot_description/"):
            raise ValueError("Unexpected robot mesh URI: " + filename)
        path = asset_root / filename.removeprefix("package://")
        with path.open("rb") as stream:
            header = stream.read(200)
        if len(header) < 84 or header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("Missing mesh payload or Git LFS pointer: " + str(path))
        mesh_paths.add(path)
    read_sonic_robot_parameters(asset_root.parents[2] / "gear_sonic/envs/manager_env/robots/g1.py")
    environment = {
        **environment,
        "PYTHONPATH": str(source) + os.pathsep + environment["PYTHONPATH"],
    }
    checks = [
        (
            "physics",
            arguments.isaac_workspace / "IsaacLab/.venv/bin/python",
            "import torch, newton, cyclonedds, zmq; import humanoid_agent.runtime.shared; from gear_sonic.trl.utils.order_converter import G1Converter; G1Converter(); assert torch.cuda.is_available(), 'CUDA unavailable'",
        )
    ]
    if arguments.episode_kind == "model":
        checks.append(
            (
                "operator",
                Path(sys.executable),
                "from humanoid_agent.task_model_client import TaskModelClient; from humanoid_agent.model_api_configuration import ApiServiceTier, ReasoningEffort; "
                "from humanoid_agent.sensor_feedback import EncoderKinematics; import sys; "
                "TaskModelClient(sys.argv[1], service_tier=ApiServiceTier(sys.argv[2]), request_timeout_seconds=float(sys.argv[3]), reasoning_effort=ReasoningEffort(sys.argv[4])); EncoderKinematics(); "
                "from humanoid_agent.hand_control import upstream_hand_positions; upstream_hand_positions('left', 'open'); upstream_hand_positions('right', 'close')",
            )
        )
    for name, python, code in checks:
        with (output / f"preflight-{name}.log").open("wb") as log:
            subprocess.run(
                [
                    str(python),
                    "-c",
                    code,
                    arguments.provider,
                    arguments.api_service_tier.value,
                    str(arguments.api_request_timeout_seconds),
                    arguments.reasoning_effort.value,
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=True,
            )
    atomic_json(
        output / "preflight.json",
        {
            "passed": True,
            "meshes_checked": len(mesh_paths),
            "checks": [name for name, _, _ in checks],
            "model_requests": 0,
        },
    )


@dataclass(frozen=True)
class PreparedEpisodeInputs:
    constants: Path
    robot_urdf: Path
    scene: Path
    decorations: Path | None
    source: Path
    protocol: EpisodeProtocol


def prepare_episode_inputs(
    arguments: Any,
    output: Path,
    *,
    execution_topology: Literal["single-robot", "shared-physics"] = "single-robot",
    gpu_scheduling: Literal["driver-default", "cuda-mps"] = "driver-default",
) -> PreparedEpisodeInputs:
    """Freeze exactly the inputs used by physics and operators before starting either."""
    if any(name.startswith("CUDA_MPS_") for name in os.environ):
        raise ValueError(
            "Episode owners require an unmodified MPS environment and own GPU scheduling"
        )
    if arguments.task_scene is None:
        raise ValueError("An episode requires --task-scene")
    inputs = output / "inputs"
    inputs.mkdir()
    constants = inputs / "sonic_constants.json"
    shutil.copyfile(arguments.sonic_constants, constants)
    robot_urdf = inputs / "robot.urdf"
    shutil.copyfile(
        arguments.isaac_workspace
        / "SONIC/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
        robot_urdf,
    )
    scene = inputs / "scene.json"
    shutil.copyfile(arguments.task_scene, scene)
    scene_definition = GroundWorkcellScene.model_validate_json(scene.read_text())
    facility_map = inputs / FACILITY_MAP_FILENAME
    render_facility_map(scene_definition, facility_map)
    decorations = None
    if arguments.decorative_assets is not None:
        decorations = inputs / "appearance"
        shutil.copytree(arguments.decorative_assets, decorations)
    source = output / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        source / "humanoid_agent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    atomic_json(
        output / "source-manifest.json",
        {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*.py"))
        },
    )
    shutil.copyfile(arguments.isaac_workspace / "IsaacLab/uv.lock", inputs / "physics-uv.lock")
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(PROJECT_ROOT / filename, inputs / filename)
    recording_runtime = inputs / "recording-runtime-requirements.txt"
    shutil.copyfile(PROJECT_ROOT / "configs/recording-runtime-requirements.txt", recording_runtime)
    protocol = EpisodeProtocol(
        source_manifest_sha256=canonical_json_sha256(
            json.loads((output / "source-manifest.json").read_text())
        ),
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        appearance_manifest_sha256=hashlib.sha256(
            (decorations / "manifest.json").read_bytes()
        ).hexdigest()
        if decorations
        else None,
        task_instruction_sha256=hashlib.sha256(arguments.task_instruction.encode()).hexdigest(),
        facility_map_sha256=hashlib.sha256(facility_map.read_bytes()).hexdigest(),
        wall_seconds=arguments.wall_seconds,
        execution_topology=execution_topology,
        gpu_scheduling=gpu_scheduling,
        sensor_protocol=PROTOCOL,
        observation_interface_protocol=OBSERVATION_INTERFACE_PROTOCOL
        if arguments.external_camera
        else None,
        recording_protocol="webolt-recording-v1",
        recording_runtime_sha256=hashlib.sha256(recording_runtime.read_bytes()).hexdigest(),
        command_protocol=episode_command_protocol(arguments),
        completion_protocol=COMPLETION_PROTOCOL,
        balance_contact_protocol=ISAAC_CONTACT_BALANCE_PROTOCOL,
        clock_policy="wall-clock-lag-v1",
        sonic_constants_sha256=hashlib.sha256(constants.read_bytes()).hexdigest(),
        robot_urdf_sha256=hashlib.sha256(robot_urdf.read_bytes()).hexdigest(),
        api_service_tier=arguments.api_service_tier.value,
        api_request_timeout_seconds=arguments.api_request_timeout_seconds,
        api_retry_policy=arguments.api_retry_policy.value,
        reasoning_effort=arguments.reasoning_effort.value,
        model_max_output_tokens=arguments.model_max_output_tokens,
        project_lock_sha256=hashlib.sha256((inputs / "uv.lock").read_bytes()).hexdigest(),
        physics_lock_sha256=hashlib.sha256((inputs / "physics-uv.lock").read_bytes()).hexdigest(),
    )
    return PreparedEpisodeInputs(constants, robot_urdf, scene, decorations, source, protocol)


def uses_stepped_turns(scene_path: Path) -> bool:
    """The scene declares whether the model gets the stepped in-place turn tool."""
    if not scene_path.exists():
        return False
    return GroundWorkcellScene.model_validate_json(scene_path.read_text()).stepped_turn_tool


def uses_leashed_facing(scene_path: Path) -> bool:
    """The scene declares whether raw facing changes are approached gradually."""
    if not scene_path.exists():
        return False
    return GroundWorkcellScene.model_validate_json(scene_path.read_text()).leashed_facing


def model_operator_command(arguments: Any, output: Path) -> list[str]:
    """Construct the native model operator command for the episode owner."""
    commands = output / "commands"
    return [
        "taskset",
        "-c",
        CpuAllocation.from_options(arguments).operator,
        sys.executable,
        "-m",
        "humanoid_agent",
        "--role",
        "operator",
        "--output",
        str(output / "operator"),
        "--physics-directory",
        str(output / "physics"),
        "--command-directory",
        str(commands),
        "--mode",
        arguments.mode,
        "--provider",
        arguments.provider,
        "--wall-seconds",
        str(arguments.wall_seconds),
        "--task-instruction",
        arguments.task_instruction,
        *(
            ["--facility-map", str(output / "inputs" / FACILITY_MAP_FILENAME)]
            if (output / "inputs" / FACILITY_MAP_FILENAME).exists()
            else []
        ),
        *(["--stepped-turns"] if uses_stepped_turns(output / "inputs" / "scene.json") else []),
        *(["--leashed-facing"] if uses_leashed_facing(output / "inputs" / "scene.json") else []),
        "--api-service-tier",
        arguments.api_service_tier.value,
        *(["--external-camera"] if arguments.external_camera else []),
        "--reasoning-effort",
        arguments.reasoning_effort.value,
        "--model-max-output-tokens",
        str(arguments.model_max_output_tokens),
        "--api-request-timeout-seconds",
        str(arguments.api_request_timeout_seconds),
        "--api-retry-policy",
        arguments.api_retry_policy.name.lower(),
    ]


def write_terminal_episode(
    output: Path,
    *,
    provider: str | None,
    episode_kind: str | None,
    protocol: EpisodeProtocol | None,
    status: str,
    error: str | None,
    cleanup_errors: list[str],
    component_exit_codes: Mapping[str, int | None],
) -> EpisodeRecord:
    """Persist one canonical record after all component states have been reconciled."""
    outcome_path = output / "physics/task-outcome.json"
    task_outcome = None
    simulation_validity = {"status": "not_established"}
    terminal_errors = list(cleanup_errors)
    try:
        if record_exists(outcome_path):
            task_outcome = TASK_OUTCOME_ADAPTER.validate_json(
                read_record_bytes(outcome_path)
            ).model_dump(mode="json")
    except (OSError, ValueError) as outcome_error:
        terminal_errors.append(f"Task outcome unreadable: {type(outcome_error).__name__}")
    try:
        result_path = output / "physics/result.json"
        if record_exists(result_path):
            simulation_validity = PhysicsResult.model_validate_json(
                read_record_bytes(result_path)
            ).simulation_validity.model_dump(mode="json")
    except (OSError, ValueError) as result_error:
        terminal_errors.append(f"Physics result unreadable: {type(result_error).__name__}")
    if terminal_errors:
        status = "infrastructure_error"
    terminal_payload = {
        "schema_version": "webolt-episode-v3",
        "provider": provider,
        "episode_kind": episode_kind,
        "protocol": protocol,
        "status": status,
        "error": error,
        "cleanup_errors": terminal_errors,
        "component_exit_codes": component_exit_codes,
        "task_outcome": task_outcome,
        "simulation_validity": simulation_validity,
        "analysis": "not_run",
        "rendering": "not_run",
        "publication": "not_run",
    }
    try:
        terminal = EpisodeRecord.model_validate(terminal_payload)
        if terminal.status == "completed":
            validate_physical_record(output, terminal)
    except (OSError, ValueError) as evidence_error:
        if status != "completed":
            raise
        terminal_payload["status"] = "infrastructure_error"
        terminal_errors.append(
            f"Terminal evidence invalid: {type(evidence_error).__name__}: {evidence_error}"
        )
        terminal = EpisodeRecord.model_validate(terminal_payload)
    atomic_json(output / "episode.json", terminal.model_dump(mode="json"), durable=True)
    return terminal


def run_episode(arguments: Any) -> None:
    """Run the ordinary owner with a single environment and root episode records."""
    from humanoid_agent.runtime.group import run_episode_group

    if arguments.num_environments != 1:
        raise ValueError("Use explicit --providers with --role group for multiple environments")
    run_episode_group(arguments, single_episode=True)
