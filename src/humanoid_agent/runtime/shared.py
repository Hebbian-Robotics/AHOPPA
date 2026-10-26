"""Shared Newton physics, tiled rendering and independent native SONIC controllers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import uuid
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import CANONICAL_URDF_SHA256, JOINT_NAMES, POINT_OFFSETS
from humanoid_agent.dex3 import HAND_JOINT_NAMES, HandEncoderState, articulated_hand_urdf
from humanoid_agent.hand_control import HandTransitionInitials
from humanoid_agent.isaac_workcell import (
    TASK_CONTACT_PAIRS,
    GroundWorkcellScene,
    task_contact_evidence,
    task_contact_evidence_batch,
    task_contact_tensor_fields,
)
from humanoid_agent.runtime.actor import RobotEpisode
from humanoid_agent.runtime.camera import NativeCameraProcess, NativeVisualBatch, NativeVisualState
from humanoid_agent.runtime.contact_capacity import ContactCapacityMonitor
from humanoid_agent.runtime.deployment import EXECUTABLE_SHA256
from humanoid_agent.runtime.diagnostic import episode_command_protocol, scripted_episode_options
from humanoid_agent.runtime.environment import (
    create_newton_simulation_configuration,
    create_newton_workcell_scene,
    launch_newton_runtime,
    register_newton_scene_schemas,
)
from humanoid_agent.runtime.episode_io import (
    AppendRecord,
    CaptureBatchSnapshot,
    EpisodeIO,
    ObservationReady,
    WriteRecord,
)
from humanoid_agent.runtime.evaluation import GroundTaskEvaluation, GroundTaskMeasurements
from humanoid_agent.runtime.layout import EpisodeLayout
from humanoid_agent.runtime.measurements import (
    TaskMeasurementSnapshot,
    cabinet_joint_positions,
    measurements,
    shutdown_tensor_fields,
    storage_occluders,
    task_measurement_tensor_fields,
)
from humanoid_agent.runtime.motor_targets import BatchedMotorTargets
from humanoid_agent.runtime.native_control import neutral_planner_sample
from humanoid_agent.runtime.robot import SONIC_CONFIGURATION_SHA256
from humanoid_agent.runtime.scripted_control import StandingControlSource
from humanoid_agent.runtime.sdk import (
    SonicEncoderImuState,
    SonicSdkProcessBridge,
    motor_order_indices,
)
from humanoid_agent.runtime.shutdown_reference import ShutdownReferenceControl
from humanoid_agent.runtime.state import (
    TickSnapshotCache,
    clock_validity,
    recorded_state_from_newton,
)
from humanoid_agent.runtime.stepped_turn import TurnCheckControl
from humanoid_agent.runtime.timing import DurationSummary, EpisodeTiming
from humanoid_agent.sensor_feedback import EncoderKinematics


class PhysicsCommandSource(Enum):
    MODEL = "model"
    SCRIPTED = "scripted"


def run_episode_physics(arguments: Any) -> None:
    """Use the production owner with either model operators or scripted controller inputs."""
    command_source = PhysicsCommandSource.MODEL
    if arguments.episode_kind == "controller-check":
        arguments = scripted_episode_options(arguments)
        command_source = PhysicsCommandSource.SCRIPTED
    run_shared_physics(arguments, command_source=command_source)


def run_shared_physics(arguments: Any, *, command_source: PhysicsCommandSource) -> None:
    """Keep one shared physics clock, independent controller states and a tiled camera worker."""
    scripted_episodes = command_source == PhysicsCommandSource.SCRIPTED
    camera_enabled = not scripted_episodes or arguments.diagnostic_camera == "on"
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "infrastructure_error",
        "model_requests": 0,
        "scope": "Production episode physics with explicit command source",
        "command_source": command_source.value,
        "camera_enabled": camera_enabled,
    }
    bridges: list[SonicSdkProcessBridge] = []
    actors: list[RobotEpisode] = []
    finished_actors: set[int] = set()
    camera = None
    evidence_io = None
    runtime = None
    runtime_started = False
    episode_timing = EpisodeTiming(0.005)
    contact_capacity = ContactCapacityMonitor()
    previous_lag = 0.0
    try:
        layout = EpisodeLayout(
            output.parent, arguments.num_environments, arguments.episode_directory is not None
        )
        if arguments.episode_directory is not None and (
            arguments.episode_directory.resolve() != layout.root.resolve()
        ):
            raise ValueError("The single episode directory must be the physics owner's parent")
        if output.resolve() != layout.physics_directory.resolve():
            raise ValueError("Physics output must be the owner's shared-physics directory")
        if camera_enabled:
            camera = NativeCameraProcess(
                Path(sys.executable),
                arguments.isaac_workspace / "SONIC",
                arguments.task_scene,
                arguments.decorative_assets,
                output,
                num_environments=layout.num_environments,
                batch_captures=True,
                external_camera=arguments.external_camera,
                watchdog_seconds=arguments.wall_seconds + 900,
            )
        register_newton_scene_schemas()
        torch: Any = importlib.import_module("torch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        simulation_module: Any = importlib.import_module("isaaclab.sim")
        configuration = create_newton_simulation_configuration()
        runtime = launch_newton_runtime(configuration)
        runtime.__enter__()
        runtime_started = True
        simulation = simulation_module.SimulationContext(configuration)
        scene_definition = GroundWorkcellScene.model_validate_json(arguments.task_scene.read_text())
        scene = create_newton_workcell_scene(
            arguments.isaac_workspace / "SONIC/gear_sonic/data/assets",
            scene_definition,
            contact_sensors=True,
            display_collision_boxes=True,
            num_environments=layout.num_environments,
        )
        if scene_definition.dex3_hands:
            definition = articulated_hand_urdf(
                arguments.isaac_workspace
                / "SONIC/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
            )
            shutil.copyfile(definition, output / "articulated-robot.urdf")
            atomic_json(
                output / "articulated-robot.json",
                {
                    "canonical_source_sha256": CANONICAL_URDF_SHA256,
                    "derived_urdf_sha256": hashlib.sha256(definition.read_bytes()).hexdigest(),
                    "adaptation": "Enable fourteen existing finger joints; preserve body transforms, inertias, limits and mesh geometry",
                },
            )
        robot = scene["robot"]
        shutdown = scene_definition.shutdown
        cabinets = [scene["cabinet_" + cabinet.identifier] for cabinet in shutdown.cabinets]
        simulation.reset()
        scene.reset()
        origins = scene.env_origins.cpu().numpy()
        root_defaults = robot.data.default_root_pose.torch.clone()
        root_defaults[:, :3] += scene.env_origins
        root_defaults[:, :2] += torch.tensor(scene_definition.robot_start_xy_m, device="cuda:0")
        joint_defaults = robot.data.default_joint_pos.torch.clone()
        initialization_started = time.monotonic()
        for _ in range(2):
            robot.actuators.target_command.set_position_index(value=joint_defaults)
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(configuration.dt)
            for name, _, _ in TASK_CONTACT_PAIRS:
                scene["task_contact_" + name].data.force_matrix_w.torch.cpu().numpy()
            for name in ("pelvis", "torso"):
                scene["imu_" + name].data.lin_acc_b.torch.cpu().numpy()
        scene.reset()

        def restore_initial_state() -> None:
            robot.write_root_pose_to_sim_index(root_pose=root_defaults)
            robot.write_root_velocity_to_sim_index(
                root_velocity=robot.data.default_root_vel.torch.clone()
            )
            robot.write_joint_position_to_sim_index(position=joint_defaults)
            robot.write_joint_velocity_to_sim_index(
                velocity=robot.data.default_joint_vel.torch.clone()
            )
            for cabinet in cabinets:
                # Drawer states are assigned only at reset; physics owns them afterwards.
                cabinet.write_joint_position_to_sim_index(
                    position=cabinet.data.default_joint_pos.torch.clone()
                )
                cabinet.write_joint_velocity_to_sim_index(
                    velocity=cabinet.data.default_joint_vel.torch.clone()
                )
            simulation.forward()
            robot.update(configuration.dt)

        restore_initial_state()
        atomic_json(
            output / "cabinet-articulation.json",
            {
                cabinet.identifier: {
                    "body_names": articulation.body_names,
                    "body_mass_kg": articulation.data.body_mass.torch[0].tolist(),
                    "joint_names": articulation.joint_names,
                    "joint_stiffness": articulation.data.joint_stiffness.torch[0].tolist(),
                    "joint_damping": articulation.data.joint_damping.torch[0].tolist(),
                    "joint_friction_coefficient": articulation.data.joint_friction_coeff.torch[
                        0
                    ].tolist(),
                    "initial_joint_positions": articulation.data.default_joint_pos.torch[
                        0
                    ].tolist(),
                }
                for cabinet, articulation in zip(shutdown.cabinets, cabinets, strict=True)
            },
        )
        initialization_record = {
            "warmup_steps_before_controller": 2,
            "restored_neutral_pose": True,
            "duration_seconds": time.monotonic() - initialization_started,
            "episode_started": False,
            "model_requests": 0,
        }
        atomic_json(output / "physics-initialization.json", initialization_record)
        episode_ids = [uuid.uuid4().hex for _ in range(layout.num_environments)]
        motor_indices = motor_order_indices(robot.joint_names, JOINT_NAMES)
        hand_indices = (
            {
                side: [robot.joint_names.index(name) for name in names]
                for side, names in HAND_JOINT_NAMES.items()
            }
            if scene_definition.dex3_hands
            else {}
        )
        torso_index = robot.body_names.index("torso_link")
        encoder = EncoderKinematics()
        evaluations = [GroundTaskEvaluation(shutdown) for _ in range(layout.num_environments)]
        contact_environment = SimpleNamespace(scene=scene, cfg=SimpleNamespace(decimation=4))

        snapshots = TickSnapshotCache()

        def visual_tensor_fields() -> dict[str, Any]:
            return {
                **{
                    f"cabinet_{cabinet.identifier}_joint_positions": articulation.data.joint_pos.torch
                    for cabinet, articulation in zip(shutdown.cabinets, cabinets, strict=True)
                },
                "root_states": robot.data.root_state_w.torch,
                "joint_positions": robot.data.joint_pos.torch,
                "joint_velocities": robot.data.joint_vel.torch,
            }

        contact_snapshot_tick = -1
        contact_snapshot: list[dict[str, Any]] = []

        def contact_for_robot(environment_index: int) -> dict[str, Any]:
            nonlocal contact_snapshot_tick, contact_snapshot
            if contact_snapshot_tick != tick:
                contact_snapshot = task_contact_evidence_batch(
                    contact_environment,
                    snapshot=snapshots.read(tick, task_contact_tensor_fields(contact_environment)),
                )
                contact_snapshot_tick = tick
            return contact_snapshot[environment_index]

        measurement_snapshot: TaskMeasurementSnapshot | None = None
        measurement_snapshot_tick = -1

        def measure_robot(environment_index: int) -> GroundTaskMeasurements:
            nonlocal measurement_snapshot, measurement_snapshot_tick
            if measurement_snapshot_tick != tick:
                measurement_snapshot = TaskMeasurementSnapshot(
                    **snapshots.read(
                        tick,
                        task_measurement_tensor_fields(robot, include_joint_positions=True),
                    ),
                    shutdown_fields=snapshots.read(tick, shutdown_tensor_fields(scene, shutdown)),
                    cabinet_joint_names=cabinets[0].joint_names,
                    shutdown_contact_body_names=scene[
                        "shutdown_drawer_contacts_" + shutdown.cabinets[0].identifier
                    ].body_names,
                )
                measurement_snapshot_tick = tick
            return measurements(
                robot,
                scene_definition,
                contact_for_robot(environment_index),
                environment_index=environment_index,
                environment_origin_m=origins[environment_index],
                snapshot=measurement_snapshot,
            )

        max_displacements = [0.0] * layout.num_environments
        tick = 0
        release_at = None

        def snapshot_batch(identifier: str) -> NativeVisualBatch:
            fields = snapshots.read(tick, visual_tensor_fields())
            roots = fields["root_states"]
            joints = fields["joint_positions"]
            velocities = fields["joint_velocities"]
            sampled_at = snapshots.sampled_monotonic_seconds(tuple(fields))
            captured_elapsed = 0.0 if release_at is None else sampled_at - release_at
            return NativeVisualBatch(
                identifier=identifier,
                states=[
                    NativeVisualState(
                        identifier=identifier,
                        episode_id=episode_ids[index],
                        episode_deadline_elapsed_wall_seconds=arguments.wall_seconds,
                        simulation_seconds=tick * configuration.dt,
                        elapsed_wall_seconds=captured_elapsed,
                        source_monotonic_seconds=sampled_at,
                        joint_names=robot.joint_names,
                        joint_positions=joints[index].tolist(),
                        joint_velocities=velocities[index].tolist(),
                        root_state_world=recorded_state_from_newton(roots[index]).tolist(),
                        cabinet_joint_positions=cabinet_joint_positions(
                            fields, shutdown, cabinets[0].joint_names, index
                        ),
                        indicator_activated=evaluations[index].activated,
                    )
                    for index in range(layout.num_environments)
                ],
            )

        packed_contacts = task_contact_evidence_batch(contact_environment)
        if any(
            packed_contacts[index] != task_contact_evidence(contact_environment, index)
            for index in range(len(packed_contacts))
        ):
            raise ValueError("Packed contact evidence differs from original sensor reads")
        atomic_json(
            output / "contact-packing-check.json",
            {
                "passed": True,
                "robot_count": len(packed_contacts),
                "phase": "pre-release warmup",
                "model_requests": 0,
            },
        )
        initial = snapshot_batch("initial")
        if camera is not None:
            camera.publish(initial)
            camera.wait_for_initial_batch()
        neutral_samples = []
        for index, state in enumerate(initial.states):
            neutral_samples.append(neutral_planner_sample(state, encoder=encoder))
            directory = output / f"robot_{index}"
            directory.mkdir()
            bridges.append(
                SonicSdkProcessBridge(
                    Path(sys.executable),
                    directory,
                    native_workspace=arguments.native_workspace,
                    dex3_hands=scene_definition.dex3_hands,
                )
            )
            bridges[-1].publish_planner(neutral_samples[-1])
        for index, state in enumerate(initial.states):
            episode_directory = layout.episode_directories[index]
            actor_output = episode_directory / "physics"
            actor_output.mkdir()
            if camera_enabled:
                (actor_output / "camera-output").symlink_to(
                    os.path.relpath(layout.camera_directory(index), actor_output),
                    target_is_directory=True,
                )
            controller_directory = output / f"robot_{index}"
            controller_configuration = json.loads(
                (controller_directory / "controller-configuration.json").read_text()
            )
            for name in (
                "controller-configuration.json",
                "controller-process.json",
                "controller.log",
                "sdk-worker-process.json",
                "sdk-worker.log",
                "native-state-logs",
                "policy-input.csv",
                "target-motion.csv",
            ):
                (actor_output / name).symlink_to(
                    os.path.relpath(controller_directory / name, actor_output)
                )
            atomic_json(actor_output / "physics-initialization.json", initialization_record)
            atomic_json(
                actor_output / "neutral-targets.json",
                {
                    "source": "Canonical URDF FK of the frozen released initial pose",
                    "positions_m": neutral_samples[index].vr_position_body_frame,
                    "orientations_wxyz": neutral_samples[index].vr_orientation_body_frame_wxyz,
                },
            )
            atomic_json(
                actor_output / "configuration.json",
                {
                    "environment_index": index,
                    "environment_origin_m": origins[index].tolist(),
                    "trace_coordinates": "Episode-local translation; unchanged world orientation",
                    "shared_owner": os.path.relpath(output, actor_output),
                    "episode_id": state.episode_id,
                    "physics_dt": configuration.dt,
                    "controller_command": controller_configuration["arguments"],
                    "executable_sha256": EXECUTABLE_SHA256,
                    "robot_joint_names": robot.joint_names,
                    "motor_joint_names": list(JOINT_NAMES),
                    "motor_order_indices": motor_indices,
                    "upstream_environment": "Isaac-Velocity-Flat-G1 / G1FlatEnvCfg",
                    "canonical_robot_urdf_sha256": CANONICAL_URDF_SHA256,
                    "sonic_robot_configuration_sha256": SONIC_CONFIGURATION_SHA256,
                    "imu_backend": type(scene["imu_pelvis"]).__module__,
                    "simulation_validity_policy": clock_validity(0),
                    "physics_device": str(simulation.device),
                    "physics_backend": simulation.physics_manager.__name__,
                    "actuation": "Native SDK q/dq/tau/kp/kd through Isaac implicit actuator targets; source effort/velocity limits retained",
                    "sensors": "29 joint encoders/velocity/acceleration/estimated torque plus upstream Newton pelvis and torso IMU at canonical URDF mounts; no odometry publisher",
                    "initial_observation": "Frozen release-state pose; RGB-D rendered only when camera workload is enabled",
                    "command_source": command_source.value,
                    "camera_enabled": camera_enabled,
                    "diagnostic_motion": arguments.diagnostic_motion if scripted_episodes else None,
                    "diagnostic_phase_seconds": arguments.diagnostic_phase_seconds
                    if scripted_episodes
                    else None,
                    "startup": "Released neutral pose on the floor; physics starts after native CONTROL and VR readiness, with no elastic band or external support",
                    "task_scene_sha256": hashlib.sha256(
                        arguments.task_scene.read_bytes()
                    ).hexdigest(),
                    "command_protocol": episode_command_protocol(arguments),
                    "episode_wall_seconds": arguments.wall_seconds,
                    "model_requests": 0,
                },
            )
            actors.append(
                RobotEpisode(
                    episode_directory,
                    state,
                    neutral_samples[index],
                    origins[index].tolist(),
                    shutdown,
                    observation_history_enabled=arguments.external_camera and camera_enabled,
                )
            )
        evaluations = [actor.evaluation for actor in actors]
        scripted_sources = (
            [
                TurnCheckControl(neutral)
                if arguments.diagnostic_motion == "turn-check"
                else StandingControlSource(neutral, arguments.diagnostic_phase_seconds)
                if arguments.diagnostic_motion == "standing"
                else ShutdownReferenceControl(
                    neutral,
                    shutdown,
                    list(scene_definition.robot_start_xy_m),
                    storage_occluders(scene_definition),
                )
                for neutral in neutral_samples
            ]
            if scripted_episodes
            else []
        )
        evidence_io = EpisodeIO(output, output / "unused-model-commands")
        dt = configuration.dt
        tick = 0
        release_at = None
        startup_deadline = time.monotonic() + 900
        next_sensor = 0.0
        next_planner = 0.0
        next_actor_poll = 0.0
        next_trace = 0.0
        next_capture = arguments.capture_interval_seconds if scripted_episodes else 2.0
        capture_number = 0
        pending_captures: set[str] = set()
        completed_captures = 0
        maximum_lag = 0.0
        maximum_command_ages = [0.0] * layout.num_environments
        work_duration = DurationSummary(dt)
        stage_durations = {
            name: DurationSummary(dt)
            for name in (
                "control_and_sensors",
                "actuation_targets",
                "physics_and_scene_update",
                "evaluation_trace_capture",
                "reference_sampling",
                "reference_publication",
            )
        }
        motor_targets = BatchedMotorTargets(robot, motor_indices, layout.num_environments)
        active_target_updates = 0
        sensor_packing_validated = False
        while simulation.is_running():
            started = time.monotonic()
            tick_events = []
            if camera is not None and camera.process.poll() is not None:
                raise RuntimeError("Camera worker exited during physics; inspect camera-worker.log")
            elapsed = started - release_at if release_at is not None else 0.0
            if release_at is None and started >= startup_deadline:
                raise TimeoutError("Independent native controllers did not become ready")
            for event in evidence_io.events():
                if isinstance(event, ObservationReady):
                    if event.identifier not in pending_captures:
                        raise ValueError("Camera completion belongs to another capture batch")
                    pending_captures.remove(event.identifier)
                    completed_captures += 1
            if release_at is None:
                # Controller setup can write targets while the physics tick is frozen.
                snapshots.invalidate()
            sensor_due = started >= next_sensor
            planner_due = started >= next_planner
            actor_poll_due = started >= next_actor_poll or elapsed >= arguments.wall_seconds
            control_fields = {}
            if release_at is None or sensor_due or planner_due:
                control_fields["root_states"] = robot.data.root_state_w.torch
            if actor_poll_due and hand_indices:
                control_fields["joint_positions"] = robot.data.joint_pos.torch
            if sensor_due:
                control_fields.update(
                    {
                        "torso_state": robot.data.body_link_state_w.torch[:, torso_index],
                        "joint_positions": robot.data.joint_pos.torch,
                        "joint_velocities": robot.data.joint_vel.torch,
                        "joint_accelerations": robot.data.joint_acc.torch,
                        "joint_torques": robot.data.applied_torque.torch,
                        **{
                            name + "_gyro": scene["imu_" + name].data.ang_vel_b.torch
                            for name in ("pelvis", "torso")
                        },
                        **{
                            name + "_acceleration": scene["imu_" + name].data.lin_acc_b.torch
                            for name in ("pelvis", "torso")
                        },
                    }
                )
            control_snapshot = snapshots.read(tick, control_fields)
            recorded_roots = [
                recorded_state_from_newton(root) for root in control_snapshot.get("root_states", ())
            ]
            if sensor_due:
                sensor_fields = control_fields
                sensor_snapshot = control_snapshot
                if not sensor_packing_validated:
                    if any(
                        not np.array_equal(tensor.cpu().numpy(), sensor_snapshot[name])
                        for name, tensor in sensor_fields.items()
                    ):
                        raise ValueError("Packed sensor snapshot differs from original field reads")
                    atomic_json(
                        output / "sensor-packing-check.json",
                        {
                            "passed": True,
                            "fields": list(sensor_fields),
                            "phase": "pre-release warmup",
                            "model_requests": 0,
                        },
                    )
                    sensor_packing_validated = True
                torso_states = sensor_snapshot["torso_state"]
                joint_positions = sensor_snapshot["joint_positions"][:, motor_indices]
                joint_velocities = sensor_snapshot["joint_velocities"][:, motor_indices]
                joint_accelerations = sensor_snapshot["joint_accelerations"][:, motor_indices]
                torques = sensor_snapshot["joint_torques"][:, motor_indices]
                imu = {
                    name: (
                        sensor_snapshot[name + "_gyro"],
                        sensor_snapshot[name + "_acceleration"],
                    )
                    for name in ("pelvis", "torso")
                }
                for index, bridge in enumerate(bridges):
                    root = recorded_roots[index]
                    torso = recorded_state_from_newton(torso_states[index])
                    bridge.publish(
                        SonicEncoderImuState(
                            simulation_seconds=tick * dt,
                            **{
                                f"{side}_hand": HandEncoderState(
                                    positions=sensor_snapshot["joint_positions"][
                                        index, indices
                                    ].tolist(),
                                    velocities=sensor_snapshot["joint_velocities"][
                                        index, indices
                                    ].tolist(),
                                    accelerations=sensor_snapshot["joint_accelerations"][
                                        index, indices
                                    ].tolist(),
                                    measured_torques=sensor_snapshot["joint_torques"][
                                        index, indices
                                    ].tolist(),
                                )
                                for side, indices in hand_indices.items()
                            },
                            joint_positions=joint_positions[index].tolist(),
                            joint_velocities=joint_velocities[index].tolist(),
                            joint_accelerations=joint_accelerations[index].tolist(),
                            measured_joint_torques=torques[index].tolist(),
                            pelvis_quaternion_wxyz=root[3:7].tolist(),
                            pelvis_gyroscope_body_rad_s=imu["pelvis"][0][index].tolist(),
                            pelvis_accelerometer_body_m_s2=imu["pelvis"][1][index].tolist(),
                            torso_quaternion_wxyz=torso[3:7].tolist(),
                            torso_gyroscope_body_rad_s=imu["torso"][0][index].tolist(),
                            torso_accelerometer_body_m_s2=imu["torso"][1][index].tolist(),
                        )
                    )
                next_sensor = started + 0.01
            sensors_finished = time.monotonic()
            receipts = [bridge.latest_command() for bridge in bridges]
            for index, bridge in enumerate(bridges):
                for publication in bridge.take_planner_publications():
                    actors[index].record_planner_publication(publication)
            if (
                release_at is None
                and all(bridge.controller_ready for bridge in bridges)
                and all(receipt is not None for receipt in receipts)
            ):
                # The first step after the warmup idle costs ~0.1 s; pay it before time
                # zero, then restore the frozen initial state checked below.
                release_warm_step_started = time.monotonic()
                scene.write_data_to_sim()
                simulation.step(render=False)
                scene.update(dt)
                contact_capacity.attach()
                contact_capacity.sample()
                restore_initial_state()
                snapshots.invalidate()
                recorded_roots = [
                    recorded_state_from_newton(root)
                    for root in robot.data.root_state_w.torch.cpu().numpy()
                ]
                atomic_json(
                    output / "release-warm-step.json",
                    {
                        "duration_seconds": time.monotonic() - release_warm_step_started,
                        "initial_state_restored": True,
                        "purpose": "Absorb the first-step cost after the warmup idle before the episode clock starts",
                        "model_requests": 0,
                    },
                )
                for index, state in enumerate(initial.states):
                    if not np.allclose(
                        recorded_roots[index],
                        state.root_state_world,
                        atol=1e-6,
                        rtol=0,
                    ) or not np.allclose(
                        robot.data.joint_pos.torch[index].cpu().numpy(),
                        state.joint_positions,
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError(
                            "A frozen initial observation differs from the shared release state"
                        )
                # All camera tiles and native controllers are warm before time zero.
                release_at = time.monotonic()
                elapsed = 0.0
                for actor in actors:
                    actor.start(release_at, capture_initial_observation=camera_enabled)
                evidence_io.submit(
                    AppendRecord(
                        "physical-trace.jsonl",
                        {
                            "event": "shared_release",
                            "release_monotonic_seconds": release_at,
                            "episode_ids": episode_ids,
                        },
                    )
                )
            if release_at is not None and actor_poll_due:
                next_actor_poll = started + 0.02
                finish_timing_summary = None
                for index, actor in enumerate(actors):
                    if index in finished_actors:
                        continue
                    actor.poll(
                        elapsed,
                        partial(measure_robot, index),
                        accept_model_commands=not scripted_episodes,
                        hand_initials=HandTransitionInitials(
                            **{
                                side: control_snapshot["joint_positions"][index, indices].tolist()
                                for side, indices in hand_indices.items()
                            }
                        )
                        if hand_indices
                        else None,
                    )
                    if actor.finish_reason is not None:
                        if finish_timing_summary is None:
                            finish_timing_summary = episode_timing.summary()
                        reference = scripted_sources[index] if scripted_episodes else None
                        if isinstance(reference, (ShutdownReferenceControl, TurnCheckControl)):
                            actor.io.submit(
                                WriteRecord(
                                    actor.output / "reference-progress.json",
                                    reference.summary(),
                                    durable=True,
                                )
                            )
                        actor.finish(elapsed, tick, timing_summary=finish_timing_summary)
                        finished_actors.add(index)
                if len(finished_actors) == len(actors):
                    break
            if planner_due:
                for index, bridge in enumerate(bridges):
                    reference_started = time.monotonic()
                    sample = neutral_samples[index]
                    command_id = None
                    if release_at is not None:
                        match command_source:
                            case PhysicsCommandSource.MODEL:
                                actor = actors[index]
                                sample = actor.planner_sample(
                                    elapsed, recorded_roots[index][3:7].tolist()
                                )
                                if actor.stream.active_timeline is not None:
                                    command_id = actor.stream.active_timeline.publication.identifier
                            case PhysicsCommandSource.SCRIPTED:
                                scripted = scripted_sources[index]
                                scripted_elapsed = (
                                    scripted.duration_seconds
                                    if index in finished_actors
                                    else elapsed
                                )
                                if isinstance(
                                    scripted, (ShutdownReferenceControl, TurnCheckControl)
                                ):
                                    local_root = recorded_roots[index].copy()
                                    local_root[:3] -= origins[index]
                                    ready_snapshot = TaskMeasurementSnapshot(
                                        **snapshots.read(
                                            tick,
                                            task_measurement_tensor_fields(robot),
                                        )
                                    )
                                    hand_points = []
                                    for side, offset in zip(
                                        ("left", "right"), POINT_OFFSETS[:2], strict=True
                                    ):
                                        wrist = robot.body_names.index(f"{side}_wrist_yaw_link")
                                        wrist_rotation = Rotation.from_quat(
                                            ready_snapshot.body_quaternions_xyzw[index, wrist]
                                        )
                                        hand_points.append(
                                            (
                                                ready_snapshot.body_positions[index, wrist]
                                                + wrist_rotation.apply(offset)
                                                - origins[index]
                                            ).tolist()
                                        )
                                    sample = scripted.sample(
                                        scripted_elapsed, local_root.tolist(), hand_points
                                    )
                                else:
                                    sample = scripted.sample(scripted_elapsed)
                    reference_sampled = time.monotonic()
                    stage_durations["reference_sampling"].observe(
                        reference_sampled - reference_started
                    )
                    bridge.publish_planner(sample, command_id=command_id)
                    stage_durations["reference_publication"].observe(
                        time.monotonic() - reference_sampled
                    )
                next_planner = started + 1 / 30
            actuation_started = time.monotonic()
            if all(receipt is not None for receipt in receipts):
                commands = []
                for index, receipt in enumerate(receipts):
                    if receipt is None:
                        raise RuntimeError("Missing native motor receipt")
                    age = max(0.0, started - receipt.received_monotonic_seconds)
                    maximum_command_ages[index] = max(maximum_command_ages[index], age)
                    if release_at is not None:
                        actors[index].observe_motor_command(receipt.command, age)
                    if release_at is not None and age > 1:
                        raise TimeoutError("Native controller motor stream expired")
                    commands.append(receipt.command)

                if motor_targets.apply(commands):
                    active_target_updates += release_at is not None
            actuation_finished = time.monotonic()
            if release_at is None:
                time.sleep(0.005)
                continue
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(dt)
            physics_finished = time.monotonic()
            tick += 1
            trace_due = elapsed >= next_trace
            if tick % 20 == 0:
                contact_capacity.sample()
            capture_requested = (
                any(
                    actor.requested_capture is not None
                    and actor.current_capture_ready()
                    and index not in finished_actors
                    for index, actor in enumerate(actors)
                )
                if not scripted_episodes
                else elapsed >= next_capture
            )
            history_requests = [
                actor.due_history_frame(elapsed) if index not in finished_actors else None
                for index, actor in enumerate(actors)
            ]
            capture_due = camera_enabled and (capture_requested or any(history_requests))
            if capture_due and len(pending_captures) >= 6:
                raise RuntimeError("Camera history queue exceeded six outstanding batches")
            post_step_fields = {}
            if tick % 4 == 0:
                post_step_fields.update(
                    task_measurement_tensor_fields(robot, include_joint_positions=True)
                )
                post_step_fields.update(shutdown_tensor_fields(scene, shutdown))
            if trace_due or capture_due:
                post_step_fields.update(visual_tensor_fields())
            if tick % 4 == 0 or trace_due:
                post_step_fields.update(task_contact_tensor_fields(contact_environment))
            snapshots.read(tick, post_step_fields)
            if tick % 4 == 0:
                for index, evaluation in enumerate(evaluations):
                    if index in finished_actors:
                        continue
                    measured = measure_robot(index)
                    evaluation.update(measured)
                    if scripted_episodes:
                        reference = scripted_sources[index]
                        if isinstance(
                            reference,
                            (
                                ShutdownReferenceControl,
                                TurnCheckControl,
                            ),
                        ):
                            reference.latest_measurement = measured
                    max_displacements[index] = max(
                        max_displacements[index], measured.root_distance_from_start_m
                    )
            if trace_due:
                tick_events.append("trace_snapshot_enqueued")
                trace = snapshot_batch(f"trace_{tick:06d}")
                for index, actor in enumerate(actors):
                    if index in finished_actors:
                        continue
                    receipt = receipts[index]
                    actor.record_trace(
                        trace.states[index],
                        contact_for_robot(index),
                        receipt.command if receipt is not None else None,
                        phase=scripted_sources[index].phase(elapsed)
                        if scripted_episodes
                        else "native-task",
                    )
                next_trace = elapsed + 0.05
            if capture_due:
                tick_events.append("camera_snapshot_enqueued")
                capture_number += 1
                batch = snapshot_batch(f"capture_{capture_number:03d}")
                observation_replies: list[Path | None] = [None] * layout.num_environments
                for index, actor in enumerate(actors):
                    if scripted_episodes and index not in finished_actors:
                        actor.requested_capture = batch.states[index].identifier
                    history_context = history_requests[index]
                    if history_context is not None:
                        batch.states[index] = actor.begin_history_capture(
                            batch.states[index], history_context, started
                        )
                        observation_replies[index] = actor.output / "camera-events"
                        continue
                    requested = actor.requested_capture
                    if (
                        requested is not None
                        and actor.current_capture_ready()
                        and index not in finished_actors
                    ):
                        batch.states[index] = batch.states[index].model_copy(
                            update={"identifier": requested}
                        )
                        batch.states[index] = actor.begin_capture(batch.states[index], started)
                        observation_replies[index] = actor.output / "camera-events"
                evidence_io.submit(
                    CaptureBatchSnapshot(
                        batch.identifier,
                        batch.model_dump(mode="json"),
                        time.monotonic(),
                        tuple(observation_replies),
                    )
                )
                pending_captures.add(batch.identifier)
                next_capture = elapsed + (
                    arguments.capture_interval_seconds if scripted_episodes else 2
                )
            ended = time.monotonic()
            work_duration.observe(ended - started)
            for name, duration in (
                ("control_and_sensors", actuation_started - started),
                ("actuation_targets", actuation_finished - actuation_started),
                ("physics_and_scene_update", physics_finished - actuation_finished),
                ("evaluation_trace_capture", ended - physics_finished),
            ):
                stage_durations[name].observe(duration)
            lag = ended - release_at - tick * dt
            episode_timing.observe_tick(
                {
                    "sensors": sensors_finished - started,
                    "actuation": actuation_finished - actuation_started,
                    "physics_and_update": physics_finished - actuation_finished,
                    "tick_before_trace_and_sleep": physics_finished - started,
                    "evaluation_and_trace": ended - physics_finished,
                    "tick_work_before_sleep": ended - started,
                },
                elapsed_wall_seconds=started - release_at,
                simulation_seconds=tick * dt,
                lag_seconds=lag,
                lag_increase_seconds=lag - previous_lag,
                events=tick_events,
                command_id=None,
            )
            previous_lag = lag
            maximum_lag = max(maximum_lag, lag)
            for index, actor in enumerate(actors):
                if index not in finished_actors:
                    actor.maximum_lag = max(actor.maximum_lag, lag)
            remaining = release_at + tick * dt - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        if release_at is None:
            raise RuntimeError("Simulation stopped before controller readiness")
        if len(finished_actors) != len(actors):
            raise RuntimeError("Simulation stopped before all episodes reached a terminal state")
        # End the native streams before waiting for offline camera completion.
        # Otherwise loss-intolerant send receipts could fill an undrained socket.
        for bridge in bridges:
            bridge.request_stop()
        for index, bridge in enumerate(bridges):
            bridge.close()
            for publication in bridge.take_planner_publications():
                actors[index].record_planner_publication(publication)
        drain_deadline = time.monotonic() + 30
        while pending_captures:
            for event in evidence_io.events():
                if isinstance(event, ObservationReady) and event.identifier in pending_captures:
                    pending_captures.remove(event.identifier)
                    completed_captures += 1
            if time.monotonic() >= drain_deadline:
                raise TimeoutError("Final shared camera capture did not drain")
            time.sleep(0.01)
        evidence_io.close()
        evidence_io = None
        for actor in actors:
            actor.io.close()
        result.update(
            status="completed",
            num_environments=layout.num_environments,
            physics_steps=tick,
            wall_seconds=arguments.wall_seconds,
            simulation_validity=clock_validity(maximum_lag),
            tick_work=work_duration.summary(),
            host_stage_timings={
                name: duration.summary() for name, duration in stage_durations.items()
            },
            timing_scope="Host elapsed stages without explicit CUDA synchronization; transfers can include earlier queued GPU work",
            batched_native_target_updates=active_target_updates,
            maximum_root_displacements_m=max_displacements,
            maximum_motor_command_ages_seconds=maximum_command_ages,
            completed_shared_captures=completed_captures,
            physical_criteria=[evaluation.physical_criterion_met for evaluation in evaluations],
            activated=[evaluation.activated for evaluation in evaluations],
            independent_controller_pids=[
                json.loads((output / f"robot_{index}/controller-process.json").read_text())["pid"]
                for index in range(layout.num_environments)
            ],
            architecture="One production Newton physics scene; optional camera worker; one isolated native SONIC instance per environment",
            episode_directories=[str(actor.directory) for actor in actors],
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        try:
            atomic_json(output / "timing.json", episode_timing.summary())
            atomic_json(output / "contact-capacity.json", contact_capacity.summary())
        except Exception as error:
            cleanup_errors.append(f"Timing evidence: {error}")
        for actor in actors:
            try:
                actor.io.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        for bridge in bridges:
            try:
                bridge.request_stop()
            except Exception as error:
                cleanup_errors.append(str(error))
        for bridge in bridges:
            try:
                bridge.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        if evidence_io is not None:
            try:
                evidence_io.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        if camera is not None:
            try:
                camera.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        if runtime_started and runtime is not None:
            try:
                runtime.__exit__(None, None, None)
            except Exception as error:
                cleanup_errors.append(str(error))
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["status"] = "infrastructure_error"
        atomic_json(output / "result.json", result, durable=True)
