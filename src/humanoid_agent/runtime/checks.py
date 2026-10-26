"""Physical preflight for the same environment factory used by task execution."""

from __future__ import annotations

import hashlib
import importlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import BODY_NAMES, POINT_OFFSETS
from humanoid_agent.isaac_workcell import TASK_CONTACT_PAIRS, GroundWorkcellScene
from humanoid_agent.robot_camera import calibration
from humanoid_agent.robot_observations import RobotSensors
from humanoid_agent.runtime.environment import (
    create_newton_simulation_configuration,
    create_newton_workcell_scene,
    launch_newton_runtime,
    register_newton_scene_schemas,
)
from humanoid_agent.sensor_feedback import EncoderKinematics


def check_workcell_scene(asset_root: Path, scene_path: Path, output: Path) -> None:
    """Require finite CUDA physics, upright support and both foot-contact signals."""
    output.mkdir(parents=True, exist_ok=False)
    register_newton_scene_schemas()
    simulation_module: Any = importlib.import_module("isaaclab.sim")
    configuration = create_newton_simulation_configuration()
    with launch_newton_runtime(configuration):
        simulation = simulation_module.SimulationContext(configuration)
        scene = create_newton_workcell_scene(
            asset_root,
            GroundWorkcellScene.model_validate_json(scene_path.read_text()),
            contact_sensors=True,
            display_collision_boxes=True,
        )
        robot = scene["robot"]
        simulation.reset()
        scene.reset()
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        robot.actuators.target_command.set_position_index(
            value=robot.data.default_joint_pos.torch.clone()
        )
        encoder_parity = check_encoder_pose_parity(robot, simulation, asset_root, configuration.dt)
        freefall_imu = check_freefall_imu(scene, simulation, configuration.dt)
        maximum_forces = {name: 0.0 for name, _, _ in TASK_CONTACT_PAIRS}
        for _ in range(100):
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(configuration.dt)
            if not robot.data.joint_pos.torch.isfinite().all():
                raise ValueError("Nonfinite joint state in scene preflight")
            for sensor_name, _, _ in TASK_CONTACT_PAIRS:
                forces = (
                    scene["task_contact_" + sensor_name].data.force_matrix_w.torch[0].cpu().numpy()
                )
                if not np.isfinite(forces).all():
                    raise ValueError("Nonfinite task contact signal: " + sensor_name)
                maximum_forces[sensor_name] = max(
                    maximum_forces[sensor_name], float(np.linalg.norm(forces, axis=-1).max())
                )
        root_pose = robot.data.root_link_pose_w.torch[0].cpu().tolist()
        device = str(robot.data.joint_pos.torch.device)
        passed = (
            device == "cuda:0"
            and root_pose[2] > 0.5
            and all(maximum_forces[name] > 1 for name in ("left_foot_floor", "right_foot_floor"))
        )
        atomic_json(
            output / "result.json",
            {
                "passed": passed,
                "encoder_parity": encoder_parity,
                "freefall_imu": freefall_imu,
                "imu_backend": type(scene["imu_pelvis"]).__module__,
                "imu_readings": {
                    name: {
                        "linear_acceleration_body_m_s2": scene["imu_" + name]
                        .data.lin_acc_b.torch[0]
                        .cpu()
                        .tolist(),
                        "angular_velocity_body_rad_s": scene["imu_" + name]
                        .data.ang_vel_b.torch[0]
                        .cpu()
                        .tolist(),
                    }
                    for name in ("pelvis", "torso")
                },
                "upstream_configuration": "Isaac-Velocity-Flat-G1 / G1FlatEnvCfg",
                "physics_backend": simulation.physics_manager.__name__,
                "device": device,
                "physics_steps": simulation.get_physics_step_count(),
                "root_pose_xyzw": root_pose,
                "max_contact_force_n": maximum_forces,
                "model_requests": 0,
                "scope": "Scene/contact preflight; not a learned-controller or task-success result",
            },
        )
        if not passed:
            raise RuntimeError("Scene preflight failed; inspect result.json")


def check_encoder_pose_parity(
    robot: Any, simulation: Any, asset_root: Path, time_step: float
) -> dict[str, Any]:
    """Compare independent Newton and MuJoCo FK without using task geometry."""
    os.environ["SONIC_ROBOT_URDF"] = str(asset_root / "robot_description/urdf/g1/main.urdf")
    encoder = EncoderKinematics()
    torch: Any = importlib.import_module("torch")
    default_pose = robot.data.default_root_pose.torch.clone()
    default_joints = robot.data.default_joint_pos.torch.clone()
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    cases = []
    for index, angles in enumerate(
        ((0, 0, 0), (0.15, -0.2, 1.2), (-0.2, 0.1, -2.0), (0.05, 0.25, 2.7))
    ):
        pose = default_pose.clone()
        root_rotation = Rotation.from_euler("xyz", angles)
        pose[0, :3] = torch.tensor([0.3 * index, -0.2 * index, 0.9], device=pose.device)
        pose[0, 3:7] = torch.tensor(root_rotation.as_quat(), device=pose.device)
        joints = default_joints.clone()
        perturbation = torch.tensor(
            0.08 * np.sin(index + np.arange(joints.shape[1])), device=joints.device
        )
        joints += perturbation
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_joint_position_to_sim_index(position=joints)
        simulation.forward()
        robot.update(time_step)
        sensors = RobotSensors(
            joint_names=robot.joint_names,
            joint_positions_rad=joints[0].cpu().tolist(),
            joint_velocities_rad_per_s=[0.0] * 29,
            imu_angular_velocity_body_rad_per_s=[0, 0, 0],
            imu_gravity_direction_body=root_rotation.inv().apply([0, 0, -1]).tolist(),
        )
        measured = encoder.measure(sensors)
        heading = Rotation.from_euler("z", angles[2])
        root_position = pose[0, :3].cpu().numpy()
        positions = robot.data.body_link_pos_w.torch[0].cpu().numpy()
        rotations = robot.data.body_link_quat_w.torch[0].cpu().numpy()
        expected = []
        for body_name, offset in zip(BODY_NAMES, POINT_OFFSETS, strict=True):
            body_index = robot.body_names.index(body_name)
            body_rotation = Rotation.from_quat(rotations[body_index])
            expected_orientation = heading.as_matrix().T @ body_rotation.as_matrix()
            name = ("left_wrist", "right_wrist", "head")[BODY_NAMES.index(body_name)]
            measured_orientation = Rotation.from_quat(
                measured["tracking_orientations_wxyz"][name], scalar_first=True
            ).as_matrix()
            maximum_rotation_error = max(
                maximum_rotation_error,
                float(
                    Rotation.from_matrix(expected_orientation.T @ measured_orientation).magnitude()
                ),
            )
            expected.append(
                heading.inv().apply(
                    positions[body_index] + body_rotation.apply(offset) - root_position
                )
            )
        actual = np.asarray(list(measured["tracking_points_m"].values()))
        position_error = float(np.max(np.linalg.norm(actual - expected, axis=1)))
        torso_index = robot.body_names.index("torso_link")
        camera_calibration = calibration()
        expected_camera = heading.inv().apply(
            positions[torso_index]
            + Rotation.from_quat(rotations[torso_index]).apply(
                camera_calibration["camera_position_torso_m"]
            )
            - root_position
        )
        position_error = max(
            position_error,
            float(np.linalg.norm(expected_camera - measured["camera_optical_origin_m"])),
        )
        expected_optical = (
            heading.as_matrix().T
            @ Rotation.from_quat(rotations[torso_index]).as_matrix()
            @ np.asarray(camera_calibration["optical_to_torso_rotation"])
        )
        rotation_error = float(
            Rotation.from_matrix(
                expected_optical.T @ np.asarray(measured["camera_optical_to_heading_rotation"])
            ).magnitude()
        )
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_rotation_error = max(maximum_rotation_error, rotation_error)
        cases.append(
            {
                "root_pose_xyzw": pose[0].cpu().tolist(),
                "joint_positions": joints[0].cpu().tolist(),
                "position_error_m": position_error,
                "camera_rotation_error_rad": rotation_error,
            }
        )
    robot.write_root_pose_to_sim_index(root_pose=default_pose)
    robot.write_joint_position_to_sim_index(position=default_joints)
    simulation.forward()
    robot.update(time_step)
    result = {
        "cases": cases,
        "maximum_position_error_m": maximum_position_error,
        "maximum_rotation_error_rad": maximum_rotation_error,
        "position_tolerance_m": 1e-5,
        "rotation_tolerance_rad": 1e-5,
    }
    if maximum_position_error > 1e-5 or maximum_rotation_error > 1e-5:
        raise ValueError(f"Independent encoder FK parity failed: {result}")
    return result


def check_freefall_imu(scene: Any, simulation: Any, time_step: float) -> dict[str, Any]:
    """A freely falling, nonrotating robot should measure zero proper acceleration."""
    robot = scene["robot"]
    pose = robot.data.default_root_pose.torch.clone()
    pose[0, 2] = 3.0
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    scene.reset()
    readings = []
    for index in range(10):
        scene.write_data_to_sim()
        simulation.step(render=False)
        scene.update(time_step)
        if index >= 2:
            readings.append(
                {
                    name: {
                        "acceleration": scene["imu_" + name].data.lin_acc_b.torch[0].cpu().tolist(),
                        "gyroscope": scene["imu_" + name].data.ang_vel_b.torch[0].cpu().tolist(),
                    }
                    for name in ("pelvis", "torso")
                }
            )
    maximum_acceleration = max(
        float(np.linalg.norm(row[name]["acceleration"]))
        for row in readings
        for name in ("pelvis", "torso")
    )
    maximum_gyroscope = max(
        float(np.linalg.norm(row[name]["gyroscope"]))
        for row in readings
        for name in ("pelvis", "torso")
    )
    result = {
        "maximum_acceleration_m_s2": maximum_acceleration,
        "maximum_gyroscope_rad_s": maximum_gyroscope,
        "acceleration_tolerance_m_s2": 0.1,
        "gyroscope_tolerance_rad_s": 0.01,
        "readings": readings,
    }
    if maximum_acceleration > 0.1 or maximum_gyroscope > 0.01:
        raise ValueError(f"Upstream IMU proper acceleration check failed: {result}")
    scene.reset()
    robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    simulation.forward()
    robot.update(time_step)
    return result


def check_shared_workcell_scene(
    asset_root: Path, scene_path: Path, output: Path, num_environments: int
) -> None:
    """Exercise upstream cloned G1 states, sensors and batched actuator targets."""
    from humanoid_agent.runtime.camera import NativeVisualState
    from humanoid_agent.runtime.state import recorded_state_from_newton

    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"passed": False, "model_requests": 0}
    try:
        register_newton_scene_schemas()
        simulation_module: Any = importlib.import_module("isaaclab.sim")
        torch: Any = importlib.import_module("torch")
        configuration = create_newton_simulation_configuration()
        with launch_newton_runtime(configuration):
            simulation = simulation_module.SimulationContext(configuration)
            scene = create_newton_workcell_scene(
                asset_root,
                GroundWorkcellScene.model_validate_json(scene_path.read_text()),
                contact_sensors=True,
                display_collision_boxes=True,
                num_environments=num_environments,
            )
            robot = scene["robot"]
            simulation.reset()
            scene.reset()
            # Follow upstream create_scene.py: root defaults are local to each
            # clone; add env_origins once when writing world-frame root poses.
            root_poses = robot.data.default_root_pose.torch.clone()
            root_poses[:, :3] += scene.env_origins
            joints = robot.data.default_joint_pos.torch.clone()
            shoulder_index = robot.joint_names.index("left_shoulder_pitch_joint")
            joints[:, shoulder_index] += torch.arange(num_environments, device="cuda:0") * 0.1
            robot.write_root_pose_to_sim_index(root_pose=root_poses)
            robot.write_root_velocity_to_sim_index(
                root_velocity=robot.data.default_root_vel.torch.clone()
            )
            robot.write_joint_position_to_sim_index(position=joints)
            robot.write_joint_velocity_to_sim_index(
                velocity=robot.data.default_joint_vel.torch.clone()
            )
            robot.actuators.target_command.set_position_index(value=joints)
            simulation.forward()
            robot.update(configuration.dt)
            origins = scene.env_origins.cpu().numpy()
            measured_joints = robot.data.joint_pos.torch.cpu().numpy()
            if not np.allclose(measured_joints, joints.cpu().numpy(), atol=1e-6, rtol=0):
                raise ValueError("Cloned joint states do not match their independent targets")
            measured_roots = robot.data.root_link_pos_w.torch.cpu().numpy()
            if not np.allclose(
                measured_roots - origins,
                root_poses[:, :3].cpu().numpy() - origins,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("Cloned root states do not match environment origins")
            snapshots = [
                NativeVisualState(
                    identifier="initial",
                    episode_id=f"shared_scene_{environment_index}",
                    episode_deadline_elapsed_wall_seconds=30,
                    simulation_seconds=0,
                    elapsed_wall_seconds=0,
                    source_monotonic_seconds=time.monotonic(),
                    joint_names=robot.joint_names,
                    joint_positions=measured_joints[environment_index].tolist(),
                    joint_velocities=robot.data.joint_vel.torch[environment_index].cpu().tolist(),
                    root_state_world=recorded_state_from_newton(
                        robot.data.root_state_w.torch[environment_index].cpu().numpy()
                    ).tolist(),
                )
                for environment_index in range(num_environments)
            ]
            atomic_json(
                output / "snapshots.json",
                {
                    "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                    "environment_origins_m": origins.tolist(),
                    "states": [state.model_dump(mode="json") for state in snapshots],
                },
            )
            contacts: dict[str, Any] = {}
            for _ in range(100):
                scene.write_data_to_sim()
                simulation.step(render=False)
                scene.update(configuration.dt)
                for sensor_name, _, _ in TASK_CONTACT_PAIRS:
                    data = scene["task_contact_" + sensor_name].data
                    forces = data.force_matrix_w.torch.cpu().numpy()
                    if forces.shape != (num_environments, 1, 1, 3) or not np.isfinite(forces).all():
                        raise ValueError(
                            f"Unexpected batched filtered contact data for {sensor_name}: {forces.shape}"
                        )
                    peak = np.linalg.norm(forces[:, 0, 0], axis=1)
                    contacts[sensor_name] = np.maximum(
                        contacts.get(sensor_name, np.zeros(num_environments)), peak
                    ).tolist()
            roots = robot.data.root_link_pos_w.torch.cpu().numpy()
            if not np.isfinite(robot.data.joint_pos.torch.cpu().numpy()).all() or not np.all(
                roots[:, 2] > 0.5
            ):
                raise ValueError(
                    "Cloned G1 states did not remain finite and supported during scene check"
                )
            for name in ("left_foot_floor", "right_foot_floor"):
                if not np.all(np.asarray(contacts[name]) > 1):
                    raise ValueError("A cloned G1 is missing its own foot support signal")
            result.update(
                passed=True,
                num_environments=num_environments,
                device=str(robot.data.joint_pos.torch.device),
                environment_origins_m=origins.tolist(),
                root_positions_world_m=roots.tolist(),
                contact_peaks_n=contacts,
                physics_steps=simulation.get_physics_step_count(),
                dt_seconds=configuration.dt,
                scope="Batched scene, independent state writes and per-environment contact preflight. No SONIC control, model task result or wall-clock throughput claim.",
            )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(output / "result.json", result, durable=True)


def check_shared_camera_batch(
    asset_root: Path,
    scene_path: Path,
    output: Path,
    num_environments: int,
    decorative_assets: Path | None,
) -> None:
    """Use upstream tiled cameras to test image/pose separation without physics."""
    from humanoid_agent.runtime.camera import NativeCameraRenderer, NativeVisualState

    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"passed": False, "model_requests": 0}
    renderer = None
    try:
        register_newton_scene_schemas(rendering=True)
        configuration = create_newton_simulation_configuration()
        with launch_newton_runtime(configuration):
            renderer = NativeCameraRenderer(
                asset_root.parents[2],
                scene_path,
                decorative_assets,
                output / "camera",
                simulation_configuration=configuration,
                num_environments=num_environments,
            )
            robot = renderer.robot
            origins = renderer.scene.env_origins.cpu().numpy()
            default_roots = robot.data.default_root_pose.torch.cpu().numpy()
            default_roots[:, :3] += origins
            joints = robot.data.default_joint_pos.torch.cpu().numpy()
            states = [
                NativeVisualState(
                    identifier="baseline",
                    episode_id=f"shared_camera_{environment_index}",
                    episode_deadline_elapsed_wall_seconds=30,
                    simulation_seconds=0,
                    elapsed_wall_seconds=0,
                    source_monotonic_seconds=time.monotonic(),
                    joint_names=robot.joint_names,
                    joint_positions=joints[environment_index].tolist(),
                    joint_velocities=[0.0] * 29,
                    root_state_world=[
                        *default_roots[environment_index, :3],
                        *default_roots[environment_index, [6, 3, 4, 5]],
                        *([0.0] * 6),
                    ],
                )
                for environment_index in range(num_environments)
            ]
            baseline = renderer.capture_batch(states)
            baseline_depths = [
                np.load(Path(record["snapshot_directory"]) / "depth-m.npy") for record in baseline
            ]
            valid = np.isfinite(baseline_depths[0]) & np.isfinite(baseline_depths[1])
            if not valid.any():
                raise ValueError("Cloned cameras have no mutually valid depth pixels")
            same_pose_error = float(
                np.max(np.abs(baseline_depths[0][valid] - baseline_depths[1][valid]))
            )
            if same_pose_error > 0.002:
                raise ValueError(
                    f"Identical local poses produced different tiled depth: {same_pose_error}"
                )
            changed_states = [
                state.model_copy(
                    update={
                        "identifier": "independence",
                        "source_monotonic_seconds": time.monotonic(),
                    }
                )
                for state in states
            ]
            changed_joints = list(changed_states[1].joint_positions)
            changed_joints[robot.joint_names.index("waist_yaw_joint")] += 0.3
            changed_states[1] = changed_states[1].model_copy(
                update={"joint_positions": changed_joints, "indicator_activated": True}
            )
            changed = renderer.capture_batch(changed_states)
            changed_depths = [
                np.load(Path(record["snapshot_directory"]) / "depth-m.npy") for record in changed
            ]
            valid_unchanged = np.isfinite(baseline_depths[0]) & np.isfinite(changed_depths[0])
            unchanged_error = float(
                np.max(
                    np.abs(baseline_depths[0][valid_unchanged] - changed_depths[0][valid_unchanged])
                )
            )
            valid_changed = np.isfinite(baseline_depths[1]) & np.isfinite(changed_depths[1])
            changed_mean = float(
                np.mean(
                    np.abs(baseline_depths[1][valid_changed] - changed_depths[1][valid_changed])
                )
            )
            if unchanged_error > 0.002 or changed_mean < 0.01:
                raise ValueError(
                    f"Camera isolation failed: unchanged maximum {unchanged_error}, changed mean {changed_mean}"
                )
            result.update(
                passed=True,
                num_environments=num_environments,
                rgb_shape=list(renderer.camera.data.output["rgb"].torch.shape),
                same_local_pose_depth_max_error_m=same_pose_error,
                unchanged_environment_depth_max_error_m=unchanged_error,
                changed_environment_depth_mean_difference_m=changed_mean,
                captures=baseline + changed,
                scope="One upstream tiled renderer; camera 1 responds to its own changed waist pose while camera 0 depth remains unchanged. No native controller, timed-physics or model-capability claim.",
            )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if renderer is not None:
            renderer.close()
        atomic_json(output / "result.json", result, durable=True)
