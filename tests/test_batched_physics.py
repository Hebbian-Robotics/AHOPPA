"""Exercise the actual episode loop with CPU arrays and owned-component doubles."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.cli import EpisodeOptions
from humanoid_agent.dex3 import HAND_JOINT_NAMES, HandMotorCommand
from humanoid_agent.isaac_workcell import TASK_CONTACT_PAIRS
from humanoid_agent.paths import PROJECT_ROOT
from humanoid_agent.recording import EpisodeRecorder
from humanoid_agent.robot_camera import calibration
from humanoid_agent.runtime.camera import NativeCameraRenderer, NativeVisualState
from humanoid_agent.runtime.episode_io import (
    AppendRecord,
    BeginEpisode,
    CaptureBatchSnapshot,
    CaptureSnapshot,
    ObservationReady,
    WriteRecord,
)
from humanoid_agent.runtime.motor_targets import BatchedMotorTargets
from humanoid_agent.runtime.sdk import ReceivedMotorCommand, SonicMotorCommand
from humanoid_agent.runtime.shared import run_episode_physics
from humanoid_agent.runtime.state import TensorReadbackBuffers, copy_tensor_fields_to_cpu
from humanoid_agent.sensor_feedback import TRACKING_NAMES
from humanoid_agent.shutdown_task import STORAGE_CABINETS, TASK_DRAWER_JOINT


class CpuTensor(np.ndarray):
    def copy_(self, source: Any) -> CpuTensor:
        np.copyto(self, source)
        return self

    def permute(self, *dimensions: int) -> CpuTensor:
        return self.transpose(dimensions)

    def clone(self) -> CpuTensor:
        return self.copy()

    def cpu(self) -> CpuTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self)


def cpu_tensor(values: Any, **_options: Any) -> CpuTensor:
    return np.asarray(values, dtype=np.float32).view(CpuTensor)


def tensor_field(values: Any) -> SimpleNamespace:
    return SimpleNamespace(torch=cpu_tensor(values))


def concatenate_tensors(
    values: list[CpuTensor], dim: int, out: CpuTensor | None = None
) -> CpuTensor:
    if out is not None:
        np.concatenate(values, axis=dim, out=out)
        return out
    return np.concatenate(values, axis=dim).view(CpuTensor)


def empty_tensor(shape: tuple[int, ...], *, dtype: Any = np.float32, **_options: Any) -> CpuTensor:
    return np.empty(shape, dtype=dtype).view(CpuTensor)


class BatchedPhysicsTests(unittest.TestCase):
    def test_readback_reuses_transfer_storage_without_overwriting_owned_snapshots(self) -> None:
        torch = MagicMock()
        torch.empty.side_effect = empty_tensor
        torch.cat.side_effect = concatenate_tensors
        buffers = TensorReadbackBuffers()
        positions = cpu_tensor([[1, 2, 3], [4, 5, 6]])
        velocities = cpu_tensor([[7, 8], [9, 10]])
        with patch("humanoid_agent.runtime.state.importlib.import_module", return_value=torch):
            original = copy_tensor_fields_to_cpu(
                {"positions": positions, "velocities": velocities}, buffers=buffers
            )
            retained_view = original["positions"][0]
            for _ in range(5):
                positions += 1
                current = copy_tensor_fields_to_cpu(
                    {"positions": positions, "velocities": velocities}, buffers=buffers
                )
                np.testing.assert_array_equal(current["positions"], positions)
            self.assertEqual(torch.empty.call_count, 2)
            self.assertTrue(
                all(call.kwargs.get("out") is not None for call in torch.cat.call_args_list)
            )
            np.testing.assert_array_equal(original["positions"], [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_array_equal(retained_view, [1, 2, 3])
            self.assertFalse(np.shares_memory(original["positions"], current["positions"]))

    def test_camera_batch_reads_device_data_before_saving_owned_arrays(self) -> None:
        robot_count = 2
        height, width = 480, 640
        focal_x, focal_y, center_x, center_y = calibration()["intrinsics_fx_fy_cx_cy"]
        intrinsics = np.asarray(
            [[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]], dtype=np.float32
        )
        depth = np.full((robot_count, height, width, 1), 2, dtype=np.float32)
        images = np.stack(
            [
                np.full((height, width, 4), 30 + index, dtype=np.uint8)
                for index in range(robot_count)
            ]
        )
        observer_images = np.full((robot_count, 540, 960, 4), 90, dtype=np.uint8)
        camera = SimpleNamespace(
            data=SimpleNamespace(
                output={
                    "rgb": SimpleNamespace(torch=images.view(CpuTensor)),
                    "distance_to_image_plane": tensor_field(depth),
                },
                intrinsic_matrices=tensor_field(np.repeat(intrinsics[None], robot_count, axis=0)),
                pos_w=tensor_field(np.zeros((robot_count, 3))),
                quat_w_ros=tensor_field([[0, 0, 0, 1]] * robot_count),
            )
        )
        observer = SimpleNamespace(
            data=SimpleNamespace(
                output={"rgb": SimpleNamespace(torch=observer_images.view(CpuTensor))},
                pos_w=tensor_field(np.zeros((robot_count, 3))),
                quat_w_ros=tensor_field([[0, 0, 0, 1]] * robot_count),
                intrinsic_matrices=tensor_field(np.repeat(intrinsics[None], robot_count, axis=0)),
            )
        )
        torch = MagicMock()
        torch.float32 = np.float32
        torch.empty.side_effect = empty_tensor
        torch.cat.side_effect = concatenate_tensors
        torch.split.side_effect = lambda values, widths, dim: np.split(
            values, np.cumsum(widths)[:-1], axis=dim
        )
        torch.stack.side_effect = lambda values: np.stack(values).view(CpuTensor)
        events: list[str] = []

        def unproject(
            source_depth: CpuTensor, source_intrinsics: CpuTensor, *, is_ortho: bool
        ) -> CpuTensor:
            self.assertTrue(is_ortho)
            events.append("unproject")
            pixel_y, pixel_x = np.indices((height, width))
            points = np.stack(
                [
                    (pixel_x - source_intrinsics[0, 2]) / source_intrinsics[0, 0] * source_depth,
                    (pixel_y - source_intrinsics[1, 2]) / source_intrinsics[1, 1] * source_depth,
                    source_depth,
                ],
                axis=-1,
            )
            return cpu_tensor(points.transpose(1, 0, 2).reshape(-1, 3))

        def readback(destination: CpuTensor, source: CpuTensor) -> CpuTensor:
            events.append("readback")
            np.copyto(destination, source)
            return destination

        renderer = object.__new__(NativeCameraRenderer)
        renderer.retain_capture_media = True
        renderer.num_environments = robot_count
        renderer.readback_buffers = TensorReadbackBuffers()
        renderer.camera = camera
        renderer.wrist_cameras = {}
        renderer.observer = observer
        renderer.robot = SimpleNamespace(
            data=SimpleNamespace(root_link_pos_w=tensor_field(np.zeros((robot_count, 3))))
        )
        renderer.torch = torch
        with (
            patch.object(CpuTensor, "copy_", readback),
            patch(
                "humanoid_agent.runtime.camera.importlib.import_module",
                side_effect=lambda name: (
                    torch if name == "torch" else SimpleNamespace(unproject_depth=unproject)
                ),
            ),
        ):
            snapshot = renderer._read_capture_batch()
        self.assertEqual(events, ["unproject", "unproject", "readback", "readback"])
        np.testing.assert_array_equal(snapshot.images["rgb"], images[:, :, :, :3])
        np.testing.assert_array_equal(snapshot.geometry["depth"], depth[:, :, :, 0])
        # Device handles become unusable before evidence encoding starts.
        renderer.camera = object()
        renderer.robot = object()
        renderer.simulation = MagicMock()
        renderer.simulation.get_physics_step_count.return_value = 0
        renderer.simulation.get_physics_dt.return_value = 0.005
        renderer.initial_simulation_seconds = 0
        renderer.rendered_states = 0
        with (
            tempfile.TemporaryDirectory() as temporary,
            closing(EpisodeRecorder(Path(temporary))) as recorder,
        ):
            renderer.output = Path(temporary)
            renderer.recorder = recorder
            for environment_index in range(robot_count):
                state = NativeVisualState(
                    identifier="initial",
                    episode_id=f"episode_{environment_index}",
                    episode_deadline_elapsed_wall_seconds=60,
                    simulation_seconds=0,
                    elapsed_wall_seconds=0,
                    source_monotonic_seconds=100,
                    joint_names=list(JOINT_NAMES),
                    joint_positions=[0.0] * 29,
                    joint_velocities=[0.0] * 29,
                    root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                )
                record = renderer._write_capture(
                    state,
                    environment_index,
                    time.monotonic(),
                    np.zeros(3),
                    Rotation.identity(),
                    snapshot,
                )
                directory = renderer.output / f"env_{environment_index}" / "initial"
                with Image.open(directory / "onboard.png") as saved_image:
                    np.testing.assert_array_equal(
                        np.asarray(saved_image), images[environment_index, :, :, :3]
                    )
                np.testing.assert_array_equal(
                    np.load(directory / "depth-m.npy"),
                    snapshot.geometry["depth"][environment_index],
                )
                np.testing.assert_array_equal(
                    np.load(directory / "optical-points-m.npy"),
                    snapshot.geometry["optical_points"][environment_index],
                )
                self.assertEqual(
                    record["private_camera_audit"]["measured_optical_quaternion_ros_wxyz"],
                    [1, 0, 0, 0],
                )
                self.assertIsNotNone(record["observer_rgb_sha256"])

            for missing_depth in (np.inf, np.nan):
                snapshot.geometry["depth"][environment_index].fill(missing_depth)
                identifier = "sky_inf" if np.isinf(missing_depth) else "sky_nan"
                sky = state.model_copy(update={"identifier": identifier})
                sky_record = renderer._write_capture(
                    sky,
                    environment_index,
                    time.monotonic(),
                    np.zeros(3),
                    Rotation.identity(),
                    snapshot,
                )
                sky_directory = renderer.output / f"env_{environment_index}" / identifier
                sky_observation = json.loads((sky_directory / "observation.json").read_text())
                self.assertEqual(sky_observation["depth_tiles_m"], [[None] * 16 for _ in range(12)])
                self.assertEqual(sky_record["projection_audit_valid_pixels"], 0)
                self.assertFalse(np.isfinite(np.load(sky_directory / "depth-m.npy")).any())

    def test_motor_buffers_reuse_storage_and_preserve_joint_order_across_updates(self) -> None:
        robot = MagicMock()
        torch = MagicMock()
        torch.float32 = np.float32
        torch.empty.side_effect = empty_tensor
        motor_indices = list(range(9, 29)) + list(range(9))
        commands = [
            SonicMotorCommand.model_validate(
                {
                    field: [
                        float(100 * environment_index + joint_index + field_index)
                        for joint_index in range(29)
                    ]
                    for field_index, field in enumerate(BatchedMotorTargets.fields)
                }
            )
            for environment_index in range(2)
        ]
        upload_shapes = []

        def upload(destination: CpuTensor, source: CpuTensor) -> CpuTensor:
            upload_shapes.append(destination.shape)
            np.copyto(destination, source)
            return destination

        with (
            patch.object(CpuTensor, "copy_", upload),
            patch(
                "humanoid_agent.runtime.motor_targets.importlib.import_module", return_value=torch
            ),
        ):
            targets = BatchedMotorTargets(robot, motor_indices, 2)
            host_identity = id(targets.host)
            device_identity = id(targets.device)
            self.assertTrue(targets.apply(commands))
            self.assertFalse(targets.apply(commands))
            changed = commands[0].model_copy(
                update={"positions": [float(index + 1) for index in range(29)]}
            )
            self.assertTrue(targets.apply([changed, commands[1]]))
            changed_gains = commands[1].model_copy(update={"stiffness": [2.0] * 29})
            self.assertTrue(targets.apply([changed, changed_gains]))
            self.assertEqual(upload_shapes, [(5, 2, 29), (3, 2, 29), (5, 2, 29)])
            self.assertEqual(torch.empty.call_count, 2)
            self.assertTrue(torch.empty.call_args_list[0].kwargs["pin_memory"])
            self.assertEqual(id(targets.host), host_identity)
            self.assertEqual(id(targets.device), device_identity)
            expected_positions = np.asarray([changed.positions, changed_gains.positions])[
                :, np.argsort(motor_indices)
            ]
            np.testing.assert_array_equal(
                robot.set_joint_position_target.call_args.args[0], expected_positions
            )
            self.assertEqual(robot.write_joint_stiffness_to_sim.call_count, 2)
            self.assertEqual(robot.set_joint_position_target.call_count, 3)
            with self.assertRaisesRegex(ValueError, "one command"):
                targets.apply([changed])

    def test_body_and_fingers_reach_their_named_joints_in_one_motor_upload(self) -> None:
        motor_names = [*JOINT_NAMES, *HAND_JOINT_NAMES["left"], *HAND_JOINT_NAMES["right"]]
        robot = MagicMock()
        robot.joint_names = motor_names[::2] + motor_names[1::2]
        motor_indices = [robot.joint_names.index(name) for name in JOINT_NAMES]
        torch = MagicMock()
        torch.float32 = np.float32
        torch.empty.side_effect = empty_tensor
        body = {
            field: [float(index) for index in range(29)] for field in BatchedMotorTargets.fields
        }
        left = HandMotorCommand.model_validate(
            {
                field: [float(100 + index) for index in range(7)]
                for field in BatchedMotorTargets.fields
            }
        )
        right = HandMotorCommand.model_validate(
            {
                field: [float(200 + index) for index in range(7)]
                for field in BatchedMotorTargets.fields
            }
        )
        command = SonicMotorCommand.model_validate({**body, "left_hand": left, "right_hand": right})
        with patch(
            "humanoid_agent.runtime.motor_targets.importlib.import_module", return_value=torch
        ):
            targets = BatchedMotorTargets(robot, motor_indices, 1)
            targets.apply([command])
            expected_by_name = dict(
                zip(motor_names, [*range(29), *range(100, 107), *range(200, 207)], strict=True)
            )
            expected = [[expected_by_name[name] for name in robot.joint_names]]
            np.testing.assert_array_equal(
                robot.set_joint_position_target.call_args.args[0], expected
            )
            np.testing.assert_array_equal(
                robot.write_joint_stiffness_to_sim.call_args.args[0], expected
            )
            changed = command.model_copy(
                update={"right_hand": right.model_copy(update={"positions": [0.0] * 7})}
            )
            targets.apply([changed])
            for name in HAND_JOINT_NAMES["right"]:
                expected[0][robot.joint_names.index(name)] = 0
            np.testing.assert_array_equal(
                robot.set_joint_position_target.call_args.args[0], expected
            )
            with self.assertRaisesRegex(ValueError, "both hands"):
                targets.apply([command.model_copy(update={"left_hand": None})])

    def run_fixture(
        self,
        root: Path,
        robot_count: int,
        *,
        camera_failed: bool = False,
        scripted: bool = False,
        camera_enabled: bool = True,
        motor_expired: bool = False,
        simulation_stops_early: bool = False,
    ) -> None:
        output = root / "shared-physics"
        single_episode = robot_count == 1
        directories = (
            [root]
            if single_episode
            else [root / "episodes" / f"env_{index}" for index in range(robot_count)]
        )
        for directory in directories:
            (directory / "commands").mkdir(parents=True)
        arguments = EpisodeOptions(
            output=output,
            task_scene=PROJECT_ROOT / "configs/shutdown-storage-none-open.json",
            num_environments=robot_count,
            episode_directory=root if single_episode else None,
            wall_seconds=0.01,
            episode_kind="controller-check" if scripted else "model",
            diagnostic_phase_seconds=1,
            diagnostic_motion="standing",
            diagnostic_camera="on" if camera_enabled else "off",
            capture_interval_seconds=0.05,
        )
        origins = cpu_tensor([[12 * index, 0, 0] for index in range(robot_count)])
        roots = cpu_tensor(
            [[*origin[:2], 0.76, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for origin in origins]
        )
        body_states = np.repeat(roots[:, None], 4, axis=1)
        joint_values = np.zeros((robot_count, 43), dtype=np.float32)
        robot = MagicMock()
        robot.joint_names = [*JOINT_NAMES, *HAND_JOINT_NAMES["left"], *HAND_JOINT_NAMES["right"]]
        robot.body_names = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
        robot.data = SimpleNamespace(
            default_root_pose=tensor_field([[0, 0, 0.76, 0, 0, 0, 1]] * robot_count),
            default_root_vel=tensor_field(np.zeros((robot_count, 6))),
            default_joint_pos=tensor_field(joint_values),
            default_joint_vel=tensor_field(joint_values),
            root_state_w=tensor_field(roots),
            body_link_state_w=tensor_field(body_states),
            body_link_pos_w=tensor_field(body_states[:, :, :3]),
            body_link_quat_w=tensor_field(body_states[:, :, 3:7]),
            joint_pos=tensor_field(joint_values),
            joint_vel=tensor_field(joint_values),
            joint_acc=tensor_field(joint_values),
            applied_torque=tensor_field(joint_values),
        )
        entities = {"robot": robot}
        for name, _, _ in TASK_CONTACT_PAIRS:
            entities["task_contact_" + name] = SimpleNamespace(
                data=SimpleNamespace(
                    force_matrix_w=tensor_field(np.zeros((robot_count, 1, 1, 3))),
                    force_matrix_w_history=tensor_field(np.zeros((robot_count, 4, 1, 1, 3))),
                )
            )
        cabinet_joint_names = ["drawer_bottom_joint", TASK_DRAWER_JOINT]
        cabinet_joints = np.zeros((robot_count, 2), dtype=np.float32)
        for identifier in STORAGE_CABINETS:
            cabinet = MagicMock()
            cabinet.joint_names = cabinet_joint_names
            cabinet.body_names = ["sektion", "drawer_top"]
            cabinet.data = SimpleNamespace(
                joint_pos=tensor_field(cabinet_joints),
                joint_vel=tensor_field(cabinet_joints),
                default_joint_pos=tensor_field(cabinet_joints),
                default_joint_vel=tensor_field(cabinet_joints),
                body_mass=tensor_field(np.ones((robot_count, 2))),
                joint_stiffness=tensor_field(cabinet_joints),
                joint_damping=tensor_field(cabinet_joints),
                joint_friction_coeff=tensor_field(cabinet_joints),
            )
            entities["cabinet_" + identifier] = cabinet
            entities["shutdown_drawer_contacts_" + identifier] = SimpleNamespace(
                body_names=["right_wrist_yaw_link", "torso_link"],
                data=SimpleNamespace(
                    normal_force_matrix_w_history=tensor_field(np.zeros((robot_count, 4, 2, 1, 3)))
                ),
            )
        for name in ("pelvis", "torso"):
            entities["imu_" + name] = SimpleNamespace(
                data=SimpleNamespace(
                    ang_vel_b=tensor_field(np.zeros((robot_count, 3))),
                    lin_acc_b=tensor_field(np.zeros((robot_count, 3))),
                )
            )
        scene = MagicMock()
        scene.__getitem__.side_effect = entities.__getitem__
        scene.env_origins = origins
        clock_seconds = [1000.0]
        original_monotonic = time.monotonic
        original_sleep = time.sleep
        simulation = MagicMock()

        def advance_clock(*_arguments: Any, **_keywords: Any) -> None:
            clock_seconds[0] += 0.05

        if scripted:
            simulation.step.side_effect = advance_clock
        if simulation_stops_early:
            simulation.is_running.side_effect = [True, False]
        simulation.device = "cuda:0"
        simulation.physics_manager = SimpleNamespace(__name__="Newton")
        torch = MagicMock()
        torch.tensor.side_effect = cpu_tensor
        torch.float32 = np.float32
        torch.empty.side_effect = empty_tensor
        torch.cat.side_effect = concatenate_tensors
        torch.split.side_effect = lambda values, widths, dim: np.split(
            values, np.cumsum(widths)[:-1], axis=dim
        )
        hand_command = HandMotorCommand(
            positions=[0.0] * 7,
            velocities=[0.0] * 7,
            feedforward_torques=[0.0] * 7,
            stiffness=[1.0] * 7,
            damping=[1.0] * 7,
        )
        command = SonicMotorCommand(
            positions=[0.0] * 29,
            velocities=[0.0] * 29,
            feedforward_torques=[0.0] * 29,
            stiffness=[1.0] * 29,
            damping=[1.0] * 29,
            left_hand=hand_command,
            right_hand=hand_command,
        )

        created_bridges = []
        submitted_operations = []

        def create_bridge(_python: Path, directory: Path, **_options: Any) -> MagicMock:
            atomic_json(directory / "controller-configuration.json", {"arguments": ["native"]})
            atomic_json(directory / "controller-process.json", {"pid": 123})
            bridge = MagicMock()
            bridge.controller_ready = True
            bridge.latest_command.side_effect = lambda: ReceivedMotorCommand(
                command, time.monotonic() - (2 if motor_expired else 0)
            )
            bridge.take_planner_publications.return_value = []
            created_bridges.append(bridge)
            return bridge

        camera = MagicMock()
        camera.output = output / "camera-output"
        camera.state_path = output / "camera-state.json"
        camera.process.poll.return_value = 1 if camera_failed else None

        def create_camera(*_arguments: Any, **_options: Any) -> MagicMock:
            camera.output.mkdir(parents=True)
            return camera

        def create_io(*_arguments: Any, **_options: Any) -> MagicMock:
            worker = MagicMock()
            pending_events = []

            def take_events() -> list[Any]:
                events = list(pending_events)
                pending_events.clear()
                return events

            worker.events.side_effect = take_events

            def submit(operation: Any) -> None:
                submitted_operations.append(operation)
                if isinstance(operation, (CaptureBatchSnapshot, CaptureSnapshot)):
                    pending_events.append(ObservationReady(operation.identifier))
                if isinstance(operation, WriteRecord):
                    payload = operation.payload
                    atomic_json(
                        operation.path,
                        payload.model_dump(mode="json")
                        if isinstance(payload, BaseModel)
                        else payload,
                    )

            worker.submit.side_effect = submit
            return worker

        hand_urdf = root / "articulated-hands.urdf"
        hand_urdf.write_text("<robot/>")
        hand_limits = {
            name: {"lower": -1.0, "upper": 1.0}
            for names in HAND_JOINT_NAMES.values()
            for name in names
        }
        encoder = MagicMock()
        encoder.measure.return_value = {
            "tracking_points_m": {name: [0, 0, 0] for name in TRACKING_NAMES},
            "tracking_orientations_wxyz": {name: [1, 0, 0, 0] for name in TRACKING_NAMES},
        }
        with (
            patch(
                "humanoid_agent.runtime.shared.NativeCameraProcess", side_effect=create_camera
            ) as camera_factory,
            patch(
                "humanoid_agent.runtime.shared.time.monotonic",
                side_effect=(lambda: clock_seconds[0]) if scripted else original_monotonic,
            ),
            patch(
                "humanoid_agent.runtime.shared.time.sleep",
                side_effect=(lambda _seconds: None) if scripted else original_sleep,
            ),
            patch("humanoid_agent.runtime.shared.register_newton_scene_schemas"),
            patch(
                "humanoid_agent.runtime.shared.create_newton_simulation_configuration",
                return_value=SimpleNamespace(dt=0.005),
            ),
            patch("humanoid_agent.runtime.shared.launch_newton_runtime"),
            patch("humanoid_agent.runtime.shared.create_newton_workcell_scene", return_value=scene),
            patch("humanoid_agent.runtime.shared.EncoderKinematics", return_value=encoder),
            patch("humanoid_agent.runtime.shared.SonicSdkProcessBridge", side_effect=create_bridge),
            patch("humanoid_agent.runtime.shared.EpisodeIO", side_effect=create_io),
            patch(
                "humanoid_agent.runtime.shared.articulated_hand_urdf",
                return_value=hand_urdf,
            ),
            patch("humanoid_agent.runtime.actor.hand_joint_limits", return_value=hand_limits),
            patch("humanoid_agent.runtime.actor.EpisodeIO", side_effect=create_io),
            patch(
                "humanoid_agent.runtime.shared.importlib.import_module",
                side_effect=lambda name: (
                    torch
                    if name == "torch"
                    else SimpleNamespace(SimulationContext=lambda _configuration: simulation)
                ),
            ),
        ):
            run_episode_physics(arguments)
        identifiers = []
        for environment_index, directory in enumerate(directories):
            configuration = json.loads((directory / "physics/configuration.json").read_text())
            identifiers.append(configuration["episode_id"])
            self.assertEqual(configuration["environment_index"], environment_index)
            self.assertEqual(
                configuration["environment_origin_m"], origins[environment_index].tolist()
            )
            outcome = json.loads((directory / "physics/task-outcome.json").read_text())
            self.assertEqual(outcome["end_reason"], "wall_time_limit")
            self.assertFalse(outcome["task_success"])
            timing = json.loads((directory / "physics/timing.json").read_text())
            self.assertGreater(timing["all"]["ticks"], 0)
            expected_camera = camera.output
            if robot_count > 1:
                expected_camera /= f"env_{environment_index}"
            if camera_enabled:
                self.assertEqual(
                    (directory / "physics/camera-output").resolve(), expected_camera.resolve()
                )
            else:
                self.assertFalse((directory / "physics/camera-output").exists())
        self.assertEqual(len(set(identifiers)), robot_count)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["num_environments"], robot_count)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["cleanup_errors"], [])
        self.assertEqual(result["command_source"], "scripted" if scripted else "model")
        self.assertEqual(result["camera_enabled"], camera_enabled)
        self.assertEqual(camera_factory.call_count, int(camera_enabled))
        if scripted:
            self.assertTrue(
                all(
                    operation.capture_initial_observation == camera_enabled
                    for operation in submitted_operations
                    if isinstance(operation, BeginEpisode)
                )
            )
            captures = [
                operation
                for operation in submitted_operations
                if isinstance(operation, CaptureBatchSnapshot)
            ]
            self.assertEqual(bool(captures), camera_enabled)
            samples = [call.args[0] for call in created_bridges[0].publish_planner.call_args_list]
            self.assertTrue(all(sample.locomotion_mode == 0 for sample in samples))
            traces = [
                operation.payload
                for operation in submitted_operations
                if isinstance(operation, AppendRecord) and "phase" in operation.payload
            ]
            self.assertEqual({row["phase"] for row in traces}, {"stand"})
            self.assertTrue(all("cabinet_joint_positions" in row for row in traces))
            for bridge in created_bridges:
                bridge.request_stop.assert_called()
                bridge.close.assert_called()

    def test_scripted_checks_use_production_physics_with_both_camera_workloads(self) -> None:
        for robot_count in (1, 3):
            for camera_enabled in (False, True):
                with (
                    self.subTest(robot_count=robot_count, camera_enabled=camera_enabled),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    self.run_fixture(
                        Path(temporary), robot_count, scripted=True, camera_enabled=camera_enabled
                    )

    def test_one_and_three_robots_finish_through_the_same_physics_loop(self) -> None:
        for robot_count in (1, 3):
            with self.subTest(robot_count=robot_count), tempfile.TemporaryDirectory() as temporary:
                self.run_fixture(Path(temporary), robot_count)

    def test_camera_failure_after_warmup_fails_the_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Camera worker exited during physics"):
                self.run_fixture(root, 1, camera_failed=True)
            result = json.loads((root / "shared-physics/result.json").read_text())
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertFalse((root / "physics/task-outcome.json").exists())

    def test_scripted_camera_off_motor_failure_runs_production_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, "motor stream expired"):
                self.run_fixture(root, 1, scripted=True, camera_enabled=False, motor_expired=True)
            result = json.loads((root / "shared-physics/result.json").read_text())
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertEqual(result["command_source"], "scripted")
            self.assertEqual(result["cleanup_errors"], [])
            self.assertFalse((root / "physics/task-outcome.json").exists())

    def test_premature_simulator_exit_is_not_a_completed_scripted_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                RuntimeError, "before all episodes reached a terminal state"
            ):
                self.run_fixture(
                    root, 1, scripted=True, camera_enabled=False, simulation_stops_early=True
                )
            result = json.loads((root / "shared-physics/result.json").read_text())
            self.assertEqual(result["status"], "infrastructure_error")
            self.assertEqual(result["cleanup_errors"], [])
            self.assertFalse((root / "physics/task-outcome.json").exists())
