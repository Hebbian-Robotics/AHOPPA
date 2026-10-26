"""One and many robots share ownership, capture routing and failure recording."""

import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.cli import EpisodeOptions, run_task
from humanoid_agent.runtime.camera import (
    CAMERA_EVENT_ADAPTER,
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    CameraCaptureRequest,
    CameraReady,
    CameraStopRequest,
    NativeVisualBatch,
    NativeVisualState,
)
from humanoid_agent.runtime.camera_worker import main as run_camera_worker
from humanoid_agent.runtime.episode import run_episode
from humanoid_agent.runtime.group import run_episode_group
from humanoid_agent.runtime.layout import EpisodeLayout
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender


class UnifiedRuntimeTests(unittest.TestCase):
    def test_layout_preserves_single_episode_and_routes_every_group_camera(self) -> None:
        root = Path("/recordings/attempt")
        single = EpisodeLayout(root, 1, single_episode=True)
        self.assertEqual(single.episode_directories, (root,))
        self.assertEqual(single.camera_directory(0), root / "shared-physics/camera-output")
        for robot_count in (1, 2, 3, 16):
            layout = EpisodeLayout(root, robot_count)
            self.assertEqual(len(layout.episode_directories), robot_count)
            for environment_index, directory in enumerate(layout.episode_directories):
                self.assertEqual(directory, root / "episodes" / f"env_{environment_index}")
                expected_camera = root / "shared-physics/camera-output"
                if robot_count > 1:
                    expected_camera /= f"env_{environment_index}"
                self.assertEqual(layout.camera_directory(environment_index), expected_camera)
            with self.assertRaises(ValueError):
                layout.camera_directory(robot_count)
        for robot_count, single_episode in ((0, False), (17, False), (2, True)):
            with (
                self.subTest(robot_count=robot_count, single_episode=single_episode),
                self.assertRaises(ValueError),
            ):
                EpisodeLayout(root, robot_count, single_episode)

    def test_owners_launch_the_same_worker_and_retain_all_assignments_on_failure(self) -> None:
        for single_episode, providers in (
            (True, ("sol",)),
            (False, ("opus",)),
            (False, ("sol", "opus", "gemini")),
        ):
            with (
                self.subTest(single_episode=single_episode, providers=providers),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                output = root / "attempt"
                frozen_source = root / "source"
                frozen_inputs = root / "inputs"
                frozen_source.mkdir()
                frozen_inputs.mkdir()
                frozen_manifest = root / "source-manifest.json"
                frozen_manifest.write_text("{}")
                prepared = SimpleNamespace(
                    source=frozen_source,
                    constants=frozen_inputs / "constants.json",
                    robot_urdf=frozen_inputs / "robot.urdf",
                    scene=frozen_inputs / "scene.json",
                    decorations=None,
                    protocol=None,
                )
                arguments = EpisodeOptions(
                    output=output,
                    provider="sol",
                    providers=providers,
                    task_scene=Path("configs/shutdown-storage-none-open.json"),
                )
                with (
                    patch(
                        "humanoid_agent.runtime.group.prepare_episode_inputs", return_value=prepared
                    ),
                    patch(
                        "humanoid_agent.runtime.group.create_host_process_environment",
                        return_value={"PYTHONPATH": str(frozen_source)},
                    ),
                    patch("humanoid_agent.runtime.group.preflight_episode"),
                    patch("humanoid_agent.runtime.group.CudaMultiprocessService") as service_class,
                    patch(
                        "humanoid_agent.runtime.group.subprocess.Popen",
                        side_effect=RuntimeError("injected component launch failure"),
                    ) as launch,
                ):
                    service = service_class.return_value
                    service.process = None
                    service.start.side_effect = lambda environment: environment
                    with self.assertRaisesRegex(RuntimeError, "injected component"):
                        if single_episode:
                            run_episode(arguments)
                        else:
                            run_episode_group(arguments)
                    command = launch.call_args.args[0]
                    self.assertEqual(command[command.index("--role") + 1], "physics")
                    self.assertEqual(
                        command[command.index("--num-environments") + 1], str(len(providers))
                    )
                    self.assertEqual("--episode-directory" in command, single_episode)
                    self.assertEqual(service.start.call_count, int(len(providers) > 1))
                layout = EpisodeLayout(output, len(providers), single_episode)
                for provider, directory in zip(providers, layout.episode_directories, strict=True):
                    terminal = json.loads((directory / "episode.json").read_text())
                    self.assertEqual(terminal["provider"], provider)
                    self.assertEqual(terminal["status"], "infrastructure_error")
                    self.assertEqual(terminal["component_exit_codes"], {})
                    self.assertTrue((directory / "commands").is_dir())

    def test_both_worker_role_names_dispatch_to_the_same_loop(self) -> None:
        for role in ("physics", "shared-physics"):
            with self.subTest(role=role):
                arguments = EpisodeOptions(
                    role=role, output=Path("attempt/shared-physics"), task_scene=Path("scene.json")
                )
                with patch("humanoid_agent.runtime.shared.run_episode_physics") as physics:
                    run_task(arguments)
                    physics.assert_called_once_with(arguments)

    def test_single_environment_uses_batch_receipts_without_breaking_single_pose_workers(
        self,
    ) -> None:
        state = NativeVisualState(
            identifier="initial",
            episode_id="episode_a",
            episode_deadline_elapsed_wall_seconds=60,
            simulation_seconds=0,
            elapsed_wall_seconds=0,
            source_monotonic_seconds=100,
            joint_names=list(JOINT_NAMES),
            joint_positions=[0.0] * 29,
            joint_velocities=[0.0] * 29,
            root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        batch = NativeVisualBatch(identifier="initial", states=[state])
        receipt = {
            "environment_index": 0,
            "episode_id": state.episode_id,
            "identifier": state.identifier,
        }
        batch.require_matching_receipt(
            CameraCaptureCompleted(
                identifier="initial", captures=[CameraCaptureIdentity.model_validate(receipt)]
            )
        )
        for batch_captures in (False, True):
            with (
                self.subTest(batch_captures=batch_captures),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                state_path = root / "state.json"
                replies = LocalPullReceiver(root / "ready.json")
                self.addCleanup(replies.close)
                sender = LocalPushSender(state_path)
                send_errors: list[Exception] = []

                def send_frames(
                    camera_sender: LocalPushSender, payload: bytes, failures: list[Exception]
                ) -> None:
                    try:
                        camera_sender.send(payload)
                        camera_sender.send(CameraStopRequest().model_dump_json().encode())
                    except Exception as error:
                        failures.append(error)

                producer = Thread(
                    target=send_frames,
                    args=(
                        sender,
                        CameraCaptureRequest(
                            state=batch if batch_captures else state, reply_to=root / "ready.json"
                        )
                        .model_dump_json()
                        .encode(),
                        send_errors,
                    ),
                )
                producer.start()
                renderer = MagicMock()
                renderer.rendered_states = 1
                renderer.capture_batch.return_value = [receipt]
                renderer.capture.return_value = receipt
                with (
                    patch("humanoid_agent.runtime.camera_worker.register_newton_scene_schemas"),
                    patch(
                        "humanoid_agent.runtime.camera_worker.create_newton_simulation_configuration"
                    ),
                    patch("humanoid_agent.runtime.camera_worker.launch_newton_runtime"),
                    patch(
                        "humanoid_agent.runtime.camera_worker.NativeCameraRenderer",
                        return_value=renderer,
                    ),
                    patch(
                        "humanoid_agent.runtime.camera_worker.os._exit", side_effect=AssertionError
                    ),
                    patch("humanoid_agent.runtime.camera_worker.importlib.import_module"),
                ):
                    run_camera_worker(
                        cli_isaac_source=root,
                        cli_scene=root / "scene.json",
                        cli_state=state_path,
                        cli_output=root,
                        cli_ready=root / "ready.json",
                        batch_captures=batch_captures,
                    )
                producer.join(timeout=2)
                sender.close()
                self.assertFalse(send_errors)
                self.assertFalse(producer.is_alive())
                ready_payload = replies.receive(timeout_ms=1000)
                assert ready_payload is not None
                self.assertIsInstance(
                    CAMERA_EVENT_ADAPTER.validate_json(ready_payload), CameraReady
                )
                captured_payload = replies.receive(timeout_ms=1000)
                assert captured_payload is not None
                completion = CAMERA_EVENT_ADAPTER.validate_json(captured_payload)
                assert isinstance(completion, CameraCaptureCompleted)
                self.assertEqual(
                    [capture.model_dump() for capture in completion.captures], [receipt]
                )
                self.assertFalse((root / "ready.json").exists())
                self.assertFalse((root / "batches").exists())
                if batch_captures:
                    renderer.capture_batch.assert_called_once_with([state])
                    renderer.capture.assert_not_called()
                else:
                    renderer.capture.assert_called_once_with(state)
                    renderer.capture_batch.assert_not_called()
                renderer.close.assert_called_once()
