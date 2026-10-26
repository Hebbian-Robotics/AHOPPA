"""Cold first-frame compilation shares the camera's bounded startup budget."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid_agent.runtime.camera import (
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    CameraReady,
    NativeCameraProcess,
)


class CameraStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.monotonic_seconds = 0.0
        self.worker = MagicMock()
        self.worker.pid = 123
        self.worker.poll.return_value = None

        def launch_worker(*_arguments: object, **_options: object) -> MagicMock:
            return self.worker

        self.events = MagicMock()
        self.events.receive.return_value = CameraReady().model_dump_json().encode()
        for patcher in (
            patch("humanoid_agent.runtime.camera.LocalPullReceiver", return_value=self.events),
            patch("humanoid_agent.runtime.camera.subprocess.Popen", side_effect=launch_worker),
            patch("humanoid_agent.runtime.camera.process_record", return_value={"pid": 123}),
            patch(
                "humanoid_agent.runtime.camera.time.monotonic",
                side_effect=lambda: self.monotonic_seconds,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.camera = NativeCameraProcess(
            Path("python"), Path("isaac"), Path("scene.json"), None, self.directory
        )
        self.addCleanup(self.camera.log.close)
        self.addCleanup(self.camera.sender.close)
        self.camera.initial_batch = MagicMock()
        self.completion = (
            CameraCaptureCompleted(
                identifier="initial",
                captures=[
                    CameraCaptureIdentity(
                        environment_index=0, episode_id="episode", identifier="initial"
                    )
                ],
            )
            .model_dump_json()
            .encode()
        )

    def test_cold_first_frame_can_arrive_after_two_minutes_before_episode_release(self) -> None:
        self.monotonic_seconds = 130

        def publish_initial_frame(**_options: object) -> bytes:
            self.monotonic_seconds += 1
            return self.completion

        with patch.object(self.events, "receive", side_effect=publish_initial_frame):
            self.camera.wait_for_initial_batch()
        record = json.loads((self.directory / "camera-startup.json").read_text())
        self.assertTrue(record["initial_rgbd_ready"])
        self.assertFalse(record["episode_started"])
        self.assertEqual(record["elapsed_seconds"], 131)

    def test_first_frame_does_not_reset_the_budget_after_scene_setup(self) -> None:
        self.monotonic_seconds = 899
        stale_receipt = self.camera.output / "batches/initial.json"
        stale_receipt.parent.mkdir(parents=True)
        stale_receipt.write_text('{"identifier": "initial", "records": []}')

        def exhaust_remaining_budget(**_options: object) -> None:
            self.monotonic_seconds += 2

        with (
            patch.object(self.events, "receive", side_effect=exhaust_remaining_budget),
            self.assertRaisesRegex(TimeoutError, "initial capture"),
        ):
            self.camera.wait_for_initial_batch()
        self.assertEqual(self.monotonic_seconds, 901)
        self.assertFalse((self.directory / "camera-startup.json").exists())

    def test_worker_failure_during_warmup_is_reported_without_waiting_for_timeout(self) -> None:
        self.worker.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "Camera worker failed startup"):
            self.camera.wait_for_initial_batch()
        self.assertFalse((self.directory / "camera-startup.json").exists())


if __name__ == "__main__":
    unittest.main()
