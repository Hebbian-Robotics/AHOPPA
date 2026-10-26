"""A real worker hard exit preserves captures recorded before a renderer failure."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.recording import iter_records, validate_recording
from humanoid_agent.runtime.camera import CameraCaptureRequest, NativeVisualState
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender


class CameraRecordingFailureTests(unittest.TestCase):
    def test_capture_failure_finalizes_prior_evidence_before_hard_exit(self) -> None:
        # Only the unavailable GPU boundary is replaced. The child uses real IPC,
        # Foxglove recording, filesystem synchronization and os._exit.
        worker_source = """
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from humanoid_agent.recording import EpisodeRecorder
from humanoid_agent.runtime import camera_worker

root = Path(sys.argv[1])
output = root / "camera-output"
output.mkdir()
class Renderer:
    def __init__(self, *_arguments, **_options):
        self.recorder = EpisodeRecorder(output)
        self.simulation = SimpleNamespace(is_running=lambda: True)
        self.rendered_states = 0
    def capture(self, state):
        if self.rendered_states:
            raise RuntimeError("injected rendering failure")
        self.recorder.write("capture-records.jsonl", {"identifier": state.identifier})
        self.rendered_states += 1
        return {"environment_index": 0, "episode_id": state.episode_id, "identifier": state.identifier}
    def close(self):
        self.recorder.close()
torch = SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
with patch.object(camera_worker, "register_newton_scene_schemas"), \
     patch.object(camera_worker, "create_newton_simulation_configuration"), \
     patch.object(camera_worker, "launch_newton_runtime", return_value=nullcontext()), \
     patch.object(camera_worker, "NativeCameraRenderer", Renderer), \
     patch.object(camera_worker.importlib, "import_module", return_value=torch):
    camera_worker.main(cli_isaac_source=root, cli_scene=root / "scene.json",
        cli_state=root / "camera-state", cli_output=output, cli_ready=root / "camera-ready.json")
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replies = LocalPullReceiver(root / "camera-ready.json")
            self.addCleanup(replies.close)
            process = subprocess.Popen(
                [sys.executable, "-c", worker_source, str(root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            sender = None
            try:
                deadline = time.monotonic() + 10
                while replies.receive(timeout_ms=10) is None:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("Camera test worker did not become ready")
                    time.sleep(0.01)
                sender = LocalPushSender(root / "camera-state")
                state = NativeVisualState(
                    identifier="initial",
                    episode_id="episode",
                    episode_deadline_elapsed_wall_seconds=60,
                    simulation_seconds=0,
                    elapsed_wall_seconds=0,
                    source_monotonic_seconds=100,
                    joint_names=JOINT_NAMES,
                    joint_positions=[0.0] * 29,
                    joint_velocities=[0.0] * 29,
                    root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                )
                for identifier in ("initial", "failed_capture"):
                    sender.send(
                        CameraCaptureRequest(
                            state=state.model_copy(update={"identifier": identifier}),
                            reply_to=root / "camera-ready.json",
                        )
                        .model_dump_json()
                        .encode()
                    )
                self.assertEqual(process.wait(timeout=10), 1)
                failure = json.loads((root / "camera-ready.error.json").read_text())
                self.assertEqual(failure["type"], "RuntimeError")
                self.assertIsNone(failure["recording_finalization_error"])
                self.assertEqual(
                    list(iter_records(root / "camera-output/capture-records.jsonl")),
                    [{"identifier": "initial"}],
                )
                self.assertEqual(validate_recording(root / "camera-output/records.mcap"), 1)
            finally:
                if sender is not None:
                    sender.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if process.stderr is not None:
                    process.stderr.close()
