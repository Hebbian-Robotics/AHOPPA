"""Camera completion messages acknowledge exact batches and isolated robot observations."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.runtime.camera import (
    CAMERA_EVENT_ADAPTER,
    CAMERA_REQUEST_ADAPTER,
    CameraCaptureCompleted,
    CameraCaptureIdentity,
    CameraCaptureRequest,
    CameraReady,
    CameraStopRequest,
    NativeVisualBatch,
    NativeVisualState,
)
from humanoid_agent.runtime.episode_io import (
    CaptureBatchSnapshot,
    ObservationReady,
    StopIO,
    _io_worker,
)
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender


def camera_batch(identifier: str) -> NativeVisualBatch:
    return NativeVisualBatch(
        identifier=identifier,
        states=[
            NativeVisualState(
                identifier=f"{identifier}_{index}",
                episode_id=f"episode_{index}",
                episode_deadline_elapsed_wall_seconds=60,
                simulation_seconds=0,
                elapsed_wall_seconds=0,
                source_monotonic_seconds=100,
                joint_names=JOINT_NAMES,
                joint_positions=[0.0] * 29,
                joint_velocities=[0.0] * 29,
                root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            for index in range(2)
        ],
    )


def batch_completion(batch: NativeVisualBatch) -> CameraCaptureCompleted:
    return CameraCaptureCompleted(
        identifier=batch.identifier,
        captures=[
            CameraCaptureIdentity(
                environment_index=index,
                episode_id=state.episode_id,
                identifier=state.identifier,
            )
            for index, state in enumerate(batch.states)
        ],
    )


def run_io_worker(
    output: Path,
    incoming: queue.Queue[object],
    outgoing: queue.Queue[object],
    ready: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        _io_worker(output, output / "commands", incoming, outgoing, ready)
    except BaseException as error:
        errors.append(error)
        ready.set()


class CameraTransportTests(unittest.TestCase):
    def test_camera_worker_routes_each_robot_only_to_its_requested_destination(self) -> None:
        worker_source = """
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from humanoid_agent.runtime import camera_worker

root = Path(sys.argv[1])
class Renderer:
    def __init__(self, *_arguments, **options):
        options_output = root / "camera-output"
        options_output.mkdir()
        self.simulation = SimpleNamespace(is_running=lambda: True)
        self.rendered_states = 0
    def capture_batch(self, states):
        self.rendered_states += len(states)
        return [dict(environment_index=index, episode_id=state.episode_id,
                     identifier=state.identifier) for index, state in enumerate(states)]
    def close(self):
        pass
torch = SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
with patch.object(camera_worker, "register_newton_scene_schemas"), \
     patch.object(camera_worker, "create_newton_simulation_configuration"), \
     patch.object(camera_worker, "launch_newton_runtime", return_value=nullcontext()), \
     patch.object(camera_worker, "NativeCameraRenderer", Renderer), \
     patch.object(camera_worker.importlib, "import_module", return_value=torch):
    camera_worker.main(cli_isaac_source=root, cli_scene=root / "scene.json",
        cli_state=root / "camera-state", cli_output=root / "camera-output",
        cli_ready=root / "ready", num_environments=2, batch_captures=True)
"""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
            root = Path(temporary)
            aggregate = resources.enter_context(closing(LocalPullReceiver(root / "ready")))
            robot_receivers = [
                resources.enter_context(closing(LocalPullReceiver(root / f"robot_{index}")))
                for index in range(2)
            ]
            process = subprocess.Popen(
                [sys.executable, "-c", worker_source, str(root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                ready = aggregate.receive(timeout_ms=10000)
                self.assertIsNotNone(ready)
                assert ready is not None
                self.assertIsInstance(CAMERA_EVENT_ADAPTER.validate_json(ready), CameraReady)
                sender = resources.enter_context(closing(LocalPushSender(root / "camera-state")))
                route_cases: tuple[tuple[str, list[Path | None]], ...] = (
                    ("first", [root / "robot_0", root / "robot_1"]),
                    ("second", [None, root / "robot_1"]),
                )
                for identifier, routes in route_cases:
                    batch = camera_batch(identifier)
                    sender.send(
                        CameraCaptureRequest(
                            state=batch, reply_to=root / "ready", observation_replies=routes
                        )
                        .model_dump_json()
                        .encode()
                    )
                    payload = aggregate.receive(timeout_ms=2000)
                    assert payload is not None
                    completed = CAMERA_EVENT_ADAPTER.validate_json(payload)
                    self.assertEqual(completed, batch_completion(batch))
                    for index, receiver in enumerate(robot_receivers):
                        payload = receiver.receive(timeout_ms=100)
                        if routes[index] is None:
                            self.assertIsNone(payload)
                        else:
                            assert payload is not None
                            self.assertEqual(
                                CAMERA_EVENT_ADAPTER.validate_json(payload),
                                CameraCaptureCompleted(
                                    identifier=batch.states[index].identifier,
                                    captures=[batch_completion(batch).captures[index]],
                                ),
                            )
                sender.send(CameraStopRequest().model_dump_json().encode())
                _stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr.decode())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if process.stderr is not None:
                    process.stderr.close()

    def test_batch_io_requires_a_matching_completion_message_despite_stale_receipt_file(
        self,
    ) -> None:
        for wrong_robot in (False, True):
            with self.subTest(wrong_robot=wrong_robot), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                batch = camera_batch("pending")
                completion = batch_completion(batch)
                receipt_path = root / "camera-output/batches/pending.json"
                receipt_path.parent.mkdir(parents=True)
                atomic_json(
                    receipt_path,
                    {
                        "identifier": batch.identifier,
                        "records": [
                            capture.model_dump(mode="json") for capture in completion.captures
                        ],
                    },
                )
                incoming: queue.Queue[object] = queue.Queue()
                outgoing: queue.Queue[object] = queue.Queue()
                ready = threading.Event()
                errors: list[BaseException] = []

                with (
                    closing(LocalPullReceiver(root / "camera-state.json")) as camera_requests,
                    closing(LocalPushSender(root / "camera-events")) as camera_replies,
                    patch("humanoid_agent.runtime.episode_io.sys.platform", "linux"),
                    patch("humanoid_agent.runtime.episode_io.os.sched_setaffinity", create=True),
                ):
                    worker = threading.Thread(
                        target=run_io_worker, args=(root, incoming, outgoing, ready, errors)
                    )
                    worker.start()
                    try:
                        self.assertTrue(ready.wait(timeout=3))
                        self.assertFalse(errors)
                        incoming.put(
                            CaptureBatchSnapshot(
                                batch.identifier, batch.model_dump(mode="json"), time.monotonic()
                            )
                        )
                        request_payload = camera_requests.receive(timeout_ms=2000)
                        assert request_payload is not None
                        request = CAMERA_REQUEST_ADAPTER.validate_json(request_payload)
                        self.assertIsInstance(request, CameraCaptureRequest)
                        with self.assertRaises(queue.Empty):
                            outgoing.get(timeout=0.1)
                        if wrong_robot:
                            completion = completion.model_copy(
                                update={
                                    "captures": [
                                        completion.captures[0].model_copy(
                                            update={"episode_id": "foreign"}
                                        ),
                                        completion.captures[1],
                                    ]
                                }
                            )
                        camera_replies.send(completion.model_dump_json().encode())
                        if wrong_robot:
                            worker.join(timeout=3)
                            self.assertEqual(len(errors), 1)
                            self.assertIsInstance(errors[0], ValueError)
                            self.assertTrue(outgoing.empty())
                        else:
                            self.assertEqual(outgoing.get(timeout=2), ObservationReady("pending"))
                            incoming.put(StopIO(1))
                            worker.join(timeout=3)
                            self.assertFalse(errors)
                        self.assertFalse(worker.is_alive())
                    finally:
                        if worker.is_alive():
                            incoming.put(object())
                            worker.join(timeout=3)
                summary = json.loads((root / "io-summary.json").read_text())
                self.assertEqual(summary["status"], "failed" if wrong_robot else "completed")
