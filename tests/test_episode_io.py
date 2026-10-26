"""The physics producer never waits for storage and incomplete evidence cannot pass."""

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Thread

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.recording import iter_records
from humanoid_agent.runtime.episode_io import (
    AppendRecord,
    BeginEpisode,
    EpisodeIO,
    ObservationReady,
    WriteRecord,
)
from humanoid_agent.runtime.operator_transport import OperatorClient, QueryOperatorState


@unittest.skipUnless(
    sys.platform == "linux", "Episode I/O workers require Linux CPU affinity and process records"
)
class EpisodeIOTests(unittest.TestCase):
    def test_paused_storage_does_not_block_producer_and_backpressure_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "commands"
            commands.mkdir()
            worker = EpisodeIO(root, commands, capacity=2)
            if worker.process.pid is None:
                self.fail("Worker did not start")
            os.kill(worker.process.pid, signal.SIGSTOP)
            completed = Event()
            failures: list[Exception] = []

            def submit() -> None:
                try:
                    for index in range(3):
                        worker.submit(AppendRecord("physical-trace.jsonl", {"tick": index}))
                except Exception as error:
                    failures.append(error)
                finally:
                    completed.set()

            producer = Thread(target=submit)
            producer.start()
            try:
                self.assertTrue(
                    completed.wait(timeout=2), "Physics producer waited for stopped storage"
                )
                self.assertEqual(len(failures), 1)
                self.assertIn("queue is full", str(failures[0]))
            finally:
                os.kill(worker.process.pid, signal.SIGCONT)
                producer.join(timeout=5)
                with self.assertRaisesRegex(RuntimeError, "recording failed"):
                    worker.close()
            recorded = list(iter_records(root / "physical-trace.jsonl"))
            self.assertEqual(recorded, [{"tick": 0}, {"tick": 1}])

    def test_shutdown_drains_ordered_evidence_and_writer_failure_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "commands"
            commands.mkdir()
            worker = EpisodeIO(root, commands)
            for index in range(20):
                worker.submit(AppendRecord("executed-inputs.jsonl", {"command": index}))
            worker.close()
            recorded = list(iter_records(root / "executed-inputs.jsonl"))
            self.assertEqual(recorded, [{"command": index} for index in range(20)])
            summary = json.loads((root / "io-summary.json").read_text())
            self.assertEqual(summary["operations_submitted"], summary["operations_processed"])
            failed_root = root / "failed"
            failed_root.mkdir()
            failed_commands = failed_root / "commands"
            failed_commands.mkdir()
            failed = EpisodeIO(failed_root, failed_commands)
            failed.submit(WriteRecord(failed_root / "invalid.json", {"lag": float("nan")}))
            with self.assertRaisesRegex(RuntimeError, "recording failed"):
                failed.close()
            self.assertEqual(
                json.loads((failed_root / "io-summary.json").read_text())["status"], "failed"
            )

    def test_observation_copy_preserves_bytes_and_the_absolute_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "commands"
            commands.mkdir()
            capture = root / "camera-output/initial"
            capture.mkdir(parents=True)
            for name, content in (
                ("onboard.png", b"RGB frame bytes"),
                ("left_wrist.png", b"left wrist RGB bytes"),
                ("right_wrist.png", b"right wrist RGB bytes"),
                ("depth-m.npy", b"depth bytes"),
                ("optical-points-m.npy", b"point bytes"),
            ):
                (capture / name).write_bytes(content)
            atomic_json(
                capture / "observation.json",
                {
                    "simulation_seconds": 0.0,
                    "elapsed_wall_seconds": 0.0,
                    "robot_sensors": {
                        "joint_names": ["example_joint"],
                        "joint_positions_rad": [0.0],
                        "joint_velocities_rad_per_s": [0.0],
                        "imu_angular_velocity_body_rad_per_s": [0, 0, 0],
                        "imu_gravity_direction_body": [0, 0, -1],
                    },
                    "wrist_camera_names": ["left_wrist", "right_wrist"],
                    "depth_tiles_m": [[None] * 16 for _ in range(12)],
                    "camera_intrinsic_matrix": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                    "observation_identity": {
                        "episode_id": "episode_a",
                        "identifier": "initial",
                        "captured_elapsed_wall_seconds": 0.0,
                        "episode_deadline_elapsed_wall_seconds": 10.0,
                    },
                },
            )
            worker = EpisodeIO(root, commands)
            deadline = time.monotonic() + 10
            clock = EpisodeClock(
                deadline_monotonic_seconds=deadline,
                episode_id="episode_a",
                elapsed_wall_seconds=0,
                deadline_elapsed_wall_seconds=10,
            )
            try:
                worker.submit(BeginEpisode(clock, {"wall_seconds": 0, "simulation_seconds": 0}))
                received = []
                until = time.monotonic() + 5
                while not received and time.monotonic() < until:
                    received.extend(worker.events())
                    time.sleep(0.01)
                self.assertTrue(
                    any(
                        isinstance(event, ObservationReady) and event.identifier == "initial"
                        for event in received
                    )
                )
                self.assertEqual(
                    (root / "observations/initial/rgb.png").read_bytes(), b"RGB frame bytes"
                )
                for side in ("left", "right"):
                    self.assertEqual(
                        (root / f"observations/initial/{side}_wrist.png").read_bytes(),
                        (capture / f"{side}_wrist.png").read_bytes(),
                    )
                state = OperatorClient(commands).request(QueryOperatorState(), time.monotonic() + 1)
                assert state is not None and state.clock is not None
                self.assertEqual(state.clock.deadline_monotonic_seconds, deadline)
            finally:
                worker.close()
