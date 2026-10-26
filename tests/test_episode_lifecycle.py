"""Keep startup failures, component failures and descendant cleanup observable."""

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.cli import EpisodeOptions
from humanoid_agent.model_api_configuration import ApiServiceTier, ReasoningEffort
from humanoid_agent.process_lifecycle import process_identity, process_record
from humanoid_agent.runtime.episode import (
    interrupt_pending_attempts,
    preflight_episode,
    reconcile_execution,
    require_episode_storage,
    run_episode,
    stop_process,
)
from humanoid_agent.runtime.operator import run_isaac_model_operator


class EpisodeLifecycleTests(unittest.TestCase):
    def test_insufficient_evidence_storage_is_rejected_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            unavailable_margin = shutil.disk_usage(output).total + 1
            with self.assertRaisesRegex(OSError, "Insufficient episode evidence storage"):
                require_episode_storage(output, minimum_free_bytes=unavailable_margin)
            record = json.loads((output / "storage-preflight.json").read_text())
            self.assertFalse(record["passed"])
            self.assertLess(record["available_bytes"], record["minimum_free_bytes"])
            require_episode_storage(output, minimum_free_bytes=1)
            self.assertTrue(json.loads((output / "storage-preflight.json").read_text())["passed"])

    def test_missing_startup_asset_has_terminal_record_without_started_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = EpisodeOptions(
                output=root / "episode",
                task_scene=root / "scene.json",
                sonic_constants=root / "missing.json",
            )
            with self.assertRaises(FileNotFoundError):
                run_episode(arguments)
            terminal = json.loads((arguments.output / "episode.json").read_text())
            self.assertEqual(terminal["status"], "infrastructure_error")
            self.assertEqual(terminal["component_exit_codes"], {})
            self.assertIsNone(terminal["task_outcome"])

    def test_lfs_pointer_is_rejected_before_loading_a_physics_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "isaac"
            mesh = workspace / "SONIC/gear_sonic/data/assets/robot_description/meshes/g1/pelvis.STL"
            mesh.parent.mkdir(parents=True)
            mesh.write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:"
                + "0" * 64
                + "\nsize 1000\n"
            )
            constants = root / "constants.json"
            atomic_json(
                constants,
                {
                    "default_angles": [0.0] * 29,
                    "action_scale": [1.0] * 29,
                    "kp": [1.0] * 29,
                    "kd": [0.0] * 29,
                },
            )
            robot_urdf = root / "robot.urdf"
            robot_urdf.write_text(
                '<robot name="preflight-fixture"><link name="pelvis"><visual><geometry>'
                '<mesh filename="package://robot_description/meshes/g1/pelvis.STL"/>'
                "</geometry></visual></link></robot>"
            )
            with (
                patch(
                    "humanoid_agent.canonical_robot.CANONICAL_URDF_SHA256",
                    hashlib.sha256(robot_urdf.read_bytes()).hexdigest(),
                ),
                self.assertRaisesRegex(ValueError, "Git LFS pointer"),
            ):
                preflight_episode(
                    SimpleNamespace(isaac_workspace=workspace),
                    root,
                    root,
                    {"SONIC_CONSTANTS": str(constants), "SONIC_ROBOT_URDF": str(robot_urdf)},
                )
            self.assertFalse((root / "preflight-physics.log").exists())

    def test_missing_credentials_are_recorded_by_operator_before_api_or_kinematics(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
        ):
            root = Path(directory)
            arguments = SimpleNamespace(
                output=root / "operator",
                mode="hybrid",
                external_camera=True,
                physics_directory=root,
                command_directory=root,
                facility_map=None,
                stepped_turns=False,
                leashed_facing=False,
                task_instruction="Close any open drawer, return home and declare completion.",
                provider="sol",
                api_service_tier=ApiServiceTier.STANDARD,
                api_request_timeout_seconds=10,
                reasoning_effort=ReasoningEffort.LOW,
            )
            with self.assertRaisesRegex(ValueError, "Empty API credential"):
                run_isaac_model_operator(arguments)
            self.assertEqual(
                json.loads((arguments.output / "operator-error.json").read_text())[
                    "exception_type"
                ],
                "ValueError",
            )
            self.assertEqual(
                json.loads((arguments.output / "operator-summary.json").read_text())[
                    "motion_decisions"
                ],
                0,
            )

    def test_physics_success_does_not_hide_operator_failure_or_missing_outcome(self) -> None:
        for physics, operator, error, interrupted, outcome, success in [
            (0, 0, None, False, True, True),
            (0, 1, None, False, True, False),
            (0, 1, None, True, True, False),
            (0, 0, {"exception_type": "ValueError"}, False, True, False),
            (0, -2, {"exception_type": "KeyboardInterrupt"}, True, True, True),
            (0, 1, {"exception_type": "RuntimeError"}, True, True, False),
            (1, 0, None, False, True, False),
            (0, None, None, False, False, False),
        ]:
            with self.subTest(physics=physics, operator=operator, error=error):
                self.assertEqual(
                    reconcile_execution(physics, operator, error, interrupted, outcome) is None,
                    success,
                )

    def test_pending_api_attempt_is_interrupted_without_overwriting_received_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "pending.timing.json", {"state": "started", "request_number": 1})
            received = {"state": "received", "request_number": 2}
            atomic_json(root / "received.timing.json", received)
            interrupt_pending_attempts(root)
            pending = json.loads((root / "pending.timing.json").read_text())
            self.assertEqual(pending["state"], "interrupted")
            self.assertFalse(pending["provider_completion_known"])
            self.assertEqual(json.loads((root / "received.timing.json").read_text()), received)

    @unittest.skipUnless(
        sys.platform == "linux", "Process identity and cleanup require Linux boot IDs"
    )
    def test_owner_cleanup_stops_grandchild_in_another_session(self) -> None:
        script = (
            "import os,signal,subprocess,sys,time; "
            "signal.signal(signal.SIGINT,lambda *_:sys.exit(0)); "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
            "print(child.pid,flush=True); time.sleep(60)"
        )
        owner = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.assertIsNotNone(owner.stdout)
        if owner.stdout is None:
            self.fail("Owner pipe was not created")
        child_pid = int(owner.stdout.readline())
        try:
            stop_process(owner)
            child = process_identity(child_pid)
            self.assertIsNone(child)
            self.assertIsNotNone(owner.poll())
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
            owner.stdout.close()

    @unittest.skipUnless(
        sys.platform == "linux", "Process identity and cleanup require Linux boot IDs"
    )
    def test_recorded_orphan_is_cleaned_after_owner_has_already_exited(self) -> None:
        script = "import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); print(child.pid,flush=True)"
        with tempfile.TemporaryDirectory() as temporary_directory:
            owner = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE)
            if owner.stdout is None:
                self.fail("Owner pipe was not created")
            child_pid = int(owner.stdout.readline())
            owner.wait(timeout=10)
            try:
                records = Path(temporary_directory)
                controller_records = records / "robot_1"
                controller_records.mkdir()
                atomic_json(
                    controller_records / "controller-process.json", process_record(child_pid)
                )
                stop_process(owner, records)
                self.assertIsNone(process_identity(child_pid))
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(child_pid, signal.SIGKILL)
                owner.stdout.close()

    @unittest.skipUnless(
        sys.platform == "linux", "Process identity and cleanup require Linux boot IDs"
    )
    def test_stale_identity_does_not_kill_an_unowned_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
            owner = subprocess.Popen([sys.executable, "-c", "pass"])
            owner.wait(timeout=10)
            try:
                record = process_record(child.pid)
                record["created_time_seconds"] = float(str(record["created_time_seconds"])) + 1
                records = Path(temporary_directory)
                atomic_json(records / "camera-worker-process.json", record)
                stop_process(owner, records)
                self.assertIsNone(child.poll())
            finally:
                child.kill()
                child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
