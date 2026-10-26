"""Every supported controller launch verifies the production deployment pins and layout."""

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.runtime.controller import NativeControllerSession
from humanoid_agent.runtime.controller import (
    native_controller_arguments as runtime_controller_arguments,
)
from humanoid_agent.runtime.deployment import (
    SONIC_REVISION,
    native_controller_arguments,
    native_controller_environment,
)


class NativeDeploymentTests(unittest.TestCase):
    executable_bytes = b"fixture executable"

    def prepare_deployment(self, workspace: Path) -> Path:
        executable = workspace / "SONIC/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(self.executable_bytes)
        models = workspace / "deployment-models-v1"
        assets = (
            "sonic_v1_1/model_decoder.onnx",
            "sonic_v1_1/model_encoder.onnx",
            "sonic_v1_1/observation_config.yaml",
            "planner_sonic.onnx",
        )
        manifest = {}
        for relative_path in assets:
            asset = models / relative_path
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_text(relative_path)
            manifest[relative_path] = {"sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
        atomic_json(workspace / "deployment-assets-v1.json", {"files": manifest})
        return models / "planner/target_vel/V2/planner_sonic.onnx"

    def test_runtime_requires_prepared_link_and_uses_configured_cpus(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("humanoid_agent.runtime.deployment.subprocess.run") as revision_command,
            patch(
                "humanoid_agent.runtime.deployment.EXECUTABLE_SHA256",
                hashlib.sha256(self.executable_bytes).hexdigest(),
            ),
            patch.dict(os.environ, {"HUMANOID_AGENT_CONTROLLER_CPU_SET": "20-23"}),
        ):
            revision_command.return_value = subprocess.CompletedProcess(
                [], 0, stdout=SONIC_REVISION + "\n"
            )
            workspace = Path(temporary)
            planner = self.prepare_deployment(workspace)
            with self.assertRaisesRegex(ValueError, "verified asset"):
                runtime_controller_arguments(workspace, workspace / "output")
            self.assertFalse(planner.parent.exists())
            planner.parent.mkdir(parents=True)
            planner.symlink_to(workspace / "deployment-models-v1/planner_sonic.onnx")
            runtime_arguments = runtime_controller_arguments(workspace, workspace / "output")
            self.assertEqual(runtime_arguments[2], "20-23")
            for flag in ("--motor-kp-scale", "--motor-kd-scale"):
                self.assertEqual(runtime_arguments[runtime_arguments.index(flag) + 1], "4,10=1.5")
            self.assertEqual(
                runtime_arguments[runtime_arguments.index("--planner-file") + 1], str(planner)
            )

    def test_native_control_and_teleop_readiness_does_not_add_a_frozen_physics_delay(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "humanoid_agent.runtime.deployment.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout=SONIC_REVISION),
            ),
            patch(
                "humanoid_agent.runtime.deployment.EXECUTABLE_SHA256",
                hashlib.sha256(self.executable_bytes).hexdigest(),
            ),
            patch("humanoid_agent.runtime.controller.subprocess.Popen") as launch,
            patch(
                "humanoid_agent.process_lifecycle.process_record",
                return_value={"pid": 1, "created_time_seconds": 1.0, "boot_id": "fixture-boot"},
            ),
            patch("time.monotonic", return_value=100.0),
        ):
            workspace = Path(temporary)
            planner = self.prepare_deployment(workspace)
            planner.parent.mkdir(parents=True)
            planner.symlink_to(workspace / "deployment-models-v1/planner_sonic.onnx")
            output = workspace / "output"
            output.mkdir()
            process = MagicMock()
            process.pid = os.getpid()
            process.poll.return_value = None
            process.returncode = 0
            launch.return_value = process
            session = NativeControllerSession(workspace, output)
            try:
                log = output / "controller.log"
                log.write_text("transitioning to CONTROL state\nVR 3-point control enabled\n")
                self.assertFalse(session.ready())
                with log.open("a") as stream:
                    stream.write("Switched to mode 'teleop'\n")
                self.assertTrue(session.ready())
                self.assertTrue((output / "controller-ready.json").is_file())
            finally:
                session.close()

    def test_deployment_mismatches_are_rejected(self) -> None:
        for mismatch in ("revision", "executable", "asset", "planner"):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory() as temporary,
                patch("humanoid_agent.runtime.deployment.subprocess.run") as revision_command,
                patch(
                    "humanoid_agent.runtime.deployment.EXECUTABLE_SHA256",
                    hashlib.sha256(self.executable_bytes).hexdigest(),
                ),
            ):
                workspace = Path(temporary)
                planner = self.prepare_deployment(workspace)
                if mismatch != "planner":
                    planner.parent.mkdir(parents=True)
                    planner.symlink_to(workspace / "deployment-models-v1/planner_sonic.onnx")
                revision_command.return_value = subprocess.CompletedProcess(
                    [], 0, stdout="changed revision" if mismatch == "revision" else SONIC_REVISION
                )
                match mismatch:
                    case "executable":
                        executable = (
                            workspace / "SONIC/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
                        )
                        executable.write_text("changed executable")
                    case "asset":
                        (workspace / "deployment-models-v1/planner_sonic.onnx").write_text(
                            "changed"
                        )
                    case "planner":
                        planner.parent.mkdir(parents=True)
                        planner.write_text("another planner")
                with self.assertRaises(ValueError):
                    native_controller_arguments(
                        workspace,
                        workspace / "output",
                        cpu_set="6-9",
                    )

    def test_verified_planner_copy_cannot_replace_required_link(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "humanoid_agent.runtime.deployment.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout=SONIC_REVISION),
            ),
            patch(
                "humanoid_agent.runtime.deployment.EXECUTABLE_SHA256",
                hashlib.sha256(self.executable_bytes).hexdigest(),
            ),
        ):
            workspace = Path(temporary)
            planner = self.prepare_deployment(workspace)
            planner.parent.mkdir(parents=True)
            planner.write_bytes(
                (workspace / "deployment-models-v1/planner_sonic.onnx").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "verified asset"):
                runtime_controller_arguments(workspace, workspace / "output")

    def test_library_paths_prepend_pinned_libraries_and_preserve_environment(self) -> None:
        with patch.dict(
            os.environ, {"LD_LIBRARY_PATH": "/existing/lib", "HUMANOID_AGENT_TEST": "kept"}
        ):
            environment = native_controller_environment(Path("/workspace"))
        self.assertEqual(environment["HUMANOID_AGENT_TEST"], "kept")
        library_paths = environment["LD_LIBRARY_PATH"].split(os.pathsep)
        self.assertEqual(library_paths[-1], "/existing/lib")
        self.assertIn("/workspace/TensorRT-10.13.3.9/lib", library_paths)
