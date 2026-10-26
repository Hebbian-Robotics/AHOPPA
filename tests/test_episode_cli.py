"""Exercise the entry-point boundaries used by parent and worker subprocesses."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EpisodeCliTests(unittest.TestCase):
    def test_role_help_does_not_require_parent_options_or_initialize_hardware(self) -> None:
        for role, expected_option in (("sdk", "--socket-fd"), ("camera", "--watchdog-seconds")):
            with self.subTest(role=role):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "humanoid_agent",
                        "--role",
                        role,
                        "--help",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected_option, result.stdout)
                self.assertNotIn("--api-service-tier", result.stdout)

    def test_invalid_wall_budget_fails_before_creating_an_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humanoid_agent",
                    "--output",
                    str(output),
                    "--wall-seconds",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_group_startup_failure_records_both_explicit_provider_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "group"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humanoid_agent",
                    "--role",
                    "group",
                    "--providers",
                    "sol",
                    "opus",
                    "--task-scene",
                    str(Path(temporary) / "missing-scene.json"),
                    "--sonic-constants",
                    str(Path(temporary) / "missing-constants.json"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Unused Tokens", result.stderr)
            for index, provider in enumerate(("sol", "opus")):
                record = json.loads(
                    (output / "episodes" / f"env_{index}" / "episode.json").read_text()
                )
                self.assertEqual(record["provider"], provider)
                self.assertEqual(record["status"], "infrastructure_error")
                self.assertEqual(record["component_exit_codes"], {})
            self.assertEqual(
                json.loads((output / "group.json").read_text())["status"], "infrastructure_error"
            )
