"""Scripted controller checks select production ownership without model operators."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from humanoid_agent.cli import EpisodeOptions
from humanoid_agent.runtime.diagnostic import run_native_controller_check, scripted_episode_options
from humanoid_agent.runtime.shutdown_reference import SHUTDOWN_REFERENCE_DURATION_SECONDS
from humanoid_agent.runtime.stepped_turn import TURN_CHECK_DURATION_SECONDS

STORAGE_SCENE = Path("configs/shutdown-storage-none-open.json")


class NativeDiagnosticCliTests(unittest.TestCase):
    def test_each_motion_sets_its_own_workload_in_the_storage_scene(self) -> None:
        for motion, wall_seconds in (
            ("standing", 5),
            ("turn-check", TURN_CHECK_DURATION_SECONDS),
            ("standard", SHUTDOWN_REFERENCE_DURATION_SECONDS),
        ):
            options = EpisodeOptions(
                output=Path("attempt"),
                task_scene=STORAGE_SCENE,
                diagnostic_motion=motion,
                diagnostic_phase_seconds=5,
            )
            configured = scripted_episode_options(options)
            self.assertEqual(configured.wall_seconds, wall_seconds)
            self.assertEqual(configured.episode_kind, "controller-check")

    def test_alias_configures_the_production_episode_owner(self) -> None:
        options = EpisodeOptions(
            output=Path("attempt"),
            task_scene=STORAGE_SCENE,
            diagnostic_phase_seconds=2,
        )
        with patch("humanoid_agent.runtime.episode.run_episode") as run_episode:
            run_native_controller_check(options)
        configured = run_episode.call_args.args[0]
        self.assertEqual(configured.episode_kind, "controller-check")
        self.assertEqual(configured.wall_seconds, SHUTDOWN_REFERENCE_DURATION_SECONDS)
        self.assertEqual(configured.task_scene, options.task_scene)
        self.assertEqual(configured.output, options.output)
        self.assertEqual(options.episode_kind, "model")

    def test_production_owner_launches_only_its_physics_worker_for_scripted_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = EpisodeOptions(output=root / "attempt", task_scene=STORAGE_SCENE)
            prepared = SimpleNamespace(
                source=root / "source",
                constants=root / "constants.json",
                robot_urdf=root / "robot.urdf",
                scene=root / "scene.json",
                decorations=None,
                protocol=None,
            )
            with (
                patch("humanoid_agent.runtime.group.prepare_episode_inputs", return_value=prepared),
                patch(
                    "humanoid_agent.runtime.group.create_host_process_environment",
                    return_value={"PYTHONPATH": "source"},
                ),
                patch("humanoid_agent.runtime.group.preflight_episode"),
                patch("humanoid_agent.runtime.group.model_operator_command") as operator_command,
                patch(
                    "humanoid_agent.runtime.group.subprocess.Popen",
                    side_effect=RuntimeError("injected launch failure"),
                ) as launch,
                self.assertRaisesRegex(RuntimeError, "injected launch failure"),
            ):
                run_native_controller_check(options)
            operator_command.assert_not_called()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--role") + 1], "physics")
            self.assertEqual(command[command.index("--episode-kind") + 1], "controller-check")
            self.assertEqual(
                command[command.index("--wall-seconds") + 1],
                str(float(SHUTDOWN_REFERENCE_DURATION_SECONDS)),
            )
            self.assertEqual(command[command.index("--diagnostic-camera") + 1], "off")
            terminal = json.loads((options.output / "episode.json").read_text())
            self.assertEqual(terminal["episode_kind"], "controller-check")
            self.assertEqual(terminal["status"], "infrastructure_error")
            self.assertFalse((options.output / "operator").exists())

    def test_invalid_workload_is_rejected_before_creating_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for overrides in (
                {"task_scene": None},
                {"diagnostic_phase_seconds": float("nan")},
                {"diagnostic_phase_seconds": 0.5},
                {"capture_interval_seconds": 0},
            ):
                options = EpisodeOptions(
                    output=Path(temporary) / "invalid",
                    task_scene=STORAGE_SCENE,
                )
                for name, value in overrides.items():
                    setattr(options, name, value)
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    run_native_controller_check(options)
                self.assertFalse(options.output.exists())
