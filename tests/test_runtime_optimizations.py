"""CPU placement and snapshot ownership must survive batching optimizations."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from humanoid_agent.cli import EpisodeOptions
from humanoid_agent.runtime.cpu import CpuAllocation, parse_cpu_set
from humanoid_agent.runtime.episode import create_host_process_environment, model_operator_command
from humanoid_agent.runtime.state import TickSnapshotCache


class RuntimeOptimizationTests(unittest.TestCase):
    def test_owner_cpu_allocations_survive_child_environment_and_operator_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = EpisodeOptions(
                output=Path(temporary),
                external_camera=True,
                physics_cpu_set="1,3-4",
                operator_cpu_set="5",
                sdk_cpu_set="6,8",
                io_cpu_set="9",
                controller_cpu_set="10-11",
                camera_cpu_set="12-13",
            )
            expected = CpuAllocation("1,3-4", "5", "6,8", "9", "10-11", "12-13")
            environment = create_host_process_environment(arguments, Path(temporary) / "constants")
            self.assertEqual(CpuAllocation.from_environment(environment), expected)
            command = model_operator_command(arguments, arguments.output)
            self.assertEqual(command[:3], ["taskset", "-c", "5"])
            self.assertIn("--external-camera", command)
            self.assertEqual(parse_cpu_set(expected.physics), {1, 3, 4})
            self.assertEqual(parse_cpu_set(expected.sdk), {6, 8})
            self.assertEqual(CpuAllocation.from_environment({}), CpuAllocation())

    def test_invalid_cpu_masks_fail_before_launch(self) -> None:
        for invalid in ("", "-1", "3-1", "1,,2", "0-65536", "1;2", " 1", "1-"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CpuAllocation(physics=invalid)

    def test_snapshots_batch_missing_fields_and_refresh_after_every_physics_step(self) -> None:
        cache = TickSnapshotCache()
        roots = np.asarray([[1, 2, 3]], dtype=np.float32)
        joints = np.asarray([[4, 5]], dtype=np.float32)
        bodies = np.asarray([[[6, 7, 8]]], dtype=np.float32)
        with patch("humanoid_agent.runtime.state.copy_tensor_fields_to_cpu") as transfer:
            transfer.side_effect = lambda fields, **_options: {
                name: value.copy() for name, value in fields.items()
            }
            sensor = cache.read(0, {"roots": roots, "joints": joints})
            visual = cache.read(0, {"roots": roots, "joints": joints})
            sample_time = cache.sampled_monotonic_seconds(("roots", "joints"))
            self.assertEqual(transfer.call_count, 1)
            self.assertIs(sensor["roots"], visual["roots"])
            cache.read(0, {"roots": roots, "bodies": bodies})
            self.assertEqual(cache.sampled_monotonic_seconds(("roots", "joints")), sample_time)
            self.assertEqual(set(transfer.call_args.args[0]), {"bodies"})
            self.assertEqual(transfer.call_count, 2)
            roots += 10
            np.testing.assert_array_equal(sensor["roots"], [[1, 2, 3]])
            refreshed = cache.read(1, {"roots": roots, "joints": joints})
            self.assertEqual(transfer.call_count, 3)
            np.testing.assert_array_equal(refreshed["roots"], [[11, 12, 13]])
            self.assertFalse(np.shares_memory(sensor["roots"], refreshed["roots"]))
            cache.invalidate()
            cache.read(1, {"roots": roots})
            self.assertEqual(transfer.call_count, 4)
            cache.read(2, {})
            self.assertEqual(transfer.call_count, 4)
