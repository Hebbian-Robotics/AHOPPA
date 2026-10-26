"""A camera-free native diagnostic must not inherit the initial camera timeout."""

import itertools
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.recording import record_exists
from humanoid_agent.runtime.episode_io import BeginEpisode, StopIO, _io_worker


class CameraFreeIOTests(unittest.TestCase):
    def test_clock_can_start_without_an_initial_camera_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            incoming: queue.Queue[object] = queue.Queue()
            outgoing: queue.Queue[object] = queue.Queue()
            incoming.put(
                BeginEpisode(
                    EpisodeClock(
                        deadline_monotonic_seconds=100,
                        episode_id="diagnostic",
                        elapsed_wall_seconds=0,
                        deadline_elapsed_wall_seconds=100,
                    ),
                    {"wall_seconds": 0, "simulation_seconds": 0},
                    capture_initial_observation=False,
                )
            )
            incoming.put(None)
            incoming.put(StopIO(1))
            with (
                patch("humanoid_agent.runtime.episode_io.sys.platform", "linux"),
                patch("humanoid_agent.runtime.episode_io.os.sched_setaffinity", create=True),
                patch(
                    "humanoid_agent.runtime.episode_io.time.monotonic",
                    side_effect=itertools.count(0, 35),
                ),
            ):
                _io_worker(output, output / "commands", incoming, outgoing, threading.Event())
            summary = json.loads((output / "io-summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertIsNone(summary["interrupted_capture"])
            self.assertTrue(outgoing.empty())
            self.assertTrue(record_exists(output / "episode-clock.json"))
