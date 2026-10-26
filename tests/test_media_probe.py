"""Media readers share stream selection and preserve distinct duration clocks."""

import json
import subprocess
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from humanoid_agent.presentation.media_probe import probe_video


class MediaProbeTests(unittest.TestCase):
    def test_video_metadata_keeps_stream_and_container_duration_distinct(self) -> None:
        metadata = {
            "streams": [
                {
                    "nb_frames": "75",
                    "r_frame_rate": "25/2",
                    "duration": "6.000000",
                    "width": 1440,
                    "height": 1080,
                }
            ],
            "format": {"duration": "6.080000"},
        }
        with patch(
            "humanoid_agent.presentation.media_probe.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(metadata)),
        ) as run_probe:
            probe = probe_video(Path("recorded video.mp4"))
        self.assertEqual(probe.video.frame_count, 75)
        self.assertEqual(probe.video.frame_rate, Fraction(25, 2))
        self.assertEqual(probe.video.duration_seconds, 6)
        self.assertEqual(float(probe.format.duration), 6.08)
        self.assertEqual((probe.video.width, probe.video.height), (1440, 1080))
        arguments = run_probe.call_args.args[0]
        self.assertEqual(arguments[arguments.index("-select_streams") + 1], "v:0")
        self.assertEqual(arguments[-1], "recorded video.mp4")
        self.assertEqual(run_probe.call_args.kwargs["timeout"], 30)
        self.assertTrue(run_probe.call_args.kwargs["check"])

    def test_missing_video_is_rejected(self) -> None:
        with (
            patch(
                "humanoid_agent.presentation.media_probe.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout='{"streams": [], "format": {"duration": "6"}}'
                ),
            ),
            self.assertRaises(ValidationError),
        ):
            probe_video(Path("audio-only.mp4"))

    def test_container_duration_cannot_substitute_for_missing_stream_duration(self) -> None:
        with patch(
            "humanoid_agent.presentation.media_probe.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"nb_frames": "75", "r_frame_rate": "25/2"}],
                        "format": {"duration": "6"},
                    }
                ),
            ),
        ):
            probe = probe_video(Path("container-duration-only.mp4"))
        with self.assertRaisesRegex(ValueError, "no recorded duration"):
            _ = probe.video.duration_seconds
