"""Shared episodes discard terminal requests and replay only admitted evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid_agent.canonical_robot import JOINT_NAMES
from humanoid_agent.runtime.actor import RobotEpisode
from humanoid_agent.runtime.camera import NativeVisualState
from humanoid_agent.runtime.episode_io import CommandReceived, ObservationRequested
from humanoid_agent.shutdown_task import make_storage_round_task
from humanoid_agent.sonic_planner_input import (
    NativePlannerSample,
)
from humanoid_agent.task_outcome import EpisodeEndReason


class SharedRuntimeTests(unittest.TestCase):
    def test_finishing_an_episode_discards_its_queued_camera_request(self) -> None:
        initial = NativeVisualState(
            identifier="initial",
            episode_id="episode_a",
            episode_deadline_elapsed_wall_seconds=60,
            simulation_seconds=0,
            elapsed_wall_seconds=0,
            source_monotonic_seconds=100,
            joint_names=list(JOINT_NAMES),
            joint_positions=[0.0] * 29,
            joint_velocities=[0.0] * 29,
            root_state_world=[0, 0, 0.76, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        neutral = NativePlannerSample(
            locomotion_mode=0,
            movement_direction_world=[1, 0, 0],
            facing_direction_world=[1, 0, 0],
            speed_m_per_s=0,
            body_height_m=-1,
            vr_position_body_frame=[0.0] * 9,
            vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
            endpoint_hold=True,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("humanoid_agent.runtime.actor.EpisodeIO") as io_factory,
        ):
            actor = RobotEpisode(
                Path(temporary), initial, neutral, [0, 0, 0], make_storage_round_task("none-open")
            )
            for event in (
                CommandReceived(MagicMock(), None, "unavailable"),
                ObservationRequested(identifier="unexpected"),
            ):
                io_factory.return_value.events.return_value = [event]
                with (
                    self.subTest(event=type(event).__name__),
                    self.assertRaisesRegex(ValueError, "Scripted controller checks"),
                ):
                    actor.poll(
                        0,
                        lambda: self.fail("Rejected ingress must not require task measurements"),
                        accept_model_commands=False,
                    )
            actor.requested_capture = "feedback_1"
            actor.finish_reason = EpisodeEndReason.WALL_TIME_LIMIT
            actor.finish(60, 12000)
            self.assertIsNone(actor.requested_capture)
            self.assertEqual(actor.final_elapsed, 60)
            self.assertEqual(actor.final_tick, 12000)


if __name__ == "__main__":
    unittest.main()
