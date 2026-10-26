"""Named motor ordering must address the same physical joints in both directions."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import IO
from unittest.mock import patch

import numpy as np

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.process_lifecycle import process_identity, process_record
from humanoid_agent.runtime.sdk import SonicSdkProcessBridge, motor_order_indices
from humanoid_agent.sonic_planner_input import NativePlannerSample


class SdkMotorOrderingTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "SDK transport and process records require Linux")
    def test_shutdown_cleans_recorded_controller_after_worker_was_killed(self) -> None:
        messenger = """
import socket,sys,time
connection=socket.socket(fileno=int(sys.argv[1]))
connection.send(b'ready')
time.sleep(60)
"""
        launch_process = subprocess.Popen

        def launch_messenger(
            command: list[str],
            *,
            pass_fds: tuple[int, ...],
            stdout: IO[str],
            stderr: int,
            start_new_session: bool,
        ) -> subprocess.Popen[bytes]:
            descriptor = command[command.index("--socket-fd") + 1]
            return launch_process(
                [sys.executable, "-c", messenger, descriptor],
                pass_fds=pass_fds,
                stdout=stdout,
                stderr=stderr,
                start_new_session=start_new_session,
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            controller = launch_process(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            bridge = None
            try:
                atomic_json(output / "controller-process.json", process_record(controller.pid))
                with patch(
                    "humanoid_agent.runtime.sdk.subprocess.Popen", side_effect=launch_messenger
                ):
                    bridge = SonicSdkProcessBridge(Path(sys.executable), output)
                bridge.process.kill()
                bridge.process.wait(timeout=5)
                with self.assertRaisesRegex(RuntimeError, "without a complete shutdown handshake"):
                    bridge.close()
                controller.wait(timeout=5)
                self.assertIsNone(process_identity(controller.pid))
            finally:
                if bridge is not None:
                    bridge.close()
                if controller.poll() is None:
                    controller.kill()
                controller.wait(timeout=5)

    def test_encoder_gather_and_command_scatter_preserve_physical_joint_identity(self) -> None:
        motor_names = [f"physical_joint_{index}" for index in range(29)]
        articulation_names = motor_names[::2] + motor_names[1::2]
        indices = motor_order_indices(articulation_names, motor_names)
        encoder_values = np.asarray([float(name.rsplit("_", 1)[1]) for name in articulation_names])
        np.testing.assert_array_equal(encoder_values[indices], np.arange(29))
        command_in_motor_order = np.zeros(29)
        command_in_motor_order[17] = 0.25
        command_in_articulation_order = command_in_motor_order[np.argsort(indices)]
        self.assertEqual(
            command_in_articulation_order[articulation_names.index(motor_names[17])], 0.25
        )
        self.assertEqual(np.count_nonzero(command_in_articulation_order), 1)
        for invalid_names in [
            articulation_names[:-1],
            [*articulation_names[:-1], "unrecognized_joint"],
            [*articulation_names[:-1], articulation_names[0]],
        ]:
            with self.subTest(names=invalid_names), self.assertRaises(ValueError):
                motor_order_indices(invalid_names, motor_names)

    @unittest.skipUnless(
        sys.platform == "linux", "SDK transport requires Linux SOCK_SEQPACKET and process records"
    )
    def test_shutdown_drains_real_socket_backpressure_and_rejects_incomplete_evidence(self) -> None:
        # The local test has no vendor DDS runtime. Replace only the launched
        # process boundary; exercise the real bridge, socket, parser and cleanup.
        messenger = """
import json,socket,sys,time
connection=socket.socket(fileno=int(sys.argv[1]))
connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
connection.send(b'ready')
case=sys.argv[2]
sample={'locomotion_mode':0,'movement_direction_world':[1,0,0],'facing_direction_world':[1,0,0],'speed_m_per_s':0,'body_height_m':-1,'vr_position_body_frame':[0.0]*9,'vr_orientation_body_frame_wxyz':[1,0,0,0]*3,'endpoint_hold':True}
for sequence in range(1,241):
    if case=='gap' and sequence==2:continue
    sample['vr_position_body_frame'][0]=sequence/1000
    record={'sequence_number':sequence,'command_id':'motion_a','sample':sample,'published_monotonic_seconds':time.monotonic()}
    connection.send(b'planner-sent:'+json.dumps(record).encode())
if connection.recv(65537)!=b'stop':raise RuntimeError('Expected producer stop')
connection.send(b'planner-stream-stopped:'+str(239 if case=='count' else 240).encode())
if connection.recv(65537)!=b'stop-confirmed':raise RuntimeError('Expected evidence confirmation')
connection.close()
"""
        launch_process = subprocess.Popen
        for case in ("complete", "gap", "count"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:

                def launch_messenger(
                    command: list[str],
                    *,
                    pass_fds: tuple[int, ...],
                    stdout: IO[str],
                    stderr: int,
                    start_new_session: bool,
                    scenario: str = case,
                ) -> subprocess.Popen[bytes]:
                    descriptor = command[command.index("--socket-fd") + 1]
                    return launch_process(
                        [sys.executable, "-c", messenger, descriptor, scenario],
                        pass_fds=pass_fds,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=start_new_session,
                    )

                with patch(
                    "humanoid_agent.runtime.sdk.subprocess.Popen", side_effect=launch_messenger
                ):
                    bridge = SonicSdkProcessBridge(Path(sys.executable), Path(temporary))
                try:
                    if case == "complete":
                        bridge.close()
                        publications = bridge.take_planner_publications()
                        self.assertEqual(
                            [publication.sequence_number for publication in publications],
                            list(range(1, 241)),
                        )
                        sample = publications[-1].sample
                        assert isinstance(sample, NativePlannerSample)
                        self.assertEqual(sample.vr_position_body_frame[0], 0.24)
                    else:
                        with self.assertRaisesRegex(
                            (ValueError, RuntimeError), "gap|all planner publication evidence"
                        ):
                            bridge.close()
                    self.assertIsNotNone(bridge.process.poll())
                finally:
                    bridge.close()


if __name__ == "__main__":
    unittest.main()
