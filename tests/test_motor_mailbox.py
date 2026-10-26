"""Latest motor state replaces history without torn snapshots or waiting on a writer."""

import fcntl
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid_agent.runtime.mailbox import LatestMotorMailbox
from humanoid_agent.runtime.sdk import SonicSdkProcessBridge


class MotorMailboxTests(unittest.TestCase):
    def test_newest_cross_process_state_replaces_stale_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motor-state"
            reader = LatestMotorMailbox(path, create=True)
            try:
                self.assertIsNone(reader.read())
                # No vendor SDK, controller or GPU is involved in this producer.
                program = """
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from humanoid_agent.runtime.mailbox import LatestMotorMailbox
from humanoid_agent.runtime.sdk import SonicSdkProcessBridge
mailbox = LatestMotorMailbox(Path(sys.argv[1]))
try:
    for sequence in range(1000):
        assert mailbox.write(str(sequence).encode())
finally:
    mailbox.close()
"""
                subprocess.run([sys.executable, "-c", program, str(path)], check=True, timeout=10)
                self.assertEqual(reader.read(), b"999")
                self.assertEqual(reader.read(), b"999")
                writer = LatestMotorMailbox(path)
                try:
                    writer.write(b"7")
                    self.assertEqual(reader.read(), b"7")
                finally:
                    writer.close()
            finally:
                reader.close()

    def test_interrupted_payload_copy_leaves_no_partial_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motor-state"
            reader = LatestMotorMailbox(path, create=True)
            writer = LatestMotorMailbox(path)
            try:
                writer.write(b"previous command")
                original_memory = writer.memory
                failing_memory = MagicMock()

                def interrupt_payload_copy(region: slice, payload: bytes) -> None:
                    if region.start == writer.payload_length.size:
                        # Model a writer disappearing after publishing only a prefix.
                        original_memory[region.start : region.start + 3] = payload[:3]
                        raise RuntimeError("interrupted payload copy")
                    original_memory[region] = payload

                failing_memory.__setitem__.side_effect = interrupt_payload_copy
                with (
                    patch.object(writer, "memory", failing_memory),
                    self.assertRaisesRegex(RuntimeError, "interrupted payload copy"),
                ):
                    writer.write(b"replacement command")
                self.assertIsNone(reader.read())
                self.assertTrue(writer.write(b"recovered command"))
                self.assertEqual(reader.read(), b"recovered command")
            finally:
                writer.close()
                reader.close()

    def test_failed_bridge_setup_releases_mailbox_and_transport(self) -> None:
        parent_connection, worker_connection = socket.socketpair()
        bridge = object.__new__(SonicSdkProcessBridge)
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch(
                    "humanoid_agent.runtime.sdk.socket.socketpair",
                    return_value=(parent_connection, worker_connection),
                ),
                self.assertRaises(FileNotFoundError),
            ):
                bridge.__init__(Path(sys.executable), Path(temporary) / "missing")
            self.assertTrue(bridge.motor_mailbox.memory.closed)
            self.assertFalse(Path(bridge._motor_directory.name).exists())
            self.assertEqual(parent_connection.fileno(), -1)
            self.assertEqual(worker_connection.fileno(), -1)

    def test_contention_is_nonblocking_and_does_not_publish_partial_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motor-state"
            reader = LatestMotorMailbox(path, create=True)
            writer = LatestMotorMailbox(path)
            try:
                self.assertTrue(writer.write(b"old"))
                fcntl.flock(writer.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    with self.assertRaises(BlockingIOError):
                        reader.read()
                finally:
                    fcntl.flock(writer.descriptor, fcntl.LOCK_UN)
                fcntl.flock(reader.descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                try:
                    self.assertFalse(writer.write(b"new"))
                finally:
                    fcntl.flock(reader.descriptor, fcntl.LOCK_UN)
                self.assertEqual(reader.read(), b"old")
                self.assertTrue(writer.write(b"new"))
                self.assertEqual(reader.read(), b"new")
                for payload in (b"", b"x" * (writer.maximum_payload_bytes + 1)):
                    with self.assertRaises(ValueError):
                        writer.write(payload)
                self.assertEqual(reader.read(), b"new")
            finally:
                writer.close()
                reader.close()
