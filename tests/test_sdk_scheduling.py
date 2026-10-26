"""Bound transport work without dropping publication evidence or refreshing stale motors."""

import itertools
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

from humanoid_agent.runtime.sdk import (
    MotorReceiptPacket,
    PlannerPublicationReceipt,
    PlannerRelayInput,
    ReceivedMotorCommand,
    SonicMotorCommand,
    SonicSdkProcessBridge,
)
from humanoid_agent.runtime.sdk_worker import run_worker
from humanoid_agent.sonic_planner_input import NativePlannerSample


def neutral_sample() -> NativePlannerSample:
    return NativePlannerSample(
        locomotion_mode=0,
        movement_direction_world=[1, 0, 0],
        facing_direction_world=[1, 0, 0],
        speed_m_per_s=0,
        body_height_m=-1,
        vr_position_body_frame=[0.0] * 9,
        vr_orientation_body_frame_wxyz=[1, 0, 0, 0] * 3,
        endpoint_hold=True,
    )


def motor_command() -> SonicMotorCommand:
    return SonicMotorCommand(
        positions=[0.0] * 29,
        velocities=[0.0] * 29,
        feedforward_torques=[0.0] * 29,
        stiffness=[1.0] * 29,
        damping=[1.0] * 29,
    )


def publication_packet(sequence: int) -> bytes:
    return (
        b"planner-sent:"
        + PlannerPublicationReceipt(
            sequence_number=sequence,
            command_id="motion_a",
            sample=neutral_sample(),
            published_monotonic_seconds=float(sequence),
        )
        .model_dump_json()
        .encode()
    )


def bridge_fixture(packets: list[bytes]) -> tuple[SonicSdkProcessBridge, MagicMock, MagicMock]:
    inbox = deque(packets)
    connection = MagicMock()

    def receive(_maximum: int) -> bytes:
        if not inbox:
            raise BlockingIOError
        return inbox.popleft()

    connection.recv.side_effect = receive
    bridge = object.__new__(SonicSdkProcessBridge)
    bridge.connection = connection
    bridge.process = MagicMock()
    bridge.process.poll.return_value = None
    bridge.controller_ready = False
    bridge.stop_requested = False
    bridge.stop_acknowledged = False
    bridge.received_planner_publications = 0
    bridge._latest_command = None
    bridge._last_motor_payload = None
    bridge._motor_busy_since = None
    motor_mailbox = MagicMock()
    motor_mailbox.read.return_value = None
    bridge.motor_mailbox = motor_mailbox
    bridge._planner_publications = []
    return bridge, connection, motor_mailbox


class SdkSchedulingTests(unittest.TestCase):
    def test_publication_backlog_is_bounded_without_hiding_fresh_motor_state(self) -> None:
        packets = [publication_packet(sequence) for sequence in range(1, 81)]
        packets.append(b"planner-stream-stopped:80")
        bridge, connection, motor_mailbox = bridge_fixture(packets)
        bridge.stop_requested = True
        motor_mailbox.read.return_value = (
            MotorReceiptPacket(command=motor_command(), received_monotonic_seconds=80)
            .model_dump_json()
            .encode()
        )
        receipt = bridge.latest_command()
        assert receipt is not None
        self.assertEqual(receipt.received_monotonic_seconds, 80)
        self.assertEqual(connection.recv.call_count, bridge.maximum_packets_per_poll)
        self.assertFalse(bridge.stop_acknowledged)
        while not bridge.stop_acknowledged:
            bridge.latest_command()
        self.assertEqual(
            [record.sequence_number for record in bridge.take_planner_publications()],
            list(range(1, 81)),
        )
        connection.send.assert_called_once_with(b"stop-confirmed")

    def test_invalid_motor_data_and_receipt_gaps_still_fail_closed(self) -> None:
        invalid = MotorReceiptPacket(
            command=motor_command(), received_monotonic_seconds=1
        ).model_dump(mode="json")
        invalid["command"]["positions"][0] = None
        bridge, _connection, motor_mailbox = bridge_fixture([])
        motor_mailbox.read.return_value = json.dumps(invalid).encode()
        with self.assertRaises(ValueError):
            bridge.latest_command()
        bridge, _connection, motor_mailbox = bridge_fixture([publication_packet(2)])
        with self.assertRaises(ValueError):
            bridge.latest_command()
        bridge, _connection, motor_mailbox = bridge_fixture([b"unexpected"])
        with self.assertRaises(ValueError):
            bridge.latest_command()

    def test_mailbox_contention_skips_tick_and_times_out_without_refreshing_source_age(
        self,
    ) -> None:
        bridge, _connection, motor_mailbox = bridge_fixture([])
        motor_mailbox.read.return_value = (
            MotorReceiptPacket(command=motor_command(), received_monotonic_seconds=7)
            .model_dump_json()
            .encode()
        )
        receipt = bridge.latest_command()
        assert receipt is not None
        self.assertEqual(receipt.received_monotonic_seconds, 7)
        motor_mailbox.read.side_effect = BlockingIOError
        with patch("humanoid_agent.runtime.sdk.time.monotonic", side_effect=[10, 10.5, 11.1]):
            self.assertIsNone(bridge.latest_command())
            self.assertIsNone(bridge.latest_command())
            with self.assertRaisesRegex(TimeoutError, "mailbox remained locked"):
                bridge.latest_command()
        motor_mailbox.read.side_effect = None
        self.assertEqual(bridge.latest_command(), receipt)
        self.assertIsNone(bridge._motor_busy_since)

    def test_sdk_forwards_new_planner_samples_immediately_and_retries_motor_backpressure(
        self,
    ) -> None:
        initial = PlannerRelayInput(command_id=None, sample=neutral_sample())
        updated = PlannerRelayInput(
            command_id="motion_a", sample=neutral_sample().model_copy(update={"body_height_m": 0.7})
        )
        steps = deque(
            [
                [b"planner:" + initial.model_dump_json().encode()],
                [],
                [b"planner:" + updated.model_dump_json().encode()],
                [],
                [b"stop", b"stop-confirmed"],
            ]
        )
        inbox: deque[bytes] = deque()
        connection = MagicMock()
        sent: list[bytes] = []

        def receive(_maximum: int) -> bytes:
            if not inbox:
                raise BlockingIOError
            return inbox.popleft()

        def send(packet: bytes) -> int:
            sent.append(packet)
            return len(packet)

        def select_ready(*_arguments: object) -> tuple[list[MagicMock], list[object], list[object]]:
            next_packets = steps.popleft()
            inbox.extend(next_packets)
            return ([connection] if next_packets else [], [], [])

        connection.recv.side_effect = receive
        connection.send.side_effect = send
        sdk = MagicMock()
        sdk.latest_command.return_value = ReceivedMotorCommand(motor_command(), 7.0)
        native = MagicMock()
        native.ready.return_value = True
        zmq = MagicMock()
        motor_mailbox = MagicMock()
        motor_mailbox.write.side_effect = [False, True]
        ticks = itertools.count(100, 0.01)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("humanoid_agent.runtime.sdk_worker.socket.socket", return_value=connection),
            patch(
                "humanoid_agent.runtime.sdk_worker.LatestMotorMailbox", return_value=motor_mailbox
            ),
            patch(
                "humanoid_agent.runtime.sdk_worker.socket.if_nameindex", return_value=[(1, "lo")]
            ),
            patch("humanoid_agent.runtime.sdk_worker.subprocess.run"),
            patch("humanoid_agent.runtime.sdk_worker.SonicSdkBridge", return_value=sdk),
            patch("humanoid_agent.runtime.sdk_worker.NativeControllerSession", return_value=native),
            patch("humanoid_agent.runtime.sdk_worker.select.select", side_effect=select_ready),
            patch(
                "humanoid_agent.runtime.sdk_worker.time.monotonic", side_effect=lambda: next(ticks)
            ),
            patch("humanoid_agent.runtime.sdk_worker.publish_native_planner_sample") as publish,
            patch("humanoid_agent.runtime.sdk_worker.importlib.import_module", return_value=zmq),
        ):
            run_worker(
                10, Path(temporary), Path(temporary), motor_state_path=Path(temporary) / "motor"
            )
        self.assertEqual(publish.call_count, 2)
        receipts = [
            PlannerPublicationReceipt.model_validate_json(packet.removeprefix(b"planner-sent:"))
            for packet in sent
            if packet.startswith(b"planner-sent:")
        ]
        self.assertEqual([receipt.sequence_number for receipt in receipts], [1, 2])
        self.assertEqual(receipts[-1].sample, updated.sample)
        self.assertIn(b"planner-stream-stopped:2", sent)
        motors = [
            MotorReceiptPacket.model_validate_json(packet)
            for (packet,), _keywords in motor_mailbox.write.call_args_list
        ]
        self.assertEqual(len(motors), 2)
        self.assertEqual(motors[0], motors[1])
        self.assertFalse(any(packet.startswith(b"{") for packet in sent))
        motor_mailbox.close.assert_called_once()
        self.assertEqual(motors[0].received_monotonic_seconds, 7.0)
        native.close.assert_called_once()
        sdk.close.assert_called_once()
