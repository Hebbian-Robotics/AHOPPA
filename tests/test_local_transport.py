"""Real local sockets preserve typed commands, independent replies and bounded waits."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from types import TracebackType
from typing import Self

import zmq

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.command_admission import AdmissionReason, CommandAdmission
from humanoid_agent.runtime.local_transport import LocalPullReceiver, LocalPushSender
from humanoid_agent.runtime.operator import LocalOperatorConnection
from humanoid_agent.runtime.operator_transport import (
    OperatorClient,
    OperatorReply,
    OperatorRequest,
    OperatorServer,
    OperatorState,
    QueryOperatorState,
    SubmitCommand,
)
from humanoid_agent.runtime.records import CommandStatus, IdentifierReceipt
from humanoid_agent.sonic_planner_input import PlannerTrackingPose, SonicPlannerPublication


class RunningOperatorServer:
    """A real transport peer standing in for the independently scheduled physics owner."""

    def __init__(self, commands: Path) -> None:
        self.commands = commands
        self.state = OperatorState()
        self.requests: list[OperatorRequest] = []
        self.errors: list[BaseException] = []
        self.ready = Event()
        self.stopping = Event()
        self.thread = Thread(target=self.serve)

    def handle(self, request: OperatorRequest) -> OperatorState:
        self.requests.append(request)
        return self.state

    def serve(self) -> None:
        server = None
        try:
            server = OperatorServer(self.commands)
            self.ready.set()
            while not self.stopping.is_set():
                server.poll(self.handle)
                self.stopping.wait(0.001)
        except BaseException as error:
            self.errors.append(error)
            self.ready.set()
        finally:
            if server is not None:
                server.close()

    def __enter__(self) -> Self:
        self.thread.start()
        if not self.ready.wait(5) or self.errors:
            raise RuntimeError(f"Operator test peer failed startup: {self.errors}")
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stopping.set()
        self.thread.join(5)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError(f"Operator test peer did not shut down cleanly: {self.errors}")


def planner_publication() -> SonicPlannerPublication:
    return SonicPlannerPublication(
        identifier="motion_001",
        episode_id="episode_a",
        observation_id="initial",
        source_response_sha256="a" * 64,
        observation_elapsed_wall_seconds=0,
        valid_until_elapsed_wall_seconds=60,
        request_task_completion=False,
        control_ticks=50,
        locomotion_mode=0,
        movement_direction_heading_frame=[1, 0, 0],
        facing_yaw_radians=0,
        speed_m_per_s=0,
        body_height_m=-1,
        poses=[
            PlannerTrackingPose(
                positions_heading_frame=[0.0] * 9,
                orientations_heading_frame_wxyz=[1, 0, 0, 0] * 3,
            )
        ]
        * 4,
    )


class LocalTransportTests(unittest.TestCase):
    def test_delayed_command_acknowledgment_does_not_repeat_delivery(self) -> None:
        request_received = Event()
        release_reply = Event()

        class DelayedOperatorServer(RunningOperatorServer):
            def handle(self, request: OperatorRequest) -> OperatorState:
                state = super().handle(request)
                request_received.set()
                if not release_reply.wait(timeout=3):
                    raise TimeoutError("Delayed operator test reply was not released")
                return state

        with tempfile.TemporaryDirectory() as temporary:
            commands = Path(temporary) / "commands"
            with DelayedOperatorServer(commands) as server, ThreadPoolExecutor() as executor:
                request = SubmitCommand(publication=planner_publication())
                pending_reply = executor.submit(
                    OperatorClient(commands).request, request, time.monotonic() + 3
                )
                try:
                    self.assertTrue(request_received.wait(timeout=1))
                    # Hold the real peer beyond the client's former 200 ms retry.
                    self.assertFalse(release_reply.wait(timeout=0.3))
                finally:
                    release_reply.set()
                self.assertEqual(pending_reply.result(timeout=2), server.state)
                self.assertEqual(server.requests, [request])

    def test_in_flight_request_still_expires_at_its_caller_deadline(self) -> None:
        class WaitingOperatorServer(RunningOperatorServer):
            def handle(self, request: OperatorRequest) -> OperatorState:
                state = super().handle(request)
                self.stopping.wait(timeout=2)
                return state

        with tempfile.TemporaryDirectory() as temporary:
            commands = Path(temporary) / "commands"
            with WaitingOperatorServer(commands):
                started = time.monotonic()
                reply = OperatorClient(commands).request(
                    SubmitCommand(publication=planner_publication()), started + 0.3
                )
                self.assertIsNone(reply)
                self.assertLess(time.monotonic() - started, 1)

    def test_reply_boundary_rejects_missing_or_conflicting_outcomes(self) -> None:
        for payload in ({}, {"state": {}, "error": "rejected"}):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "exactly one"):
                OperatorReply.model_validate(payload)
        self.assertEqual(OperatorReply(state=OperatorState()).state, OperatorState())
        self.assertEqual(OperatorReply(error="rejected").error, "rejected")

    def test_command_delivery_and_execution_acknowledgment_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with RunningOperatorServer(root / "commands") as server:
                connection = LocalOperatorConnection(root / "physics", root / "commands")
                command = planner_publication()
                connection.publish_command(command)
                submitted = server.requests[-1]
                self.assertIsInstance(submitted, SubmitCommand)
                assert isinstance(submitted, SubmitCommand)
                self.assertEqual(submitted.publication, command)
                self.assertFalse(
                    connection.wait_identifier(
                        "motion", command.identifier, time.monotonic() + 0.05
                    )
                )
                server.state = OperatorState(
                    motion_finished=IdentifierReceipt(identifier=command.identifier)
                )
                self.assertTrue(
                    connection.wait_identifier("motion", command.identifier, time.monotonic() + 1)
                )
                server.state = OperatorState(finished=True)
                self.assertFalse(
                    connection.wait_identifier("motion", command.identifier, time.monotonic() + 1)
                )
                self.assertFalse((root / "commands/command.json").exists())

    def test_rejected_motion_cannot_be_reported_as_finished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with RunningOperatorServer(root) as server:
                server.state = OperatorState(
                    motion_finished=IdentifierReceipt(identifier="motion_001"),
                    command_status=CommandStatus(
                        identifier="motion_001",
                        state="rejected",
                        admission=CommandAdmission(
                            identifier="motion_001",
                            reason=AdmissionReason.WRONG_EPISODE,
                            payload_sha256="a" * 64,
                            observed_elapsed_wall_seconds=1,
                            observation_age_seconds=None,
                        ),
                    ),
                )
                connection = LocalOperatorConnection(root, root)
                with self.assertRaisesRegex(ValueError, "wrong_episode"):
                    connection.wait_identifier("motion", "motion_001", time.monotonic() + 1)

    def test_malformed_commands_receive_rejection_and_leave_server_usable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with RunningOperatorServer(root) as server:
                client = OperatorClient(root)
                with zmq.Context.instance().socket(zmq.REQ) as connection:
                    connection.setsockopt(zmq.LINGER, 0)
                    connection.setsockopt(zmq.RCVTIMEO, 1000)
                    connection.connect(client.endpoint)
                    raw = planner_publication().model_dump(mode="json")
                    raw["control_ticks"] = "50"
                    connection.send_json({"kind": "command", "publication": raw})
                    reply = OperatorReply.model_validate_json(connection.recv())
                    self.assertIsNotNone(reply.error)
                self.assertFalse(server.requests)
                self.assertEqual(
                    client.request(QueryOperatorState(), time.monotonic() + 1), server.state
                )

    def test_absent_receiver_has_a_bounded_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = OperatorClient(Path(temporary))
            started = time.monotonic()
            self.assertIsNone(client.request(QueryOperatorState(), started + 0.05))
            self.assertLess(time.monotonic() - started, 0.5)

    def test_camera_transport_preserves_capture_order_before_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = Path(temporary) / "camera"
            receiver = LocalPullReceiver(identity)
            sender = LocalPushSender(identity)
            try:
                messages = [b'{"capture":"initial"}', b'{"capture":"feedback"}', b'{"kind":"stop"}']
                for message in messages:
                    sender.send(message)
                self.assertEqual([receiver.receive(1000) for _ in messages], messages)
            finally:
                sender.close()
                receiver.close()


class AtomicEvidenceTests(unittest.TestCase):
    def test_invalid_replacement_preserves_previous_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "record.json"
            atomic_json(destination, {"state": "received"}, durable=True)
            before = destination.read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(destination, {"latency": float("nan")}, durable=True)
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_concurrent_publications_leave_one_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "record.json"
            payloads = [{"identifier": index, "values": [index] * 100} for index in range(8)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda payload: atomic_json(destination, payload), payloads))
            self.assertIn(json.loads(destination.read_text()), payloads)
            self.assertEqual(list(destination.parent.iterdir()), [destination])
