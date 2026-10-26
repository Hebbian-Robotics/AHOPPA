"""Own the simulation SDK transport in a separate process from Isaac physics."""

from __future__ import annotations

import importlib
import logging
import select
import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter

from humanoid_agent.dex3 import HandVector
from humanoid_agent.logging_config import configure_logging
from humanoid_agent.runtime.controller import NativeControllerSession
from humanoid_agent.runtime.mailbox import LatestMotorMailbox
from humanoid_agent.runtime.sdk import (
    SDK_MAXIMUM_PACKET_BYTES,
    MotorReceiptPacket,
    PlannerPublicationReceipt,
    PlannerRelayInput,
    SonicEncoderImuState,
    SonicSdkBridge,
)
from humanoid_agent.sonic_planner_input import NativePlannerSample, publish_native_planner_sample

logger = logging.getLogger(__name__)

MAXIMUM_PACKETS_PER_POLL = 32


class HandReferenceLatch:
    """Retain explicitly requested references without relinquishing native hand control."""

    def __init__(self) -> None:
        self.left: HandVector | None = None
        self.right: HandVector | None = None

    def resolve(self, sample: NativePlannerSample) -> NativePlannerSample:
        if sample.left_hand_joints is not None:
            self.left = sample.left_hand_joints
        if sample.right_hand_joints is not None:
            self.right = sample.right_hand_joints
        if not sample.retain_uncommanded_hand_targets:
            return sample
        return sample.model_copy(
            update={"left_hand_joints": self.left, "right_hand_joints": self.right}
        )


def run_worker(
    socket_descriptor: int,
    native_workspace: Path | None = None,
    output: Path | None = None,
    *,
    motor_state_path: Path,
    dex3_hands: bool = False,
) -> None:
    connection = socket.socket(fileno=socket_descriptor)
    native = None
    publisher = None
    context = None
    sender: Any = None
    if native_workspace is not None:
        if output is None:
            raise ValueError("Native controller relay requires its own output directory")
        if {name for _, name in socket.if_nameindex()} != {"lo"}:
            raise ValueError("Native controller relay requires a private loopback-only namespace")
        subprocess.run(["ip", "link", "set", "lo", "up"], check=True)
    motor_mailbox = LatestMotorMailbox(motor_state_path)
    try:
        bridge = SonicSdkBridge(dex3_hands=dex3_hands)
    except BaseException:
        motor_mailbox.close()
        connection.close()
        raise
    hand_references = HandReferenceLatch()
    motor_backpressure_count = 0
    try:
        if native_workspace is not None:
            zmq: Any = importlib.import_module("zmq")
            sender = importlib.import_module("gear_sonic.utils.teleop.zmq.zmq_planner_sender")
            context = zmq.Context()
            publisher = context.socket(zmq.PUB)
            publisher.bind("tcp://127.0.0.1:17756")
            if output is None:
                raise ValueError("Missing native output directory")
            native = NativeControllerSession(native_workspace, output)
        connection.send(b"ready")
        connection.setblocking(False)
        next_motor_publication = time.monotonic()
        next_ready_check = 0.0
        readiness_sent = False
        publication_count = 0
        last_motor_receipt_sent: float | None = None
        while True:
            planner_update = None
            if select.select([connection], [], [], 0.002)[0]:
                latest_sensor_state = None
                for _ in range(MAXIMUM_PACKETS_PER_POLL):
                    try:
                        packet = connection.recv(SDK_MAXIMUM_PACKET_BYTES + 1)
                    except BlockingIOError:
                        break
                    if packet == b"stop":
                        confirmation_deadline = time.monotonic() + 15
                        connection.settimeout(15)
                        connection.send(
                            b"planner-stream-stopped:" + str(publication_count).encode()
                        )
                        connection.settimeout(max(0.001, confirmation_deadline - time.monotonic()))
                        if connection.recv(SDK_MAXIMUM_PACKET_BYTES + 1) != b"stop-confirmed":
                            raise ValueError("Unexpected packet after SDK stream stop")
                        return
                    if not packet or len(packet) > SDK_MAXIMUM_PACKET_BYTES:
                        raise ValueError("Invalid or closed physics-to-SDK transport")
                    if packet.startswith(b"planner:"):
                        planner_update = PlannerRelayInput.model_validate_json(
                            packet.removeprefix(b"planner:")
                        )
                    else:
                        latest_sensor_state = SonicEncoderImuState.model_validate_json(packet)
                if latest_sensor_state is not None:
                    bridge.publish(latest_sensor_state)
            now = time.monotonic()
            if native is not None and now >= next_ready_check:
                if native.ready() and not readiness_sent:
                    connection.send(b"controller-ready")
                    readiness_sent = True
                next_ready_check = now + 0.02
            # Physics owns the periodic neutral/active/endpoint refresh. Relay
            # its newest sample immediately instead of rescheduling at 30 Hz.
            if publisher is not None and planner_update is not None:
                resolved_sample = hand_references.resolve(planner_update.sample)
                publisher.send(sender.build_command_message(start=True, stop=False, planner=True))
                publish_native_planner_sample(publisher, sender, resolved_sample)
                publication_count += 1
                sent = PlannerPublicationReceipt(
                    sequence_number=publication_count,
                    command_id=planner_update.command_id,
                    sample=resolved_sample,
                    published_monotonic_seconds=time.monotonic(),
                )
                # Unlike the motor latest-value stream, planner evidence is loss-intolerant.
                connection.send(b"planner-sent:" + sent.model_dump_json().encode())
            if now >= next_motor_publication:
                receipt = bridge.latest_command()
                if (
                    receipt is not None
                    and receipt.received_monotonic_seconds != last_motor_receipt_sent
                ):
                    packet = MotorReceiptPacket(
                        command=receipt.command,
                        received_monotonic_seconds=receipt.received_monotonic_seconds,
                    )
                    if motor_mailbox.write(packet.model_dump_json().encode()):
                        last_motor_receipt_sent = receipt.received_monotonic_seconds
                    else:
                        # A reader owns the slot briefly. Retry the latest DDS
                        # state next pass without altering its source timestamp.
                        motor_backpressure_count += 1
                next_motor_publication = now + 0.005
    finally:
        logger.info("SDK motor backpressure count: %d", motor_backpressure_count)
        try:
            if native is not None:
                native.close()
        finally:
            if publisher is not None:
                publisher.close(linger=0)
            if context is not None:
                context.term()
            bridge.close()
            connection.close()
            motor_mailbox.close()


def main(
    *,
    cli_socket_fd: Annotated[int, Parameter(name="--socket-fd")],
    motor_state_path: Path,
    native_workspace: Path | None = None,
    output: Path | None = None,
    dex3_hands: bool = False,
) -> None:
    run_worker(
        cli_socket_fd,
        native_workspace,
        output,
        motor_state_path=motor_state_path,
        dex3_hands=dex3_hands,
    )


if __name__ == "__main__":
    configure_logging()
    App(default_command=main, help=__doc__, suppress_keyboard_interrupt=False)()
