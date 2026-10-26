"""Simulation-only Unitree SDK boundary for unchanged native SONIC deployment.

Publish encoder and ideal IMU readings, never simulator position or odometry.
The vendor owns DDS serialization. Isaac owns physical actuation and limits.
"""

from __future__ import annotations

import importlib
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, FiniteFloat, model_validator

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.dex3 import HAND_JOINT_NAMES, HandEncoderState, HandMotorCommand
from humanoid_agent.finite_command import Identifier
from humanoid_agent.process_lifecycle import process_record
from humanoid_agent.runtime.cpu import CpuAllocation
from humanoid_agent.runtime.mailbox import LatestMotorMailbox
from humanoid_agent.sonic_planner_input import NativePlannerSample
from humanoid_agent.validated_types import (
    ImmutableSequence,
    UnitQuaternion,
    ValidatedModel,
    Vector3,
)

# Sized for the earlier dense joint-reference windows and kept unchanged.
# Keep this separate from the much smaller motor-state mailbox.
SDK_MAXIMUM_PACKET_BYTES = 131072

MotorVector = Annotated[ImmutableSequence[FiniteFloat], Field(min_length=29, max_length=29)]


class PlannerRelayInput(ValidatedModel):
    command_id: Identifier | None
    sample: NativePlannerSample


class PlannerPublicationReceipt(PlannerRelayInput):
    """A successful SDK-worker ZMQ send, not controller consumption or motion."""

    published_monotonic_seconds: FiniteFloat = Field(gt=0)
    sequence_number: int = Field(ge=1)


class SonicEncoderImuState(ValidatedModel):
    """Allowlisted controller sensors in Unitree motor order and body IMU frames."""

    simulation_seconds: FiniteFloat = Field(ge=0)
    left_hand: HandEncoderState | None = None
    right_hand: HandEncoderState | None = None
    joint_positions: MotorVector
    joint_velocities: MotorVector
    joint_accelerations: MotorVector
    measured_joint_torques: MotorVector
    pelvis_quaternion_wxyz: UnitQuaternion
    pelvis_gyroscope_body_rad_s: Vector3
    pelvis_accelerometer_body_m_s2: Vector3
    torso_quaternion_wxyz: UnitQuaternion
    torso_gyroscope_body_rad_s: Vector3
    torso_accelerometer_body_m_s2: Vector3


class SonicMotorCommand(ValidatedModel):
    left_hand: HandMotorCommand | None = None
    right_hand: HandMotorCommand | None = None
    positions: MotorVector
    velocities: MotorVector
    feedforward_torques: MotorVector
    stiffness: MotorVector
    damping: MotorVector

    @model_validator(mode="after")
    def require_nonnegative_gains(self) -> SonicMotorCommand:
        if min(self.stiffness) < 0 or min(self.damping) < 0:
            raise ValueError("Native motor gains must be nonnegative")
        return self


@dataclass(frozen=True)
class ReceivedMotorCommand:
    command: SonicMotorCommand
    received_monotonic_seconds: float


def motor_order_indices(
    articulation_joint_names: list[str], motor_joint_names: list[str]
) -> list[int]:
    """Match physical joint names, rather than assume an articulation traversal."""
    if (
        len(motor_joint_names) != 29
        or len(set(motor_joint_names)) != 29
        or len(set(articulation_joint_names)) != len(articulation_joint_names)
        or set(articulation_joint_names)
        not in (
            set(motor_joint_names),
            set(motor_joint_names) | set(HAND_JOINT_NAMES["left"]) | set(HAND_JOINT_NAMES["right"]),
        )
    ):
        raise ValueError("The SDK bridge requires the same 29 uniquely named body joints")
    return [articulation_joint_names.index(name) for name in motor_joint_names]


class SonicSdkBridge:
    """DDS transport restricted to a network namespace containing only loopback."""

    def __init__(self, *, dex3_hands: bool = False) -> None:
        # sysfs can retain the host namespace's mounted network-device tree.
        # Query the calling thread's actual namespace through the socket API.
        interfaces = {name for _, name in socket.if_nameindex()}
        if interfaces != {"lo"}:
            raise ValueError("Run the simulation SDK bridge in a loopback-only network namespace")
        channel: Any = importlib.import_module("unitree_sdk2py.core.channel")
        message_types: Any = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        defaults: Any = importlib.import_module("unitree_sdk2py.idl.default")
        channel.ChannelFactoryInitialize(0, "lo")
        self.low_state = defaults.unitree_hg_msg_dds__LowState_()
        self.torso_imu = defaults.unitree_hg_msg_dds__IMUState_()
        self.low_state_publisher = channel.ChannelPublisher("rt/lowstate", message_types.LowState_)
        self.torso_imu_publisher = channel.ChannelPublisher(
            "rt/secondary_imu", message_types.IMUState_
        )
        self.low_state_publisher.Init()
        self.torso_imu_publisher.Init()
        self._lock = threading.Lock()
        self._latest_command: ReceivedMotorCommand | None = None
        self._command_error: Exception | None = None
        self.command_subscriber = channel.ChannelSubscriber("rt/lowcmd", message_types.LowCmd_)
        self.command_subscriber.Init(self._receive_command, 1)
        self.hand_publishers: dict[str, Any] = {}
        self.hand_subscribers: list[Any] = []
        self.hand_states: dict[str, Any] = {}
        self.hand_commands: dict[str, tuple[HandMotorCommand, float]] = {}
        if dex3_hands:
            for side in ("left", "right"):
                publisher = channel.ChannelPublisher(
                    f"rt/dex3/{side}/state", message_types.HandState_
                )
                publisher.Init()
                self.hand_publishers[side] = publisher
                self.hand_states[side] = defaults.unitree_hg_msg_dds__HandState_()
                subscriber = channel.ChannelSubscriber(
                    f"rt/dex3/{side}/cmd", message_types.HandCmd_
                )
                subscriber.Init(
                    lambda message, side=side: self._receive_hand_command(side, message), 1
                )
                self.hand_subscribers.append(subscriber)

    def _receive_command(self, message: Any) -> None:
        try:
            motors = message.motor_cmd[:29]
            command = SonicMotorCommand(
                positions=[motor.q for motor in motors],
                velocities=[motor.dq for motor in motors],
                feedforward_torques=[motor.tau for motor in motors],
                stiffness=[motor.kp for motor in motors],
                damping=[motor.kd for motor in motors],
            )
            with self._lock:
                self._latest_command = ReceivedMotorCommand(command, time.monotonic())
        except Exception as error:
            # DDS invokes this on its receive thread. Surface failure to the
            # physics owner instead of silently retaining the preceding command.
            with self._lock:
                self._command_error = error

    def _receive_hand_command(self, side: str, message: Any) -> None:
        try:
            motors = message.motor_cmd
            command = HandMotorCommand(
                positions=[motor.q for motor in motors],
                velocities=[motor.dq for motor in motors],
                feedforward_torques=[motor.tau for motor in motors],
                stiffness=[motor.kp for motor in motors],
                damping=[motor.kd for motor in motors],
            )
            with self._lock:
                self.hand_commands[side] = (command, time.monotonic())
        except Exception as error:
            with self._lock:
                self._command_error = error

    def latest_command(self) -> ReceivedMotorCommand | None:
        with self._lock:
            if self._command_error is not None:
                raise ValueError("Invalid native SDK motor command") from self._command_error
            receipt = self._latest_command
            if receipt is None or not self.hand_publishers:
                return receipt
            if self.hand_commands.keys() != self.hand_publishers.keys():
                return None
            return ReceivedMotorCommand(
                receipt.command.model_copy(
                    update={
                        "left_hand": self.hand_commands["left"][0],
                        "right_hand": self.hand_commands["right"][0],
                    }
                ),
                min(
                    receipt.received_monotonic_seconds,
                    *(entry[1] for entry in self.hand_commands.values()),
                ),
            )

    def close(self) -> None:
        for subscriber in self.hand_subscribers:
            subscriber.Close()
        for publisher in self.hand_publishers.values():
            publisher.Close()
        self.command_subscriber.Close()
        self.low_state_publisher.Close()
        self.torso_imu_publisher.Close()

    def publish(self, sensors: SonicEncoderImuState) -> None:
        for side, publisher in self.hand_publishers.items():
            measured = getattr(sensors, f"{side}_hand")
            if measured is None:
                raise ValueError(f"Missing {side} hand encoders")
            state = self.hand_states[side]
            for index, motor in enumerate(state.motor_state):
                motor.q = measured.positions[index]
                motor.dq = measured.velocities[index]
                motor.ddq = measured.accelerations[index]
                motor.tau_est = measured.measured_torques[index]
            if not publisher.Write(state):
                raise RuntimeError(f"SDK failed to publish {side} hand encoders")
        for index, motor in enumerate(self.low_state.motor_state[:29]):
            motor.q = sensors.joint_positions[index]
            motor.dq = sensors.joint_velocities[index]
            motor.ddq = sensors.joint_accelerations[index]
            motor.tau_est = sensors.measured_joint_torques[index]
        self.low_state.tick = int(sensors.simulation_seconds * 1000)
        self.low_state.imu_state.quaternion[:] = sensors.pelvis_quaternion_wxyz
        self.low_state.imu_state.gyroscope[:] = sensors.pelvis_gyroscope_body_rad_s
        self.low_state.imu_state.accelerometer[:] = sensors.pelvis_accelerometer_body_m_s2
        self.torso_imu.quaternion[:] = sensors.torso_quaternion_wxyz
        self.torso_imu.gyroscope[:] = sensors.torso_gyroscope_body_rad_s
        self.torso_imu.accelerometer[:] = sensors.torso_accelerometer_body_m_s2
        if not self.low_state_publisher.Write(self.low_state):
            raise RuntimeError("SDK failed to publish joint encoders and pelvis IMU")
        if not self.torso_imu_publisher.Write(self.torso_imu):
            raise RuntimeError("SDK failed to publish torso IMU")


class MotorReceiptPacket(ValidatedModel):
    command: SonicMotorCommand
    received_monotonic_seconds: FiniteFloat = Field(ge=0)


class SonicSdkProcessBridge:
    """Keep vendor DDS deserialization and its Python threads off physics' GIL."""

    maximum_packet_bytes = SDK_MAXIMUM_PACKET_BYTES
    maximum_packets_per_poll = 32

    def __init__(
        self,
        python_executable: Path,
        output_directory: Path,
        *,
        native_workspace: Path | None = None,
        dex3_hands: bool = False,
    ) -> None:
        self.native_workspace = native_workspace
        self.output_directory = output_directory
        self.controller_ready = False
        self.closed = False
        self.stop_requested = False
        self.stop_acknowledged = False
        self.received_planner_publications = 0
        self._latest_command: ReceivedMotorCommand | None = None
        self._planner_publications: list[PlannerPublicationReceipt] = []
        self._last_motor_payload: bytes | None = None
        self._motor_busy_since: float | None = None
        with ExitStack() as startup_resources:
            self.connection, worker_connection = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_SEQPACKET
            )
            startup_resources.callback(self.connection.close)
            startup_resources.callback(worker_connection.close)
            self._motor_directory = tempfile.TemporaryDirectory(prefix="humanoid-agent-motor-")
            startup_resources.callback(self._motor_directory.cleanup)
            motor_path = Path(self._motor_directory.name) / "latest-state"
            self.motor_mailbox = LatestMotorMailbox(motor_path, create=True)
            startup_resources.callback(self.motor_mailbox.close)
            self.log = startup_resources.enter_context(
                (output_directory / "sdk-worker.log").open("w")
            )
            command = [
                "taskset",
                "-c",
                CpuAllocation.from_environment().sdk,
                str(python_executable),
                "-m",
                "humanoid_agent",
                "--role",
                "sdk",
                "--socket-fd",
                str(worker_connection.fileno()),
                "--motor-state-path",
                str(motor_path),
            ]
            if dex3_hands:
                command.append("--dex3-hands")
            if native_workspace is not None:
                command = [
                    "unshare",
                    "--user",
                    "--map-root-user",
                    "--net",
                    *command,
                    "--native-workspace",
                    str(native_workspace),
                    "--output",
                    str(output_directory),
                ]
            self.process = subprocess.Popen(
                command,
                pass_fds=(worker_connection.fileno(),),
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            worker_connection.close()
            startup_resources.pop_all()
        try:
            identity_path = output_directory / "sdk-worker-process.json"
            atomic_json(identity_path, process_record(self.process.pid))
            self.connection.settimeout(15)
            if self.connection.recv(self.maximum_packet_bytes + 1) != b"ready":
                raise RuntimeError(
                    "SDK worker failed its startup handshake; inspect sdk-worker.log"
                )
            self.connection.setblocking(False)
        except BaseException:
            # Cancellation must release a child even before construction returns.
            self.close()
            raise

    def publish(self, sensors: SonicEncoderImuState) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("SDK worker exited; inspect sdk-worker.log")
        try:
            self.connection.send(sensors.model_dump_json().encode())
        except BlockingIOError as error:
            raise RuntimeError("SDK worker cannot keep pace with sensor publication") from error

    def publish_planner(
        self,
        sample: NativePlannerSample,
        *,
        command_id: str | None = None,
    ) -> None:
        if self.native_workspace is None:
            raise ValueError("Planner relay requires an owned native controller")
        if self.process.poll() is not None:
            raise RuntimeError("Native controller worker exited")
        try:
            publication = PlannerRelayInput(command_id=command_id, sample=sample)
            self.connection.send(b"planner:" + publication.model_dump_json().encode())
        except BlockingIOError as error:
            raise RuntimeError(
                "Native controller cannot keep pace with planner publication"
            ) from error

    def _receive_pending_messages(self, *, allow_closed: bool) -> None:
        # Loss-intolerant publication evidence stays ordered on this channel;
        # motor freshness is read independently from the latest-value mailbox.
        for _ in range(self.maximum_packets_per_poll):
            try:
                packet = self.connection.recv(self.maximum_packet_bytes + 1)
            except BlockingIOError:
                return
            except ConnectionResetError:
                if allow_closed and self.stop_acknowledged:
                    return
                raise
            if not packet and allow_closed:
                return
            if not packet or len(packet) > self.maximum_packet_bytes:
                raise RuntimeError("SDK worker closed transport or exceeded its packet limit")
            if packet == b"controller-ready":
                self.controller_ready = True
                continue
            if packet.startswith(b"planner-stream-stopped:"):
                count = int(packet.removeprefix(b"planner-stream-stopped:"))
                if not self.stop_requested or count != self.received_planner_publications:
                    raise RuntimeError(
                        "SDK stopped before all planner publication evidence was received"
                    )
                self.stop_acknowledged = True
                self.connection.send(b"stop-confirmed")
                continue
            if packet.startswith(b"planner-sent:"):
                publication = PlannerPublicationReceipt.model_validate_json(
                    packet.removeprefix(b"planner-sent:")
                )
                if publication.sequence_number != self.received_planner_publications + 1:
                    raise ValueError("Native planner publication sequence has a gap or duplicate")
                self.received_planner_publications = publication.sequence_number
                self._planner_publications.append(publication)
                continue
            raise ValueError("Unexpected SDK control packet")

    def latest_command(self) -> ReceivedMotorCommand | None:
        if self.process.poll() is not None:
            raise RuntimeError("SDK worker exited; inspect sdk-worker.log")
        self._receive_pending_messages(allow_closed=False)
        try:
            payload = self.motor_mailbox.read()
        except BlockingIOError:
            now = time.monotonic()
            if self._motor_busy_since is None:
                self._motor_busy_since = now
            elif now - self._motor_busy_since > 1:
                raise TimeoutError(
                    "SDK motor mailbox remained locked for more than one second"
                ) from None
            # Keep the existing device targets for this tick. Do not call an old
            # cached command fresh or expire it while its replacement is locked.
            return None
        self._motor_busy_since = None
        if payload is not None and payload != self._last_motor_payload:
            receipt = MotorReceiptPacket.model_validate_json(payload)
            self._latest_command = ReceivedMotorCommand(
                receipt.command, receipt.received_monotonic_seconds
            )
            self._last_motor_payload = payload
        return self._latest_command

    def take_planner_publications(self) -> list[PlannerPublicationReceipt]:
        """Drain send evidence after consuming the worker's socket in latest_command."""
        publications = self._planner_publications
        self._planner_publications = []
        return publications

    def request_stop(self) -> None:
        """Stop every publisher before waiting for any one worker to exit."""
        if self.closed or self.stop_requested:
            return
        self.stop_requested = True
        if self.process.poll() is None:
            self.connection.send(b"stop")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.request_stop()
            deadline = time.monotonic() + (30 if self.native_workspace is not None else 10)
            while self.process.poll() is None:
                self._receive_pending_messages(allow_closed=True)
                if time.monotonic() >= deadline:
                    from humanoid_agent.runtime.episode import stop_process

                    stop_process(self.process, self.output_directory)
                    raise TimeoutError("SDK worker did not acknowledge and finish shutdown")
                time.sleep(0.002)
            self._receive_pending_messages(allow_closed=True)
            if self.process.returncode != 0 or not self.stop_acknowledged:
                raise RuntimeError(
                    f"SDK worker exited {self.process.returncode} without a complete shutdown handshake; inspect sdk-worker.log"
                )
        finally:
            self.closed = True
            try:
                from humanoid_agent.runtime.episode import stop_process

                # Recorded controllers can outlive a worker killed before its cleanup.
                stop_process(self.process, self.output_directory)
            finally:
                self.connection.close()
                self.log.close()
                self.motor_mailbox.close()
                self._motor_directory.cleanup()
                atomic_json(
                    self.output_directory / "sdk-worker-shutdown.json",
                    {
                        "worker_exit_code": self.process.returncode,
                        "stop_requested": self.stop_requested,
                        "stop_acknowledged": self.stop_acknowledged,
                        "planner_publications_received": self.received_planner_publications,
                    },
                    durable=True,
                )
