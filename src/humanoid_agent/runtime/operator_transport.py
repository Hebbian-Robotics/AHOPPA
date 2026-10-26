"""Typed local ZeroMQ request/reply for operator commands and acknowledgments."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import zmq
from pydantic import Field, TypeAdapter, model_validator

from humanoid_agent.command_admission import EpisodeClock
from humanoid_agent.finite_command import Identifier
from humanoid_agent.runtime.local_transport import local_socket_path
from humanoid_agent.runtime.records import CommandStatus, IdentifierReceipt
from humanoid_agent.runtime.timing import CommandTimingEvidence
from humanoid_agent.sonic_planner_input import SonicMotionPublication
from humanoid_agent.validated_types import ValidatedModel


class SubmitCommand(ValidatedModel):
    kind: Literal["command"] = "command"
    publication: SonicMotionPublication
    timing: CommandTimingEvidence | None = None


class RequestObservation(ValidatedModel):
    kind: Literal["observation"] = "observation"
    identifier: Identifier


class QueryOperatorState(ValidatedModel):
    kind: Literal["state"] = "state"


OperatorRequest = Annotated[
    SubmitCommand | RequestObservation | QueryOperatorState, Field(discriminator="kind")
]
REQUEST_ADAPTER = TypeAdapter(OperatorRequest)


class OperatorState(ValidatedModel):
    clock: EpisodeClock | None = None
    command_status: CommandStatus | None = None
    motion_finished: IdentifierReceipt | None = None
    observation_ready: IdentifierReceipt | None = None
    finished: bool = False


class OperatorReply(ValidatedModel):
    state: OperatorState | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_state_or_rejection(self) -> OperatorReply:
        if (self.state is None) == (self.error is None):
            raise ValueError("Operator reply must contain exactly one state or rejection")
        return self


class OperatorServer:
    def __init__(self, commands: Path) -> None:
        self.path = local_socket_path(commands)
        self.socket = zmq.Context.instance().socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 2 * 1024**2)
        try:
            self.socket.bind("ipc://" + str(self.path))
        except BaseException:
            self.socket.close()
            raise

    def poll(self, handle: Callable[[OperatorRequest], OperatorState]) -> None:
        if not self.socket.poll(timeout=0, flags=zmq.POLLIN):
            return
        payload = self.socket.recv()
        try:
            reply = OperatorReply(state=handle(REQUEST_ADAPTER.validate_json(payload)))
        except ValueError as error:
            reply = OperatorReply(error=str(error))
        self.socket.send(reply.model_dump_json().encode())

    def close(self) -> None:
        self.socket.close()
        self.path.unlink(missing_ok=True)


class OperatorClient:
    def __init__(self, commands: Path) -> None:
        self.endpoint = "ipc://" + str(local_socket_path(commands))

    def request(self, request: OperatorRequest, deadline: float) -> OperatorState | None:
        payload = request.model_dump_json().encode()
        with zmq.Context.instance().socket(zmq.REQ) as connection:
            connection.setsockopt(zmq.LINGER, 0)
            connection.setsockopt(zmq.IMMEDIATE, 1)
            connection.setsockopt(zmq.MAXMSGSIZE, 2 * 1024**2)
            connection.connect(self.endpoint)
            while time.monotonic() < deadline:
                timeout_ms = max(0, min(200, int((deadline - time.monotonic()) * 1000)))
                if not connection.poll(timeout_ms, zmq.POLLOUT):
                    continue
                try:
                    connection.send(payload, flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                break
            else:
                return None
            # Recording source artifacts can outlast a poll interval. Preserve the
            # pending reply instead of resubmitting its already delivered command.
            while time.monotonic() < deadline:
                timeout_ms = max(0, min(200, int((deadline - time.monotonic()) * 1000)))
                if not connection.poll(timeout_ms, zmq.POLLIN):
                    continue
                reply = OperatorReply.model_validate_json(connection.recv())
                if reply.error is not None:
                    raise ValueError(reply.error)
                if reply.state is None:
                    raise ValueError("Operator server returned no state")
                return reply.state
        return None
