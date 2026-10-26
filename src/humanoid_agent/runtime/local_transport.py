"""Bounded local ZeroMQ sockets; the library owns framing, queues and delivery."""

import hashlib
import os
from pathlib import Path

import zmq


def local_socket_path(identity: Path) -> Path:
    # Unix socket paths have small platform limits, independently of episode paths.
    digest = hashlib.sha256(str(identity.resolve()).encode()).hexdigest()[:32]
    directory = Path("/tmp") / f"humanoid-agent-ipc-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise PermissionError("Local socket directory must be private to its owner")
    return directory / f"{digest}.sock"


class LocalPushSender:
    def __init__(self, identity: Path) -> None:
        self.socket = zmq.Context.instance().socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 1000)
        self.socket.setsockopt(zmq.SNDHWM, 16)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.connect("ipc://" + str(local_socket_path(identity)))

    def send(self, payload: bytes) -> None:
        try:
            self.socket.send(payload)
        except zmq.Again as error:
            raise TimeoutError("Local receiver cannot accept its bounded message queue") from error

    def close(self) -> None:
        self.socket.close()


class LocalPullReceiver:
    def __init__(self, identity: Path) -> None:
        self.path = local_socket_path(identity)
        self.socket = zmq.Context.instance().socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 16)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 2 * 1024**2)
        try:
            self.socket.bind("ipc://" + str(self.path))
        except BaseException:
            self.socket.close()
            raise

    def receive(self, timeout_ms: int = 10) -> bytes | None:
        if self.socket.poll(timeout_ms, zmq.POLLIN):
            return self.socket.recv()
        return None

    def close(self) -> None:
        self.socket.close()
        self.path.unlink(missing_ok=True)
