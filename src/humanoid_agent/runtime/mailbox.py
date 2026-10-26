"""A locked shared-memory slot for latest-value motor state, without a packet backlog."""

import fcntl
import mmap
import os
import struct
from pathlib import Path


class LatestMotorMailbox:
    maximum_payload_bytes = 65536
    payload_length = struct.Struct("!I")
    capacity_bytes = payload_length.size + maximum_payload_bytes

    def __init__(self, path: Path, *, create: bool = False) -> None:
        descriptor = (
            os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            if create
            else os.open(path, os.O_RDWR)
        )
        try:
            if create:
                os.ftruncate(descriptor, self.capacity_bytes)
            elif os.fstat(descriptor).st_size != self.capacity_bytes:
                raise ValueError("Invalid motor mailbox size")
            self.memory = mmap.mmap(descriptor, self.capacity_bytes)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor

    def write(self, payload: bytes) -> bool:
        if not 0 < len(payload) <= self.maximum_payload_bytes:
            raise ValueError("Motor mailbox payload exceeds its size limit or is empty")
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            # A killed writer must leave an empty slot, never an old header
            # pointing into a partially overwritten, potentially valid JSON body.
            self.memory[: self.payload_length.size] = self.payload_length.pack(0)
            self.memory[self.payload_length.size : self.payload_length.size + len(payload)] = (
                payload
            )
            self.memory[: self.payload_length.size] = self.payload_length.pack(len(payload))
        finally:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        return True

    def read(self) -> bytes | None:
        # Independent open descriptions in reader and writer make flock exclude
        # partial copies. No polling loop or blocking lock runs on physics.
        fcntl.flock(self.descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            length = self.payload_length.unpack(self.memory[: self.payload_length.size])[0]
            if length > self.maximum_payload_bytes:
                raise ValueError("Invalid motor mailbox payload length")
            return (
                bytes(self.memory[self.payload_length.size : self.payload_length.size + length])
                if length
                else None
            )
        finally:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)

    def close(self) -> None:
        self.memory.close()
        os.close(self.descriptor)
