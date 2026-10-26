"""Publish complete JSON records atomically, with optional terminal-record durability."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import rfc8785
from boltons.fileutils import atomic_save


def canonical_json_sha256(payload: Any) -> str:
    """Hash JSON content; exact file-byte hashes must continue to hash bytes directly."""
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def atomic_json(destination: Path, payload: Any, *, durable: bool = False) -> None:
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    atomic_write_text(destination, serialized, durable=durable)


def atomic_write_text(destination: Path, text: str, *, durable: bool = False) -> None:
    """Keep live handoffs atomic; explicitly sync terminal evidence outside control."""
    encoded = text.encode("utf-8")
    if not durable:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
        # Match ordinary file creation permissions through the process umask.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return
    with atomic_save(
        str(destination), part_file=f".{destination.name}.{uuid4().hex}.part"
    ) as stream:
        stream.write(encoded)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
