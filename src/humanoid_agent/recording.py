"""Foxglove recording/live sinks and MCAP readers for versioned episode evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Generator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import foxglove
import zmq
from foxglove.channels import CompressedImageChannel
from foxglove.mcap import MCAPWriteOptions
from foxglove.messages import CompressedImage, Timestamp
from mcap.exceptions import McapError
from mcap.reader import make_reader
from mcap.stream_reader import StreamReader
from pydantic import BaseModel

from humanoid_agent.paths import PROJECT_ROOT

RECORDING_FILENAME = "records.mcap"
# Evidence identifiers stay stable across package renames.
RECORDING_PROFILE = "webolt-recording-v1"


def recording_runtime_versions(requirements_path: Path | None = None) -> dict[str, str]:
    """Check uv's generated, hash-pinned recording closure in either interpreter."""
    requirements = (
        requirements_path
        if requirements_path is not None
        else PROJECT_ROOT / "configs/recording-runtime-requirements.txt"
    ).read_text()
    expected = dict(re.findall(r"^([a-z0-9-]+)==([^\s\\]+)", requirements, flags=re.MULTILINE))
    if set(expected) != {"foxglove-sdk", "mcap", "lz4", "zstandard", "pyzmq"}:
        raise ValueError("Recording runtime requirements have an unexpected dependency closure")
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError(
            "Installed recording libraries differ from the frozen runtime requirements"
        )
    return actual


class EpisodeRecorder:
    """One background owner; Foxglove supplies serialization sinks, chunks and indexes.

    Scientific payload dictionaries are accepted at this adapter boundary. Typed
    domain records retain their Pydantic JSON schemas in the MCAP channels.
    """

    def __init__(self, directory: Path, *, live_port: int | None = None) -> None:
        frozen_requirements = next(
            (
                ancestor / "inputs/recording-runtime-requirements.txt"
                for ancestor in directory.resolve().parents
                if (ancestor / "inputs/recording-runtime-requirements.txt").is_file()
            ),
            None,
        )
        runtime_versions = recording_runtime_versions(frozen_requirements)
        self.path = directory / RECORDING_FILENAME
        self.context = foxglove.Context()
        self.channels: dict[str, foxglove.Channel] = {}
        self.image_channels: dict[str, CompressedImageChannel] = {}
        self.writer = foxglove.open_mcap(
            str(self.path),
            context=self.context,
            writer_options=MCAPWriteOptions(
                profile=RECORDING_PROFILE,
                compression_threads=0,
            ),
        )
        self.server = None
        try:
            if live_port is not None:
                self.server = foxglove.start_server(
                    host="127.0.0.1",
                    port=live_port,
                    context=self.context,
                    name="Humanoid agent episode evidence",
                    message_backlog_size=64,
                )
            self.writer.write_metadata(
                "webolt",
                {
                    "profile": RECORDING_PROFILE,
                    "clock": "host monotonic reception time; source timestamps remain in payloads",
                    "recording_runtime": json.dumps(runtime_versions, sort_keys=True),
                    "transport_runtime": json.dumps(
                        {"pyzmq": version("pyzmq"), "libzmq": zmq.zmq_version()}, sort_keys=True
                    ),
                },
            )
        except BaseException:
            self.writer.close()
            raise
        self.closed = False
        self.message_count = 0

    def write(self, name: str, payload: Any) -> None:
        if self.closed:
            raise RuntimeError("Episode recorder is closed")
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Evidence names must stay inside their episode recording")
        # Serialize before logging: Foxglove's generic JSON logger allows NaN.
        values = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        encoded = (json.dumps(values, allow_nan=False, ensure_ascii=False) + "\n").encode()
        channel = self.channels.get(name)
        if channel is None:
            schema = (
                payload.model_json_schema()
                if isinstance(payload, BaseModel)
                else {
                    "title": "HumanoidAgentJsonRecord",
                    "type": "object",
                }
            )
            channel = foxglove.Channel(
                "/" + name,
                schema=schema,
                context=self.context,
                metadata={"record_name": name},
            )
            self.channels[name] = channel
        channel.log(encoded, log_time=time.monotonic_ns())
        self.message_count += 1

    def attach(
        self,
        name: str,
        source: Path,
        media_type: str,
        *,
        image_topic: str | None = None,
        source_timestamp_ns: int | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("Episode recorder is closed")
        timestamp = time.monotonic_ns()
        source_timestamp = timestamp if source_timestamp_ns is None else source_timestamp_ns
        data = source.read_bytes()
        self.writer.attach(
            log_time=timestamp,
            create_time=source_timestamp,
            name=name,
            media_type=media_type,
            data=data,
        )
        if media_type == "image/png":
            topic = image_topic if image_topic is not None else "/images/" + source.stem
            channel = self.image_channels.get(topic)
            if channel is None:
                channel = CompressedImageChannel(topic, context=self.context)
                self.image_channels[topic] = channel
            channel.log(
                CompressedImage(
                    timestamp=Timestamp(
                        source_timestamp // 1_000_000_000, source_timestamp % 1_000_000_000
                    ),
                    frame_id=source.stem,
                    data=data,
                    format="png",
                ),
                log_time=timestamp,
            )
            self.message_count += 1

    def flush(self, *, durable: bool = False) -> None:
        self.writer.flush()
        if durable:
            with self.path.open("rb") as stream:
                os.fsync(stream.fileno())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.writer.close()
            with self.path.open("rb") as stream:
                os.fsync(stream.fileno())
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if self.server is not None:
                self.server.stop()
            for channel in self.channels.values():
                channel.close()


def recording_location(path: Path) -> tuple[Path, str] | None:
    """Locate a logical record in its nearest owning recording, including ledger rows."""
    resolved_path = path.resolve()
    for directory in resolved_path.parents:
        recording = directory / RECORDING_FILENAME
        if recording.is_file():
            return recording, "/" + resolved_path.relative_to(directory).as_posix()
    return None


def iter_record_bytes(path: Path) -> Generator[bytes]:
    location = recording_location(path)
    if location is None:
        with path.open("rb") as stream:
            yield from stream if path.suffix == ".jsonl" else [stream.read()]
        return
    recording, topic = location
    try:
        with recording.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            if reader.get_header().profile != RECORDING_PROFILE:
                raise ValueError("Unsupported episode recording profile")
            for _, channel, message in reader.iter_messages(topics=[topic], log_time_order=False):
                if channel.message_encoding != "json":
                    raise ValueError("Episode evidence requires JSON-encoded messages")
                yield message.data
    except McapError as error:
        raise ValueError(f"Invalid or incomplete MCAP recording: {recording}") from error


def iter_records(path: Path) -> Generator[dict[str, Any]]:
    for payload in iter_record_bytes(path):
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("Episode records must be JSON objects")
        yield record


def read_record_bytes(path: Path) -> bytes:
    location = recording_location(path)
    if location is None:
        return path.read_bytes()
    recording, topic = location
    try:
        with recording.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            if reader.get_header().profile != RECORDING_PROFILE:
                raise ValueError("Unsupported episode recording profile")
            record = next(reader.iter_messages(topics=[topic], reverse=True), None)
            if record is None:
                raise FileNotFoundError(f"Missing MCAP evidence: {path}")
            return record[2].data
    except McapError as error:
        raise ValueError(f"Invalid or incomplete MCAP recording: {recording}") from error


def read_record(path: Path) -> dict[str, Any]:
    record = json.loads(read_record_bytes(path))
    if not isinstance(record, dict):
        raise ValueError("Episode records must be JSON objects")
    return record


def record_paths(directory: Path, pattern: str = "*.json") -> list[Path]:
    location = recording_location(directory / "placeholder")
    if location is None:
        return sorted(directory.glob(pattern))
    recording, _ = location
    try:
        with recording.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            if reader.get_header().profile != RECORDING_PROFILE:
                raise ValueError("Unsupported episode recording profile")
            summary = reader.get_summary()
            if summary is None:
                raise ValueError("Episode recording has no finalized channel index")
            paths = {
                recording.parent / channel.topic.removeprefix("/")
                for channel in summary.channels.values()
            }
            return sorted(
                directory / path.name
                for path in paths
                if path.parent == directory.resolve() and path.match(pattern)
            )
    except McapError as error:
        raise ValueError(f"Invalid or incomplete MCAP recording: {recording}") from error


def record_exists(path: Path) -> bool:
    return path in record_paths(path.parent, path.name)


def recording_sha256(path: Path) -> str:
    location = recording_location(path)
    source = location[0] if location is not None else path
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_recording(path: Path) -> int:
    """Use the upstream streaming reader to validate all chunks, CRCs and the footer."""
    from mcap.records import Header, Message

    count = 0
    try:
        with path.open("rb") as stream:
            for record in StreamReader(stream, validate_crcs=True).records:
                if isinstance(record, Header) and record.profile != RECORDING_PROFILE:
                    raise ValueError("Unsupported episode recording profile")
                count += isinstance(record, Message)
    except McapError as error:
        raise ValueError(f"Invalid or incomplete MCAP recording: {path}") from error
    return count
