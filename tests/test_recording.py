"""Episode evidence retains schemas, order and attachments through upstream MCAP."""

import json
import queue
import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import foxglove
from foxglove.websocket import ChannelView, Client, ServerListener
from mcap.reader import make_reader
from pydantic import BaseModel
from websockets.sync.client import connect
from websockets.typing import Subprotocol

from humanoid_agent.command_admission import (
    AdmissionReason,
    CommandAdmission,
    EpisodeClock,
    ObservationIdentity,
)
from humanoid_agent.recording import (
    EpisodeRecorder,
    iter_records,
    read_record,
    record_paths,
    recording_runtime_versions,
    recording_sha256,
    validate_recording,
)
from humanoid_agent.runtime.episode_io import (
    AppendRecord,
    BeginEpisode,
    CommandReceived,
    ObservationRequested,
    StopIO,
    WriteRecord,
    _io_worker,
)
from humanoid_agent.runtime.operator import LocalOperatorConnection
from humanoid_agent.runtime.operator_transport import (
    OperatorClient,
    RequestObservation,
    SubmitCommand,
)
from humanoid_agent.runtime.records import CommandStatus, IdentifierReceipt
from humanoid_agent.runtime.timing import CommandTimingEvidence
from humanoid_agent.sonic_planner_input import ControllerObservation, SonicPlannerCommandStream
from tests.test_local_transport import planner_publication


class RecordingTests(unittest.TestCase):
    def test_mcap_ledger_restores_duplicate_authority_without_json_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = EpisodeRecorder(root)

            def record(path: Path, payload: BaseModel) -> None:
                recorder.write(path.relative_to(root).as_posix(), payload)

            stream = SonicPlannerCommandStream(
                root / "command-ledger", "episode_a", record=record, restore_records=False
            )
            stream.register_observation(
                ControllerObservation(
                    identity=ObservationIdentity(
                        episode_id="episode_a",
                        identifier="initial",
                        captured_elapsed_wall_seconds=0,
                        episode_deadline_elapsed_wall_seconds=60,
                    ),
                    pelvis_quaternion_wxyz=[1, 0, 0, 0],
                )
            )
            command = planner_publication()
            self.assertEqual(stream.activate(command, 0).reason, AdmissionReason.ACCEPTED)
            recorder.close()
            restored = SonicPlannerCommandStream(
                root / "command-ledger", "episode_a", record=record
            )
            self.assertEqual(restored.activate(command, 1).reason, AdmissionReason.DUPLICATE)
            self.assertIsNone(restored.sample(1, [1, 0, 0, 0]))
            self.assertFalse((root / "command-ledger").exists())

    def test_shared_camera_alias_selects_only_its_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera_output = root / "shared-physics/camera-output"
            (camera_output / "env_0").mkdir(parents=True)
            (camera_output / "env_1").mkdir()
            alias = root / "episode/physics/camera-output"
            alias.parent.mkdir(parents=True)
            alias.symlink_to(camera_output / "env_0", target_is_directory=True)
            recorder = EpisodeRecorder(camera_output)
            for environment_index in range(2):
                recorder.write(
                    f"env_{environment_index}/capture-records.jsonl",
                    {"environment_index": environment_index},
                )
            recorder.close()
            self.assertEqual(
                list(iter_records(alias / "capture-records.jsonl")), [{"environment_index": 0}]
            )
            self.assertEqual(record_paths(alias, "*.jsonl"), [alias / "capture-records.jsonl"])

    def test_recording_runtime_matches_the_project_lock_without_resolving_physics(self) -> None:
        lock = tomllib.loads(Path("uv.lock").read_text())
        installed = recording_runtime_versions()
        self.assertEqual(
            installed,
            {
                package["name"]: package["version"]
                for package in lock["package"]
                if package["name"] in installed
            },
        )

    def test_real_writer_delivers_commands_and_acknowledgments_and_ignores_mismatched_timing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "commands"
            incoming: queue.Queue[object] = queue.Queue()
            outgoing: queue.Queue[object] = queue.Queue()
            ready = threading.Event()
            errors: list[BaseException] = []
            incoming.put(
                BeginEpisode(
                    EpisodeClock(
                        deadline_monotonic_seconds=time.monotonic() + 60,
                        episode_id="episode_a",
                        elapsed_wall_seconds=0,
                        deadline_elapsed_wall_seconds=60,
                    ),
                    {"wall_seconds": 0, "simulation_seconds": 0},
                    capture_initial_observation=False,
                )
            )

            def run_writer() -> None:
                try:
                    _io_worker(root, commands, incoming, outgoing, ready)
                except BaseException as error:
                    errors.append(error)
                    ready.set()

            with (
                patch("humanoid_agent.runtime.episode_io.sys.platform", "linux"),
                patch("humanoid_agent.runtime.episode_io.os.sched_setaffinity", create=True),
            ):
                worker = threading.Thread(target=run_writer)
                worker.start()
                try:
                    self.assertTrue(ready.wait(3))
                    self.assertFalse(errors)
                    command = planner_publication()
                    timing = CommandTimingEvidence(
                        identifier="different_command",
                        episode_id=command.episode_id,
                        observation_id=command.observation_id,
                        source_response_sha256=command.source_response_sha256,
                        observation_received_monotonic_seconds=0,
                        request_started_monotonic_seconds=0,
                        response_received_monotonic_seconds=0,
                        publication_prepared_monotonic_seconds=0,
                    )
                    client = OperatorClient(commands)
                    self.assertIsNotNone(
                        client.request(
                            SubmitCommand(publication=command, timing=timing), time.monotonic() + 2
                        )
                    )
                    delivered = outgoing.get(timeout=2)
                    assert isinstance(delivered, CommandReceived)
                    self.assertEqual(delivered.publication, command)
                    self.assertIsNone(delivered.timing)
                    self.assertEqual(delivered.timing_status, "operator_timing_identity_mismatch")
                    incoming.put(
                        WriteRecord(
                            root / "command-status.json",
                            CommandStatus(
                                identifier=command.identifier,
                                state="admitted",
                                admission=CommandAdmission(
                                    identifier=command.identifier,
                                    reason=AdmissionReason.ACCEPTED,
                                    payload_sha256="a" * 64,
                                    observed_elapsed_wall_seconds=0,
                                    observation_age_seconds=0,
                                ),
                            ),
                        )
                    )
                    incoming.put(
                        WriteRecord(
                            root / "motion-finished.json",
                            IdentifierReceipt(identifier=command.identifier),
                        )
                    )
                    connection = LocalOperatorConnection(root, commands)
                    self.assertTrue(
                        connection.wait_identifier(
                            "motion", command.identifier, time.monotonic() + 2
                        )
                    )
                    self.assertIsNotNone(
                        client.request(
                            RequestObservation(identifier="feedback_1"), time.monotonic() + 2
                        )
                    )
                    requested = outgoing.get(timeout=2)
                    self.assertEqual(requested, ObservationRequested(identifier="feedback_1"))
                finally:
                    incoming.put(StopIO(3))
                    worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertFalse(errors)
            self.assertEqual(read_record(root / "command-status.json")["state"], "admitted")
            self.assertFalse((commands / "command.json").exists())
            self.assertFalse((root / "motion-finished.json").exists())

    def test_records_retain_schemas_order_attachments_and_independent_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            second = root / "second"
            second.mkdir()
            first_recorder = EpisodeRecorder(root)
            second_recorder = EpisodeRecorder(second)
            image = root / "original.png"
            image.write_bytes(b"retained exact image bytes")
            try:
                first_recorder.write(
                    "motion-finished.json", IdentifierReceipt(identifier="motion_1")
                )
                for index in range(100):
                    first_recorder.write("physical-trace.jsonl", {"tick": index})
                first_recorder.attach("images/original.png", image, "image/png")
                second_recorder.write(
                    "motion-finished.json", IdentifierReceipt(identifier="motion_2")
                )
            finally:
                first_recorder.close()
                second_recorder.close()
            self.assertEqual(validate_recording(first_recorder.path), 102)
            self.assertEqual(read_record(root / "motion-finished.json"), {"identifier": "motion_1"})
            self.assertEqual(
                read_record(second / "motion-finished.json"), {"identifier": "motion_2"}
            )
            self.assertEqual(
                list(iter_records(root / "physical-trace.jsonl")),
                [{"tick": index} for index in range(100)],
            )
            self.assertEqual(record_paths(root), [root / "motion-finished.json"])
            with first_recorder.path.open("rb") as stream:
                reader = make_reader(stream, validate_crcs=True)
                schemas = [
                    schema
                    for schema, _, _ in reader.iter_messages(topics=["/motion-finished.json"])
                ]
                schema = schemas[0]
                assert schema is not None
                self.assertEqual(json.loads(schema.data)["title"], "IdentifierReceipt")
            with first_recorder.path.open("rb") as stream:
                attachment = next(make_reader(stream, validate_crcs=True).iter_attachments())
                self.assertEqual(attachment.name, "images/original.png")
                self.assertEqual(attachment.data, image.read_bytes())
            with first_recorder.path.open("rb") as stream:
                image_schema, _, image_message = next(
                    make_reader(stream).iter_messages(topics=["/images/original"])
                )
                assert image_schema is not None
                self.assertEqual(image_schema.name, "foxglove.CompressedImage")
                self.assertIn(image.read_bytes(), image_message.data)

    def test_invalid_values_cannot_corrupt_previously_recorded_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = EpisodeRecorder(root)
            try:
                recorder.write("result.json", {"status": "completed"})
                with self.assertRaises(ValueError):
                    recorder.write("result.json", {"lag": float("nan")})
                with self.assertRaises(ValueError):
                    recorder.write("../outside.json", {})
            finally:
                recorder.close()
            self.assertEqual(read_record(root / "result.json"), {"status": "completed"})
            self.assertEqual(validate_recording(recorder.path), 1)
            with self.assertRaises(RuntimeError):
                recorder.write("result.json", {})

    def test_truncated_recording_fails_closed_without_reading_stale_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = EpisodeRecorder(root)
            recorder.write("result.json", {"status": "completed"})
            recorder.close()
            (root / "result.json").write_text('{"status":"completed"}')
            recorder.path.write_bytes(recorder.path.read_bytes()[:-12])
            with self.assertRaises(ValueError):
                validate_recording(recorder.path)
            with self.assertRaises((ValueError, OSError)):
                read_record(root / "result.json")

    def test_same_channel_feeds_live_websocket_and_mcap(self) -> None:
        subscribed = threading.Event()

        class Listener(ServerListener):
            def on_subscribe(self, client: Client, channel: ChannelView) -> None:
                subscribed.set()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = EpisodeRecorder(root)
            recorder.server = foxglove.start_server(
                host="127.0.0.1", port=0, context=recorder.context, server_listener=Listener()
            )
            try:
                recorder.write("sensors.jsonl", {"sample": 0})
                with connect(
                    f"ws://127.0.0.1:{recorder.server.port}",
                    subprotocols=[Subprotocol("foxglove.sdk.v1")],
                    open_timeout=2,
                ) as connection:
                    channel_id = None
                    while channel_id is None:
                        message = json.loads(connection.recv(timeout=2))
                        if message.get("op") == "advertise":
                            channel_id = next(
                                channel["id"]
                                for channel in message["channels"]
                                if channel["topic"] == "/sensors.jsonl"
                            )
                    connection.send(
                        json.dumps(
                            {
                                "op": "subscribe",
                                "subscriptions": [{"id": 1, "channelId": channel_id}],
                            }
                        )
                    )
                    self.assertTrue(subscribed.wait(2))
                    recorder.write("sensors.jsonl", {"sample": 1})
                    payload = connection.recv(timeout=2)
                    self.assertIsInstance(payload, bytes)
                    assert isinstance(payload, bytes)
                    self.assertEqual(json.loads(payload[13:]), {"sample": 1})
            finally:
                recorder.close()
            self.assertEqual(
                list(iter_records(root / "sensors.jsonl")), [{"sample": 0}, {"sample": 1}]
            )

    def test_evidence_bursts_are_recorded_before_terminal_without_jsonl_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming: queue.Queue[object] = queue.Queue()
            outgoing: queue.Queue[object] = queue.Queue()
            incoming.put(
                BeginEpisode(
                    EpisodeClock(
                        deadline_monotonic_seconds=time.monotonic() + 100,
                        episode_id="episode_a",
                        elapsed_wall_seconds=0,
                        deadline_elapsed_wall_seconds=100,
                    ),
                    {"wall_seconds": 0, "simulation_seconds": 0},
                    capture_initial_observation=False,
                )
            )
            for index in range(100):
                incoming.put(AppendRecord("physical-trace.jsonl", {"tick": index}))
            incoming.put(WriteRecord(root / "terminal.json", {"status": "completed"}, durable=True))
            incoming.put(StopIO(102))
            with (
                patch("humanoid_agent.runtime.episode_io.sys.platform", "linux"),
                patch("humanoid_agent.runtime.episode_io.os.sched_setaffinity", create=True),
            ):
                _io_worker(root, root / "commands", incoming, outgoing, threading.Event())
            self.assertEqual(
                list(iter_records(root / "physical-trace.jsonl")),
                [{"tick": index} for index in range(100)],
            )
            self.assertFalse((root / "physical-trace.jsonl").exists())
            self.assertFalse((root / "terminal.json").exists())
            self.assertEqual(read_record(root / "terminal.json"), {"status": "completed"})
            summary = json.loads((root / "io-summary.json").read_text())
            self.assertEqual(summary["operations_processed"], 102)
            self.assertEqual(summary["recording_sha256"], recording_sha256(root / "records.mcap"))
