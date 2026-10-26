"""Command admission protects motion from expired and repeated publications."""

import shutil
import tempfile
import unittest
from pathlib import Path

from humanoid_agent.command_admission import (
    AdmissionReason,
    CommandAdmissionLedger,
    ObservationIdentity,
)
from humanoid_agent.finite_command import FiniteCommandIdentity
from tests.recording_fixtures import write_test_record


class CommandAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.ledger = CommandAdmissionLedger(self.directory, "episode_a", record=write_test_record)
        self.observation = ObservationIdentity(
            episode_id="episode_a",
            identifier="snapshot_1",
            captured_elapsed_wall_seconds=12,
            episode_deadline_elapsed_wall_seconds=60,
        )
        self.ledger.register_observation(self.observation)
        self.command = FiniteCommandIdentity(
            identifier="motion_1",
            source_response_sha256="a" * 64,
            episode_id="episode_a",
            observation_id="snapshot_1",
            observation_elapsed_wall_seconds=12,
            valid_until_elapsed_wall_seconds=40,
            request_task_completion=False,
        )

    def test_persisted_admission_prevents_reexecution_and_identity_reassignment(self) -> None:
        first = self.ledger.admit(self.command, 20)
        self.assertEqual(first.reason, AdmissionReason.ACCEPTED)
        self.assertEqual(first.observation_age_seconds, 8)
        reopened = CommandAdmissionLedger(self.directory, "episode_a", record=write_test_record)
        replayed = reopened.admit(self.command, 21)
        self.assertEqual(replayed.reason, AdmissionReason.DUPLICATE)
        self.assertEqual(replayed.observation_age_seconds, 9)
        changed = self.command.model_copy(update={"request_task_completion": True})
        self.assertEqual(reopened.admit(changed, 22).reason, AdmissionReason.IDENTIFIER_CONFLICT)
        with self.assertRaisesRegex(ValueError, "cannot be reassigned"):
            reopened.register_observation(
                self.observation.model_copy(update={"captured_elapsed_wall_seconds": 13})
            )

    def test_expired_foreign_and_unbound_commands_never_gain_admission(self) -> None:
        for update, elapsed, expected in [
            ({"identifier": "expired"}, 40, AdmissionReason.EXPIRED),
            (
                {"identifier": "foreign", "episode_id": "episode_b"},
                20,
                AdmissionReason.WRONG_EPISODE,
            ),
            (
                {"identifier": "unknown", "observation_id": "missing"},
                20,
                AdmissionReason.UNKNOWN_OBSERVATION,
            ),
            (
                {"identifier": "mismatch", "observation_elapsed_wall_seconds": 11},
                20,
                AdmissionReason.OBSERVATION_MISMATCH,
            ),
            ({"identifier": "future"}, 11, AdmissionReason.OBSERVATION_MISMATCH),
            (
                {"identifier": "over_budget", "valid_until_elapsed_wall_seconds": 70},
                20,
                AdmissionReason.EXPIRED,
            ),
        ]:
            with self.subTest(expected=expected, update=update):
                command = self.command.model_copy(update=update)
                self.assertEqual(self.ledger.admit(command, elapsed).reason, expected)
                reopened = CommandAdmissionLedger(
                    self.directory, "episode_a", record=write_test_record
                )
                self.assertEqual(reopened.admit(command, elapsed + 1).reason, expected)
        self.assertEqual(
            self.ledger.admit(self.command.model_copy(update={"identifier": "expired"}), 25).reason,
            AdmissionReason.EXPIRED,
        )

    def test_invalid_clock_cannot_consume_a_command_identity(self) -> None:
        for elapsed in [float("nan"), float("inf"), -1]:
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                self.ledger.admit(self.command, elapsed)
        reopened = CommandAdmissionLedger(self.directory, "episode_a", record=write_test_record)
        self.assertEqual(reopened.admit(self.command, 20).reason, AdmissionReason.ACCEPTED)

    def test_restored_observations_cannot_cross_episode_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "ledger identity or episode"):
            CommandAdmissionLedger(self.directory, "episode_b", record=write_test_record)

    def test_renamed_records_cannot_reassign_observations_or_command_identities(self) -> None:
        self.ledger.admit(self.command, 20)
        for record_directory, identifier in (
            (self.ledger.observation_directory, self.observation.identifier),
            (self.ledger.admission_directory, self.command.identifier),
        ):
            original = record_directory / f"{identifier}.json"
            renamed = record_directory / "reassigned.json"
            with self.subTest(record_directory=record_directory):
                original.rename(renamed)
                try:
                    with self.assertRaisesRegex(ValueError, "ledger identity"):
                        CommandAdmissionLedger(
                            self.directory, "episode_a", record=write_test_record
                        )
                finally:
                    renamed.rename(original)

    def test_deferred_persistence_does_not_change_expiry_or_duplicate_authority(self) -> None:
        pending = []
        directory = self.directory / "deferred"
        directory.mkdir()
        ledger = CommandAdmissionLedger(
            directory, "episode_a", record=lambda path, payload: pending.append((path, payload))
        )
        ledger.register_observation(self.observation)
        shutil.rmtree(directory)
        self.assertEqual(ledger.admit(self.command, 20).reason, AdmissionReason.ACCEPTED)
        self.assertEqual(ledger.admit(self.command, 21).reason, AdmissionReason.DUPLICATE)
        expired = self.command.model_copy(update={"identifier": "late_motion"})
        self.assertEqual(ledger.admit(expired, 40).reason, AdmissionReason.EXPIRED)
        with self.assertRaisesRegex(ValueError, "cannot be reassigned"):
            ledger.register_observation(
                self.observation.model_copy(update={"captured_elapsed_wall_seconds": 13})
            )
        self.assertFalse(directory.exists())
