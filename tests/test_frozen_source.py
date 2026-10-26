"""Retained source verification rejects changed evidence and paths outside the snapshot."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from humanoid_agent.artifact_io import atomic_json, canonical_json_sha256
from humanoid_agent.experiment_records import validate_frozen_source


class FrozenSourceTests(unittest.TestCase):
    def test_valid_snapshot_and_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            source = episode / "source/package/module.py"
            source.parent.mkdir(parents=True)
            source.write_text("retained source")
            manifest = {"package/module.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            atomic_json(episode / "source-manifest.json", manifest)
            expected_manifest_sha256 = canonical_json_sha256(manifest)
            validate_frozen_source(episode, expected_manifest_sha256)
            source.write_text("changed source")
            with self.assertRaisesRegex(ValueError, "source changed"):
                validate_frozen_source(episode, expected_manifest_sha256)

    def test_changed_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            atomic_json(episode / "source-manifest.json", {})
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                validate_frozen_source(episode, "0" * 64)

    def test_traversal_absolute_paths_and_symlinks_cannot_escape_source_root(self) -> None:
        for escape_kind in ("parent", "absolute", "symlink"):
            with self.subTest(escape_kind=escape_kind), tempfile.TemporaryDirectory() as temporary:
                episode = Path(temporary)
                source_root = episode / "source"
                source_root.mkdir()
                outside_source = episode / "outside.py"
                outside_source.write_text("outside source")
                match escape_kind:
                    case "parent":
                        relative_path = "../outside.py"
                    case "absolute":
                        relative_path = str(outside_source)
                    case _:
                        (source_root / "linked.py").symlink_to(outside_source)
                        relative_path = "linked.py"
                manifest = {relative_path: hashlib.sha256(outside_source.read_bytes()).hexdigest()}
                atomic_json(episode / "source-manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    validate_frozen_source(episode, canonical_json_sha256(manifest))

    def test_malformed_digest_is_rejected_before_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            manifest = {"module.py": "invalid digest"}
            atomic_json(episode / "source-manifest.json", manifest)
            with self.assertRaises(ValidationError):
                validate_frozen_source(episode, canonical_json_sha256(manifest))
