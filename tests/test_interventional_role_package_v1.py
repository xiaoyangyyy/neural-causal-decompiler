"""Safety and completion-gate tests for the portable confirmation package."""
from hashlib import sha256
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import package_interventional_role_confirmation_v1 as producer
from verify_interventional_role_package_v1 import verify_members

GOOD = "runs/interventional_role_confirmation_v1/units/demo/result.json"


def archive_with(tmp_path, names, payload=b"synthetic"):
    path = tmp_path / "sample.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o644
            member.mtime = member.uid = member.gid = 0
            import io
            archive.addfile(member, io.BytesIO(payload))
    return path


def test_manifested_member_round_trips(tmp_path):
    archive = archive_with(tmp_path, [GOOD])
    expected = {GOOD: {"bytes": 9, "sha256": sha256(b"synthetic").hexdigest()}}
    assert verify_members(archive, expected) == {GOOD}


def test_altered_member_digest_is_rejected(tmp_path):
    archive = archive_with(tmp_path, [GOOD])
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_members(archive, {GOOD: {"bytes": 9, "sha256": "0" * 64}})


@pytest.mark.parametrize("names", [
    [GOOD, GOOD],
    ["runs/interventional_role_confirmation_v1/../escape.json"],
])
def test_duplicate_or_traversal_member_is_rejected(tmp_path, names):
    archive = archive_with(tmp_path, names)
    expected = {name: {"bytes": 9, "sha256": sha256(b"synthetic").hexdigest()}
                for name in names}
    with pytest.raises(ValueError, match="Unsafe"):
        verify_members(archive, expected)


def test_packager_refuses_missing_full_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "HANDOFF", tmp_path / "missing_handoff.json")
    with pytest.raises(ValueError, match="not complete"):
        producer.inputs()
