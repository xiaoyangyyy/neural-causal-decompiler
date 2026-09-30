"""Development preflight replay and tampering checks."""
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from verify_interventional_mechanism_dev_preflight_v2 import verify


def test_frozen_dev_preflight_replays_from_observation_sources():
    result = verify()
    assert result["status"] == "verified-development-preflight"
    assert not result["original_claim_closed"]
    assert result["independent_confirmation_worlds"] == 0


def test_modified_candidate_batch_is_rejected(tmp_path):
    source = ROOT / "runs" / "interventional_mechanism_dev_v2"
    target = tmp_path / "development_bundle"
    shutil.copytree(source, target)
    data = target / "candidate" / "mechanism_observations.npz"
    raw = data.read_bytes()
    data.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="Candidate file hash mismatch"):
        verify(output=target)