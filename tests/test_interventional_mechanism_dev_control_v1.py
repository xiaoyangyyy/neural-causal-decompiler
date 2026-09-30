"""Control replay rejects candidate tampering without changing the frozen bundle."""
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from verify_interventional_mechanism_dev_control_v1 import verify


def test_frozen_observational_control_replays():
    result = verify()
    assert result["status"] == "verified-development-control"
    assert result["control_rows"] == 512
    assert result["exact_row_overlap_with_mixed_arm"] == 0
    assert result["original_claim_closed"] is False


def test_control_tamper_rejected_in_isolated_copy(tmp_path):
    source = ROOT / "runs/interventional_mechanism_dev_control_v1/candidate"
    copy = tmp_path / "candidate"
    shutil.copytree(source, copy)
    archive_path = copy / "mechanism_observations.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        matrices = {key: archive[key].copy() for key in archive.files}
    matrices["fit_control_observation"][0, 0] += 1.
    np.savez_compressed(archive_path, **matrices)
    with pytest.raises(ValueError, match="Control candidate hash mismatch"):
        verify(output=tmp_path)
