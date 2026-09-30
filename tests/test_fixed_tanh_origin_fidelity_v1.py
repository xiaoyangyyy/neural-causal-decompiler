"""Tests for the one-leaf frozen neural fidelity counterexample."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from fixed_tanh_origin_fidelity_v1 import OUTPUT, PACKAGE
from check_fixed_tanh_origin_fidelity_v1 import (
    OUTPUT as RECEIPT, verify as independent_verify,
)


def test_origin_interval_replays():
    result = independent_verify()
    assert result["status"] == "independently-verified-point-counterexample"
    assert result["original_claim_closed"] is False
    assert json.loads(RECEIPT.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("path,replacement", [
    (("proof", "counterexample", "error", 0), "0"),
    (("proof", "status"), "proved"),
    (("point_is_in_original_declared_domain_proved",), True),
    (("original_claim_closed",), True),
])
def test_origin_checker_rejects_tampering(tmp_path, path, replacement):
    cert = json.loads(OUTPUT.read_text(encoding="utf-8"))
    target = cert
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    altered = tmp_path / "changed.json"
    altered.write_text(json.dumps(cert), encoding="utf-8")
    with pytest.raises(ValueError):
        independent_verify(certificate=altered)


def test_origin_checker_rejects_rehashed_bundle(tmp_path):
    copied = tmp_path / "archive"
    shutil.copytree(PACKAGE, copied)
    source = copied / "model/structured/explicit_scm.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["equations"][1]["args"][-1]["args"][0]["value"] = 0.0
    source.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from hashlib import sha256
    manifest["files"]["model/structured/explicit_scm.json"] = sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        independent_verify(package=copied)
