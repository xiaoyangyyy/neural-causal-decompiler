"""Tamper and scope checks for the exact unary-grammar obstruction."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import verify_frozen_unary_grammar_v1 as checker


def test_live_certificate_has_strict_replayed_gap():
    receipt = checker.verify()
    assert receipt["status"] == "verified-finite-grammar-obstruction"
    assert float(checker.Q(receipt["normalized_grid_lower"])) > .01
    assert receipt["all_real_coefficients_covered"] is True
    assert receipt["larger_program_grammars_covered"] is False
    assert receipt["original_claim_closed"] is False


@pytest.mark.parametrize("field,value", [
    ("dual_weights", ["0"] * 7),
    ("grid_max_error_lower", "1"),
    ("one_percent_absolute_threshold", "1"),
    ("refutes_all_possible_program_grammars", True),
])
def test_independent_checker_rejects_claim_or_dual_tamper(tmp_path, monkeypatch, field, value):
    certificate = json.loads(checker.CERT.read_text(encoding="utf-8"))
    certificate[field] = value
    changed = tmp_path / "altered.json"
    changed.write_text(json.dumps(certificate), encoding="utf-8")
    monkeypatch.setattr(checker, "CERT", changed)
    with pytest.raises(ValueError):
        checker.verify()



def test_second_certificate_is_strict_inside_unit_box():
    receipt = checker.verify(unit_box=True)
    assert receipt["status"] == "verified-finite-grammar-obstruction"
    assert receipt["domain"].startswith("[-1,1]")
    assert float(checker.Q(receipt["normalized_grid_lower"])) > .03
    assert receipt["larger_program_grammars_covered"] is False
    assert receipt["original_claim_closed"] is False


def test_second_certificate_rejects_false_dual(tmp_path, monkeypatch):
    certificate = json.loads(checker.NARROW_CERT.read_text(encoding="utf-8"))
    certificate["dual_weights"][0] = "0"
    changed = tmp_path / "altered_unit_box.json"
    changed.write_text(json.dumps(certificate), encoding="utf-8")
    monkeypatch.setattr(checker, "NARROW_CERT", changed)
    with pytest.raises(ValueError):
        checker.verify(unit_box=True)



def test_scoped_records_bind_installed_and_independent_receipts():
    from hashlib import sha256

    installed = ROOT / "validation/frozen_unary_grammar_installed_replay_v1.json"
    installed_hash = sha256(installed.read_bytes()).hexdigest()
    for record_name, cert, receipt in (
        ("frozen_unary_grammar_proof_record_v1.json",
         checker.CERT, checker.RECEIPT),
        ("frozen_unary_grammar_unit_box_proof_record_v1.json",
         checker.NARROW_CERT, checker.NARROW_RECEIPT),
    ):
        record = json.loads((ROOT / "validation" / record_name).read_text(encoding="utf-8"))
        dependencies = record["proof_dependencies"]
        assert dependencies["certificate_sha256"] == sha256(cert.read_bytes()).hexdigest()
        assert dependencies["verification_receipt_sha256"] == sha256(receipt.read_bytes()).hexdigest()
        assert dependencies["installed_replay_sha256"] == installed_hash
        assert record["original_atoms_closed"] is False
