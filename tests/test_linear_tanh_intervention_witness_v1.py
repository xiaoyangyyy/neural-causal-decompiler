"""Independent checks around the fixed double-intervention witness."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from linear_tanh_intervention_witness_v1 import derive, tanh_one_bounds, verify


def test_rational_witness_exceeds_half_and_reverifies():
    certificate = derive()
    assert certificate["status"] == "refuted-scoped"
    assert Fraction(certificate["true_minus_estimated_gap_interval"][0]) > Fraction(1, 2)
    assert verify(certificate)["status"] == "verified-refuted-scoped"
    with localcontext() as context:
        context.prec = 80
        e2 = Decimal(2).exp()
        t = Fraction((e2 - 1) / (e2 + 1))
    lower, upper = tanh_one_bounds()
    assert lower < t < upper


def test_witness_rejects_changed_certificate_and_equation(tmp_path):
    certificate = derive()
    corrupted = dict(certificate)
    corrupted["strict_gap_lower_gt_half"] = False
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify(corrupted)
    path = ROOT / "validation" / "continuous_noise_posthoc_package_v1" / "model" / "structured" / "explicit_scm.json"
    model = json.loads(path.read_text(encoding="utf-8"))
    model["equations"][2]["args"][1]["args"][1]["op"] = "sin"
    changed = tmp_path / "explicit_scm.json"
    changed.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected expression operator"):
        derive(scm_path=changed)