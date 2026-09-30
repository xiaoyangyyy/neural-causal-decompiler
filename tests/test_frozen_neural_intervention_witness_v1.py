"""Checks for the exact-weight rational neural intervention certificate."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from frozen_neural_intervention_witness_v1 import (
    derive, outward, tanh_point_bounds, verify,
)


def test_rational_neural_gap_and_simultaneous_fidelity_boundary():
    certificate = derive()
    assert verify(certificate)["status"] == "verified-refuted-scoped"
    assert Fraction(certificate["true_minus_neural_gap_interval"][0]) > Fraction(13, 20)
    for key in ("minus_one_normalized_gap_interval", "plus_one_normalized_gap_interval"):
        assert Fraction(certificate[key][0]) > Fraction(3, 5)
        assert Fraction(certificate[key][0]) > Fraction(1, 100) * 2


def test_outward_rounding_and_tanh_enclosure():
    lo, hi = outward(Fraction(-1, 3), Fraction(1, 3))
    assert lo <= Fraction(-1, 3) <= hi
    assert lo <= Fraction(1, 3) <= hi
    with localcontext() as context:
        context.prec = 80
        e = Decimal(3).exp()
        true_tanh = Fraction((e - 1) / (e + 1))
    lower, upper = tanh_point_bounds(Fraction(3, 2))
    assert lower < true_tanh < upper


def test_changed_weight_and_certificate_rejected(tmp_path):
    model = ROOT / "validation" / "continuous_noise_posthoc_package_v1" / "model" / "baseline" / "mechanism_2.pt"
    altered = tmp_path / "mechanism_2.pt"
    shutil.copyfile(model, altered)
    raw = altered.read_bytes()
    altered.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        derive(model_path=altered)
    cert = json.loads((ROOT / "validation" / "frozen_neural_intervention_witness_v1.json").read_text())
    cert["strict_gap_lower_gt_13_over_20"] = False
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify(cert)