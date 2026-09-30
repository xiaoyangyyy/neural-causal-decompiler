"""Exact nonlinear paired-do theorem, tampering, and assumption-boundary tests."""
from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
from ncd.paired_nonlinear_graph import recover_local_graph
from verify_paired_nonlinear_graph_case_v1 import (
    CERTIFICATE, OUTPUT, verify,
)


def test_exact_nonlinear_development_certificate():
    receipt = verify()
    assert receipt["status"] == "independently-verified-conditional-nonlinear-graph"
    assert receipt["graph_source_target"] == [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    assert Q(receipt["direct_error_bound_inf"]) < Q(1, 5)
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize("field", ["response", "curvature", "graph", "source", "claim"])
def test_independent_checker_rejects_certificate_tamper(tmp_path, field):
    changed = deepcopy(json.loads(CERTIFICATE.read_text(encoding="utf-8")))
    if field == "response":
        changed["estimator_input"]["responses"][0][2] = "1"
    elif field == "curvature":
        changed["estimator_input"]["curvature_bound"] = "0"
    elif field == "graph":
        changed["estimator_result"]["graph_source_target"][0][2] = 0
    elif field == "source":
        changed["source_sha256"]["ncd/paired_nonlinear_graph.py"] = "0" * 64
    else:
        changed["original_claim_closed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        verify(path)


def test_bounded_response_error_keeps_separated_edge():
    h = Q(1, 16)
    delta = Q(1, 100000)
    result = recover_local_graph(
        ["0", "0"], [[str(h), str(h/2 + delta)], ["0", str(h)]], str(h),
        curvature_bound="0", response_error_bound=str(delta),
        minimum_visible_direct_effect="1/2")
    assert result["graph_source_target"] == [[0, 1], [0, 0]]
    assert result["original_claim_closed"] is False


def test_unseparated_curvature_is_rejected():
    record = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    inputs = deepcopy(record["estimator_input"])
    inputs["curvature_bound"] = "10"
    with pytest.raises(ValueError, match="perturbation|margin"):
        recover_local_graph(**inputs)


def test_unpaired_noise_can_create_spurious_edge():
    # True graph is empty. The do(X0) response changed U1, violating pairing.
    h = Q(1, 16)
    result = recover_local_graph(
        ["0", "0"], [[str(h), str(h/2)], ["0", str(h)]], str(h),
        curvature_bound="0", response_error_bound="0",
        minimum_visible_direct_effect="2/5")
    assert result["graph_source_target"] == [[0, 1], [0, 0]]
    assert result["shared_exogenous_state_assumed"] is True