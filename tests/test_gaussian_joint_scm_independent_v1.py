"""Independent arithmetic replay of the portable Gaussian coupling example."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
from check_gaussian_joint_scm_certificate_v1 import (
    CERTIFICATE, audit, replay_receipt,
)


def test_independent_gaussian_joint_coupling_receipt_replays():
    result = replay_receipt()
    assert result["status"] == "verified-conditionally"
    assert result["verified_joint_noise_coupling"] is True
    assert result["true_noise_correlated"] is True
    assert result["premises_verified"] is False
    assert result["original_claim_closed"] is False


@pytest.mark.parametrize("field,value", [
    ("true_noise_covariance", [["1", "0"], ["0", "2"]]),
    ("gaussian_sigma_upper", ["0", "1/2"]),
    ("noise_coordinate_l1_upper", ["0", "0"]),
    ("coordinate_intervention_error_upper", ["0", "5/4"]),
    ("assumptions", ["all premises proved"] * 5),
    ("original_claim_closed", True),
])
def test_independent_checker_rejects_scientific_tampering(tmp_path, field, value):
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    certificate[field] = value
    changed = tmp_path / "modified.json"
    changed.write_text(
        json.dumps(certificate, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        audit(changed)


@pytest.mark.parametrize("nodes", [3, 5, 8])
def test_independent_checker_covers_requested_graph_scales(tmp_path, nodes):
    from ncd.gaussian_joint_scm_certificate import (
        certify_gaussian_joint_scm_error,
    )

    graph = [[int(j == i + 1) for j in range(nodes)]
             for i in range(nodes)]
    lipschitz = [["1/3" if graph[i][j] else 0 for j in range(nodes)]
                 for i in range(nodes)]
    true_mix = [[1 if i == j else ("1/2" if j == i - 1 else 0)
                 for j in range(nodes)] for i in range(nodes)]
    model_mix = [[int(i == j) for j in range(nodes)]
                 for i in range(nodes)]
    certificate = certify_gaussian_joint_scm_error(
        graph, ["1/100"] * nodes, lipschitz,
        [0] * nodes, true_mix, [0] * nodes, model_mix,
        [0] + ["1/2"] * (nodes - 1),
        interventions={0: 1, nodes - 1: -1},
    )
    path = tmp_path / ("joint_n" + str(nodes) + ".json")
    path.write_text(
        json.dumps(certificate, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    result = audit(path)
    assert result["status"] == "verified-conditionally"
    assert result["true_noise_correlated"] is True
    assert certificate["coordinate_intervention_error_upper"][0] == "0"
    assert certificate["coordinate_intervention_error_upper"][-1] == "0"
