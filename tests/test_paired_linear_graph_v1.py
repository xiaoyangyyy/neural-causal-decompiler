"""Paired-do linear graph theorem and archived diagnostic tests."""
from fractions import Fraction as Q
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
from ncd.paired_linear_graph import recover_graph
from check_paired_linear_graph_development_v1 import (
    CERTIFICATE, OUTPUT as RECEIPT, PACKAGE, verify,
)


def synthetic_case(n):
    direct = [[Q(0) for _ in range(n)] for _ in range(n)]
    for source in range(n):
        for target in (source + 1, source + 2):
            if target < n:
                direct[target][source] = Q(((-1) ** source) * (source + 2), target + 3)
    noise = [Q(i + 1, 2 * n + 1) for i in range(n)]
    def response(source, level):
        values = [Q(0)] * n
        for target in range(n):
            values[target] = (
                level if target == source else noise[target]
                + sum((direct[target][parent] * values[parent]
                       for parent in range(target)), Q(0))
            )
        return list(map(str, values))
    plus = [response(i, Q(1)) for i in range(n)]
    minus = [response(i, Q(-1)) for i in range(n)]
    expected = [[int(direct[target][source] != 0)
                 for target in range(n)] for source in range(n)]
    return plus, minus, expected, direct


@pytest.mark.parametrize("n", [3, 5, 8])
def test_exact_multinode_paired_recovery(n):
    plus, minus, graph, direct = synthetic_case(n)
    result = recover_graph(plus, minus, [["1", "-1"]] * n)
    assert result["graph"] == graph
    assert result["direct_effect"] == [[str(value) for value in row] for row in direct]
    assert result["paired_exogenous_required"] is True
    assert result["truth_graph_read_by_estimator"] is False


def test_rejects_invalid_intervention_grid():
    plus, minus, _, _ = synthetic_case(3)
    with pytest.raises(ValueError, match="Invalid intervention pair"):
        recover_graph(plus, minus, [["1", "1"]] * 3)
    plus[0][0] = "0"
    with pytest.raises(ValueError, match="Intervened coordinate"):
        recover_graph(plus, minus, [["1", "-1"]] * 3)


def test_rejects_singular_total_effect():
    plus = [["1", "1"], ["1", "1"]]
    minus = [["0", "0"], ["0", "0"]]
    with pytest.raises(ValueError, match="singular"):
        recover_graph(plus, minus, [["1", "0"], ["1", "0"]])



def test_node_order_permutation_preserves_recovery():
    plus, minus, old_graph, _ = synthetic_case(5)
    order = [3, 0, 4, 1, 2]
    mapped_plus = [[plus[old_source][old_target] for old_target in order]
                   for old_source in order]
    mapped_minus = [[minus[old_source][old_target] for old_target in order]
                    for old_source in order]
    expected = [[old_graph[old_source][old_target] for old_target in order]
                for old_source in order]
    assert recover_graph(mapped_plus, mapped_minus, [["1", "-1"]] * 5)["graph"] == expected


def test_rejects_cyclic_response_pattern():
    with pytest.raises(ValueError, match="Nonzero self mechanism|cyclic"):
        recover_graph(
            [["1", "1/2"], ["1/2", "1"]],
            [["0", "0"], ["0", "0"]],
            [["1", "0"], ["1", "0"]],
        )


def test_archived_independent_replay():
    receipt = verify()
    assert receipt["status"] == "independently-verified-scoped-exact-recovery"
    assert receipt["original_claim_closed"] is False
    assert json.loads(RECEIPT.read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize("change", ["response", "scope"])
def test_archive_checker_rejects_certificate_tamper(tmp_path, change):
    cert = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    if change == "response":
        cert["plus_responses"][0][1] = "0"
    else:
        cert["independent_intervention_samples_certified"] = True
    altered = tmp_path / "changed.json"
    altered.write_text(json.dumps(cert), encoding="utf-8")
    with pytest.raises(ValueError):
        verify(certificate=altered)


def test_archive_checker_rejects_rehashed_world(tmp_path):
    copied = tmp_path / "archive"
    shutil.copytree(PACKAGE, copied)
    world_path = copied / "world.json"
    world = json.loads(world_path.read_text(encoding="utf-8"))
    world["equations"][1][0]["coefficient"] *= 2
    world_path.write_text(json.dumps(world), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from hashlib import sha256
    manifest["files"]["world.json"] = sha256(world_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        verify(package=copied)
