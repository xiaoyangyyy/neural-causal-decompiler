"""One-step interventional support: witness, absence, and fail-closed checks."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest

from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.continuous_compositional_realization import value
from ncd.interventional_support import _exact_value
from ncd.interventional_support import (
    propose_support, certify_support, verify_support, run_case, verify_case)
from ncd.io import read_json, save_json


def _system(hidden_weights, hidden_biases, output_weights, output_bias):
    transition = ReLUMLP(
        (tuple(tuple(float(v) for v in row) for row in hidden_weights),
         tuple(tuple(float(v) for v in row) for row in output_weights)),
        (tuple(float(v) for v in hidden_biases), (float(output_bias),)))
    observation = ReLUMLP((((1.0,),),), ((0.0,),))
    return ContinuousReLUSystem(1, 1, transition, observation)


def test_interior_tent_needs_more_than_endpoint_probe():
    system = _system(((1, 0), (1, 0), (1, 0)),
                     (-.25, -.5, -.75), ((.1, -.2, .1),), .1)
    assert system.step((0,), (.5,))[0] == system.step((1,), (.5,))[0]
    proposal = propose_support(system.step, 1, 1)
    assert [(w["output"], w["input"]) for w in proposal["witnesses"]] == [(0, 0)]
    cert = certify_support(system, proposal)
    assert verify_support(system, proposal, cert) == {
        "status": "certified", "present_count": 1,
        "absent_count": 1, "unresolved_count": 0}

    empty = deepcopy(proposal)
    empty["witnesses"] = []
    uncertain = certify_support(system, empty)
    assert uncertain["status"] == "unresolved"
    assert uncertain["unresolved"] == [[0, 0]]


def test_narrow_tent_escapes_all_five_probe_levels_but_stays_unresolved():
    system = _system(((1, 0), (1, 0), (1, 0)),
                     (-.55, -.60, -.65), ((.1, -.2, .1),), .1)
    proposal = propose_support(system.step, 1, 1)
    assert proposal["witnesses"] == []
    assert value(system.transition, [Q("0.6"), Q("0.5")])[0] != value(
        system.transition, [Q("0.5"), Q("0.5")])[0]
    certificate = certify_support(system, proposal)
    assert certificate["status"] == "unresolved"
    assert certificate["unresolved"] == [[0, 0]]


def test_cancellation_stays_unresolved_instead_of_claiming_absence():
    system = _system(((1, 0), (1, 0)), (0, 0), ((1, -1),), .1)
    proposal = propose_support(system.step, 1, 1)
    assert not proposal["witnesses"]
    cert = certify_support(system, proposal)
    assert cert["status"] == "unresolved"
    assert cert["unresolved"] == [[0, 0]]
    assert cert["absent_count"] == 1


def test_frozen_sparse_model_and_tampering(tmp_path):
    model = Path(__file__).resolve().parents[1] / (
        "runs/learned_local_global_v1/seed_7201/d_8/system.json")
    system = ContinuousReLUSystem.from_dict(read_json(model))
    result = run_case(system, tmp_path)
    assert result == {"status": "certified", "present_count": 32,
                      "absent_count": 48, "unresolved_count": 0}
    proposal = read_json(tmp_path / "proposal.json")
    certificate = read_json(tmp_path / "certificate.json")
    forged = deepcopy(proposal)
    forged["witnesses"][0]["right"] = forged["witnesses"][0]["left"]
    with pytest.raises(ValueError, match="Invalid coordinate"):
        verify_support(system, forged, certificate)
    assert _exact_value(system.transition, tuple([Q(1, 2)] * 10)) == tuple(
        value(system.transition, [Q(1, 2)] * 10))
    forged_query = deepcopy(proposal)
    forged_query["query_calls"] += 1
    save_json(tmp_path / "proposal.json", forged_query)
    with pytest.raises(ValueError, match="proposal replay mismatch"):
        verify_case(system, tmp_path)
    save_json(tmp_path / "proposal.json", proposal)
    changed = deepcopy(certificate)
    changed["absent_count"] -= 1
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_support(system, proposal, changed)
    changed_model = deepcopy(system.to_dict())
    changed_model["transition"]["weights"][-1][0][0] += 0.001
    other = ContinuousReLUSystem.from_dict(changed_model)
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_case(other, tmp_path)


