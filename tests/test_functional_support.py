"""Function-level ReLU support proofs on adversarial controls."""
from copy import deepcopy
import random
from fractions import Fraction as Q
from pathlib import Path

import pytest

from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.continuous_compositional_realization import value
from ncd.functional_support import (
    _strict_point, certify_functional_support, verify_functional_support,
    run_case, verify_case)
from ncd.interventional_support import propose_support
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "runs" / "interventional_support_global_v1" / "controls"


def _load(name):
    return ContinuousReLUSystem.from_dict(read_json(CONTROLS / name / "system.json"))


def test_fourier_motzkin_strict_feasibility():
    assert _strict_point(((Q(1), Q(0)), (Q(-1), Q(0))), 1) is None
    assert _strict_point(((Q(1), Q(-1, 4)), (Q(-1), Q(3, 4))), 1) == (Q(1, 2),)
    constraints = ((Q(1), Q(0), Q(-1, 4)),
                   (Q(-1), Q(0), Q(3, 4)),
                   (Q(0), Q(1), Q(-1, 4)),
                   (Q(0), Q(-1), Q(3, 4)))
    assert _strict_point(constraints, 2) == (Q(1, 2), Q(1, 2))


def test_narrow_tent_gets_exact_interior_witness(tmp_path):
    system = _load("narrow_tent")
    proposal = propose_support(system.step, 1, 1)
    assert proposal["witnesses"] == []
    result = run_case(system, tmp_path)
    assert result["status"] == "certified"
    assert result["region_witness_count"] == 1
    certificate = read_json(tmp_path / "certificate.json")
    witness = certificate["witnesses"][0]
    assert witness["origin"] == "exact-region-search"
    assert Q(witness["exact_difference"]) != 0
    assert Q(witness["left"][1]) == Q(witness["right"][1])
    assert verify_case(system, tmp_path) == result
    forged = deepcopy(certificate)
    forged["functional_absence"] = [[0, 0]]
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_functional_support(system, proposal, forged)


def test_path_cancellation_has_functional_absence_proof():
    system = _load("path_cancellation")
    proposal = propose_support(system.step, 1, 1)
    certificate = certify_functional_support(system, proposal)
    assert certificate["status"] == "certified"
    assert certificate["functional_absence"] == [[0, 0]]
    assert certificate["region_proof"]["region_count"] >= 1
    assert certificate["absent_count"] == 2


def test_limits_fail_closed():
    system = _load("narrow_tent")
    proposal = propose_support(system.step, 1, 1)
    certificate = certify_functional_support(system, proposal, max_hidden=2)
    assert certificate["status"] == "unresolved"
    assert certificate["unresolved"] == [[0, 0]]
    assert certificate["region_proof"]["status"] == "resource_limit"


def test_degenerate_zero_preactivation_and_multilayer_cancellation():
    transition = ReLUMLP(
        weights=(((1., 0.), (1., 0.), (0., 0.)),
                 ((1., -1., 2.),),
                 ((1.,),)),
        biases=((0., 0., 0.), (0.,), (0.,)))
    observation = ReLUMLP((((1.,),),), ((0.,),))
    system = ContinuousReLUSystem(1, 1, transition, observation)
    proposal = propose_support(system.step, 1, 1)
    certificate = certify_functional_support(system, proposal)
    assert certificate["status"] == "certified"
    assert certificate["functional_absence"] == [[0, 0]]
    assert certificate["absent_count"] == 2


def test_random_small_networks_against_direct_exact_grid():
    rng = random.Random(4101)
    levels = (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1))
    observation = ReLUMLP((((1.,),),), ((0.,),))
    for _ in range(25):
        hidden = tuple(tuple(float(rng.choice((-1, 0, 1))) for _ in range(2))
                       for _ in range(3))
        hidden_bias = tuple(float(rng.choice((-.5, 0, .5))) for _ in range(3))
        head = (tuple(float(rng.choice((-1, 0, 1))) for _ in range(3)),)
        transition = ReLUMLP((hidden, head), (hidden_bias, (0.,)))
        system = ContinuousReLUSystem(1, 1, transition, observation)
        proposal = propose_support(system.step, 1, 1)
        certificate = certify_functional_support(system, proposal)
        assert certificate["status"] == "certified"
        assert verify_functional_support(system, proposal, certificate)["unresolved_count"] == 0
        present = {w["input"] for w in certificate["witnesses"]}
        for j in range(2):
            if j in present:
                continue
            for other in levels:
                outputs = []
                for level in levels:
                    point = [other, other]
                    point[j] = level
                    outputs.append(value(transition, point)[0])
                assert len(set(outputs)) == 1
