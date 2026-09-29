"""Regression tests for the exact interval-chain proof kernel."""
from fractions import Fraction as F
from itertools import combinations
import gzip
import json
from pathlib import Path
import pytest
from ncd.exact_interval_lower import (
    _benchmark, base_constraints, chain_constraints, verify_farkas,
    verify_interval_chain_exclusion, verify_exact_nine_state_upper,
    exact_nine_initial, exact_nine_step,
)
from ncd.continuous_separation import _digest


def _satisfies(rows, point):
    return all(sum((a*x for a,x in zip(row, point)), F(0)) <= bound
               for row, bound in rows)


def test_known_nine_state_model_satisfies_every_source_chain():
    _, lam, beta = _benchmark()
    m = 9
    centers = [F(i,10) for i in range(1,10)]
    radius = F('0.1005')
    point = tuple(max(F(0), c-radius) for c in centers) + tuple(
        min(F(1), c+radius) for c in centers)
    assert _satisfies(base_constraints(m,F(0.101)), point)
    chains = [chain for length in range(1,m+1)
              for chain in combinations(range(m), length)]
    assert all(any(_satisfies(chain_constraints(m,i,c,lam,beta),point)
                   for c in chains) for i in range(m))


def test_farkas_checker_rejects_modified_or_incomplete_proofs():
    rows = (((F(1),), F(0)), ((F(-1),), F(-1)))
    assert verify_farkas(rows, [[0,'1'],[1,'1']])
    with pytest.raises(ValueError):
        verify_farkas(rows, [[0,'1'],[1,'2']])
    with pytest.raises(ValueError):
        verify_farkas(rows, [[0,'1']])
    system, _, _ = _benchmark()
    certificate = {
        'schema':'ncd.exact-interval-chain-lower.v1',
        'system_sha256':_digest(system.to_dict()),
        'epsilon':repr(0.101),
        'excluded_states':8,
        'proof':[],
    }
    with pytest.raises(ValueError,match='Truncated'):
        verify_interval_chain_exclusion(certificate)


def test_exact_upper_and_stored_eight_state_exclusion():
    upper = verify_exact_nine_state_upper()
    assert upper['states'] == 9 and len(upper['target_chains']) == 9
    x = F(1,3)
    source = exact_nine_initial(x)
    target = exact_nine_step(source,F(2,3))
    assert 0 <= target < 9
    radius = F('0.1005')
    _, lam, beta = _benchmark()
    src_center = F(source+1,10)
    tgt_center = F(target+1,10)
    assert max(F(0),tgt_center-radius) <= lam*max(F(0),src_center-radius)+beta*F(2,3)
    assert lam*min(F(1),src_center+radius)+beta*F(2,3) <= min(F(1),tgt_center+radius)
    proof = Path(__file__).resolve().parents[1] / 'validation' / 'exact_interval_lower_8.json.gz'
    with gzip.open(proof,'rt',encoding='utf-8') as handle:
        certificate = json.load(handle)
    result = verify_interval_chain_exclusion(certificate)
    assert result == {
        'status':'verified', 'excluded_states':8,
        'closed_branches':40387, 'internal_branches':158,
        'proof_tokens':40545,
    }
