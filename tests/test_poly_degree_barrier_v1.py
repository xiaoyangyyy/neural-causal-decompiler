"""Exact finite-difference checks for a real frozen two-parent Tanh mechanism."""
from fractions import Fraction as Q
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation/poly_degree_barrier_package_v1'))
from poly_degree_barrier.proof import (
    POINTS, WEIGHTS, derive, finite_difference_intervals, separated_margin,
)

CHECKPOINT = ROOT / 'runs/active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline/mechanism_0.pt'


def test_seventh_finite_difference_annihilates_every_degree_six_restriction():
    assert len(POINTS) == 8 and sum(map(abs, WEIGHTS)) == 128
    values = [[str(Q(t) ** 6)] * 2 for t in POINTS]
    assert finite_difference_intervals(values) == (Q(0), Q(0))
    seventh = [[str(Q(t) ** 7)] * 2 for t in POINTS]
    lo, hi = finite_difference_intervals(seventh)
    assert lo == hi == Q(5040) * Q(11, 40) ** 7 > 0


def test_tampered_or_unseparated_point_bounds_are_rejected():
    with pytest.raises(ValueError, match='do not exclude'):
        separated_margin([['0', '0']] * 8, '1')
    with pytest.raises(ValueError, match='Reversed'):
        finite_difference_intervals([['1', '0']] + [['0', '0']] * 7)
    intervals = [['0', '0']] * 7 + [['2', '2']]
    assert separated_margin(intervals, '1')[-1] == Q(18, 25)


def test_actual_frozen_checkpoint_has_strict_degree_six_barrier(tmp_path):
    certificate = derive(CHECKPOINT)
    assert certificate['status'] == 'refuted-scoped'
    assert certificate['degree_bound'] == 6
    assert len(certificate['network_point_enclosures']) == 8
    assert Q(certificate['strict_incompatibility_margin_lower']) > Q(2, 5)
    assert certificate['original_claim_closed'] is False
    changed = tmp_path / 'wrong.pt'
    data = CHECKPOINT.read_bytes()
    changed.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match='Wrong frozen neural checkpoint'):
        derive(changed)