"""Exact arithmetic checks for the independent post-acceptance truth evaluator."""
from fractions import Fraction as Q
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation/frozen_truth_gap_package_v1'))
from frozen_truth_gap.proof import strict_gap


def test_strict_two_target_incompatibility_uses_exact_rationals():
    positive = strict_gap(['29/1000', '30/1000'], '1', '59/2000')
    negative = strict_gap(['-30/1000', '-29/1000'], '1', '-59/2000')
    assert Q(positive['two_target_incompatibility_margin_lower']) == Q(9, 1000)
    assert Q(negative['two_target_incompatibility_margin_lower']) == Q(9, 1000)
    assert Q(negative['neural_program_absolute_error_upper']) == Q(1, 2000)


def test_no_strict_gap_or_unfaithful_program_is_rejected():
    with pytest.raises(ValueError, match='No strict incompatibility'):
        strict_gap(['-15/1000', '-14/1000'], '1', '-29/2000')
    with pytest.raises(ValueError, match='lacks neural fidelity'):
        strict_gap(['-30/1000', '-29/1000'], '1', '1/10')
    with pytest.raises(ValueError, match='No strict incompatibility'):
        strict_gap(['-1/100', '1/100'], '1', '0')