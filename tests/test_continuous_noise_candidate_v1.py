"""Development checks for continuous noise candidates and sample separation."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation'))
from continuous_noise_candidate_v1 import candidates, draw, log_density, propose


def test_three_full_support_families_selected_on_disjoint_synthetic_data():
    rng = np.random.default_rng(8100)
    def rows(n):
        return np.column_stack((
            rng.normal(.4, 1.1, n),
            rng.laplace(-.2, .8, n),
            .3 + .9 * rng.standard_t(5, n),
        ))
    fit, selection = rows(4096), rows(4096)
    result = propose(fit, selection, [f'fit:{j}' for j in range(4096)],
                     [f'selection:{j}' for j in range(4096)])
    assert [row['family'] for row in result['node_models']] == [
        'gaussian', 'laplace', 'student5']
    assert result['fit_residual_sha256'] != result['selection_residual_sha256']
    assert not result['joint_noise_independence_proved']
    assert not result['true_noise_family_proved']
    assert not result['sampling_independence_proved']
    assert not result['original_claim_closed']
    sample = draw(result['node_models'], 32, 4701)
    assert sample.shape == (32, 3) and np.isfinite(sample).all()


def test_selection_rows_cannot_change_fitted_parameters():
    fit = np.arange(128, dtype=float)[:, None] / 20
    left = np.full((64, 1), -.5)
    right = np.full((64, 1), 10.)
    fit_ids = [f'fit:{j}' for j in range(128)]
    selection_ids = [f'selection:{j}' for j in range(64)]
    a = propose(fit, left, fit_ids, selection_ids)
    b = propose(fit, right, fit_ids, selection_ids)
    pool = candidates(fit[:, 0])
    assert a['node_models'][0] in pool.values()
    assert b['node_models'][0] in pool.values()
    assert pool == candidates(fit[:, 0])
    assert a['selection_diagnostics'] != b['selection_diagnostics']


def test_nonfinite_wrong_shape_and_false_same_split_rejected():
    fit = np.arange(64, dtype=float)[:, None]
    fit_ids = [f'fit:{j}' for j in range(64)]
    selection_ids = [f'selection:{j}' for j in range(64)]
    with pytest.raises(ValueError, match='overlap'):
        propose(fit, fit.copy(), fit_ids, fit_ids)
    with pytest.raises(ValueError, match='overlap'):
        propose(fit, fit[::-1], fit_ids, fit_ids[::-1])
    with pytest.raises(ValueError, match='unique'):
        propose(fit, fit.copy(), fit_ids, [selection_ids[0]] * 64)
    with pytest.raises(ValueError, match='match residual rows'):
        propose(fit, fit.copy(), fit_ids, selection_ids[:-1])
    with pytest.raises(ValueError, match='identical'):
        propose(fit, fit.copy(), fit_ids, selection_ids)
    with pytest.raises(ValueError, match='node dimensions'):
        propose(fit, np.ones((64, 2)), fit_ids, selection_ids)
    bad = fit.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        propose(bad, fit, fit_ids, selection_ids)
    with pytest.raises(ValueError, match='Invalid continuous noise parameters'):
        log_density(np.array([0.0]), {'family': 'gaussian', 'loc': 0, 'scale': 0})