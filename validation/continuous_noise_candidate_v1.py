"""Development-only continuous additive noise candidates for explicit SCMs.

This module proposes full-support marginal laws from separate fit and selection
residual matrices. It does not certify true exogenous independence or fit.
"""
from hashlib import sha256
import json
from math import lgamma, log, pi
import numpy as np
from scipy.stats import t as student_t

FAMILIES = ('gaussian', 'laplace', 'student5')
MIN_SCALE = 1e-6
STUDENT5_Q75 = float(student_t.ppf(.75, 5))


def matrix(value, name, minimum_rows=32):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < minimum_rows or not 1 <= a.shape[1] <= 8:
        raise ValueError(name + ' must be an N x d residual matrix')
    if not np.isfinite(a).all():
        raise ValueError(name + ' contains nonfinite residuals')
    return np.ascontiguousarray(a)


def identity(array):
    a = np.ascontiguousarray(array, dtype='<f8')
    return sha256(str(a.shape).encode() + b'\0' + a.tobytes()).hexdigest()


def candidates(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    gaussian_scale = float(np.sqrt(np.mean((x - mean) ** 2)))
    median = float(np.median(x))
    laplace_scale = float(np.mean(np.abs(x - median)))
    student_scale = float(np.median(np.abs(x - median)) / STUDENT5_Q75)
    return {
        'gaussian': {'family': 'gaussian', 'loc': mean,
                     'scale': max(gaussian_scale, MIN_SCALE)},
        'laplace': {'family': 'laplace', 'loc': median,
                    'scale': max(laplace_scale, MIN_SCALE)},
        'student5': {'family': 'student5', 'loc': median,
                     'scale': max(student_scale, MIN_SCALE)},
    }


def log_density(values, model):
    if set(model) != {'family', 'loc', 'scale'} or model['family'] not in FAMILIES:
        raise ValueError('Unsupported continuous noise law')
    loc, scale = float(model['loc']), float(model['scale'])
    if not np.isfinite([loc, scale]).all() or scale <= 0:
        raise ValueError('Invalid continuous noise parameters')
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite likelihood input')
    z = (x - loc) / scale
    if model['family'] == 'gaussian':
        return -.5 * log(2 * pi) - log(scale) - .5 * z * z
    if model['family'] == 'laplace':
        return -log(2 * scale) - np.abs(z)
    return (lgamma(3) - lgamma(2.5) - .5 * log(5 * pi)
            - log(scale) - 3 * np.log1p(z * z / 5))


def row_ids(value, rows, name):
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(name + ' row IDs must match residual rows')
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(name + ' row IDs must be nonempty strings')
    if len(set(value)) != rows:
        raise ValueError(name + ' row IDs must be unique')
    return tuple(value)


def id_identity(value):
    return sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def propose(fit_residuals, selection_residuals, fit_row_ids, selection_row_ids):
    fit = matrix(fit_residuals, 'fit')
    selection = matrix(selection_residuals, 'selection')
    if fit.shape[1] != selection.shape[1]:
        raise ValueError('Residual node dimensions disagree')
    fit_ids = row_ids(fit_row_ids, fit.shape[0], 'fit')
    selection_ids = row_ids(selection_row_ids, selection.shape[0], 'selection')
    if set(fit_ids) & set(selection_ids):
        raise ValueError('Fit and selection row IDs overlap')
    if identity(fit) == identity(selection):
        raise ValueError('Fit and selection matrices are identical')
    models, diagnostics = [], []
    for j in range(fit.shape[1]):
        pool = candidates(fit[:, j])
        scores = {name: float(np.mean(log_density(selection[:, j], pool[name])))
                  for name in FAMILIES}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError('Nonfinite held-out log score')
        winner = max(FAMILIES, key=lambda name: scores[name])
        models.append(pool[winner])
        diagnostics.append({'node': j, 'held_out_mean_log_density': scores,
                            'selected_family': winner})
    return {
        'schema': 'ncd.continuous-noise-candidate.v1',
        'status': 'candidate-only',
        'fit_residual_sha256': identity(fit),
        'selection_residual_sha256': identity(selection),
        'fit_row_ids_sha256': id_identity(fit_ids),
        'selection_row_ids_sha256': id_identity(selection_ids),
        'fit_rows': int(fit.shape[0]),
        'selection_rows': int(selection.shape[0]),
        'node_models': models,
        'selection_diagnostics': diagnostics,
        'joint_noise_independence_proved': False,
        'true_noise_family_proved': False,
        'sampling_independence_proved': False,
        'original_claim_closed': False,
        'original_objective_achieved': False,
    }


def draw(models, rows, seed):
    if type(rows) is not int or rows < 1:
        raise ValueError('Positive row count required')
    rng = np.random.default_rng(seed)
    result = np.empty((rows, len(models)), dtype=np.float64)
    for j, model in enumerate(models):
        log_density(np.array([0.0]), model)
        loc, scale = float(model['loc']), float(model['scale'])
        if model['family'] == 'gaussian':
            result[:, j] = rng.normal(loc, scale, rows)
        elif model['family'] == 'laplace':
            result[:, j] = rng.laplace(loc, scale, rows)
        else:
            result[:, j] = loc + scale * rng.standard_t(5, rows)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite candidate noise draw')
    return result