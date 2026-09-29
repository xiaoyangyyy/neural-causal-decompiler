"""Finite statistical primitives; regression always predicts held-out rows."""
import numpy as np

FEATURES = ("corr", "abs_corr", "var_log_ratio", "skew_x", "skew_y", "kurt_x", "kurt_y",
            "dep_xy", "resdep_xy", "resdep_yx", "reserr_xy", "reserr_yx", "mixed_xy", "mixed_yx")

def standardize(x):
    return (x - x.mean()) / max(float(x.std()), 1e-6)

def _basis(x, centers):
    x = np.clip(x, -8, 8)
    return np.column_stack([np.ones(len(x)), x, x*x/4, x*x*x/16, np.sin(x), np.cos(x),
                            np.sin(2*x), np.cos(2*x), np.tanh(x),
                            np.exp(-.5 * (x[:, None] - centers[None, :])**2)])

def residual(y, x):
    """Two-fold cross-fit, including train-only standardization/basis selection.

    Sorting ensures that permuting dataset rows only permutes the result.
    Alternating sorted rows gives deterministic, balanced interpolation folds.
    """
    if len(x) < 16: raise ValueError("At least 16 samples required")
    order = np.lexsort((y, x))
    r = np.empty(len(x), dtype=float)
    for parity in (0, 1):
        test, train = order[parity::2], order[1-parity::2]
        xm, xs = x[train].mean(), max(float(x[train].std()), 1e-6)
        ym, ys = y[train].mean(), max(float(y[train].std()), 1e-6)
        z = (x[train] - xm) / xs
        centers = np.quantile(z, [.1, .3, .5, .7, .9])
        a, b = _basis(z, centers), _basis((x[test]-xm)/xs, centers)
        ridge = np.eye(a.shape[1]) * .1
        ridge[0, 0] = 1e-6
        beta = np.linalg.solve(a.T @ a + ridge, a.T @ ((y[train]-ym)/ys))
        r[test] = y[test] - (b @ beta * ys + ym)
    return r

def dependence(x, y):
    """Normalized centered RBF-kernel HSIC, a score rather than a p-value."""
    def kernel(v):
        dist = (v[:, None] - v[None, :])**2
        positive = dist[dist > 1e-12]
        bandwidth = np.median(positive) if len(positive) else 1.
        k = np.exp(-dist / (2 * max(bandwidth, 1e-8)))
        return k - k.mean(0)[None, :] - k.mean(1)[:, None] + k.mean()
    k, l = kernel(x), kernel(y)
    return float(np.sum(k*l) / max(np.sqrt(np.sum(k*k)*np.sum(l*l)), 1e-12))

def extract_one(data):
    d = np.asarray(data, dtype=float)
    if d.ndim != 2 or d.shape[1] != 2 or len(d) < 16 or not np.isfinite(d).all():
        raise ValueError("Expected finite N x 2 dataset with N >= 16")
    raw_x, raw_y = d.T
    x, y = standardize(raw_x), standardize(raw_y)
    ry, rx = residual(y, x), residual(x, y)
    corr = float(np.mean(x*y))
    values = [corr, abs(corr), np.log(max(raw_x.var(), 1e-12)/max(raw_y.var(), 1e-12)),
              np.mean(x**3), np.mean(y**3), np.mean(x**4)-3, np.mean(y**4)-3,
              dependence(x, y), dependence(x, ry), dependence(y, rx),
              np.mean(ry**2), np.mean(rx**2), np.mean(x*x*y), np.mean(y*y*x)]
    a = np.array(values, dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError("Non-finite primitive")
    return a

def extract(data):
    return np.stack([extract_one(d) for d in data])
