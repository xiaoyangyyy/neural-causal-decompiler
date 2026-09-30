"""Independent replay agrees on paired-do semantics and rejects metric tampering."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from evaluate_interventional_mechanism_dev_v1 import evaluate_predictors
from verify_interventional_mechanism_dev_evaluation_v1 import (
    independent_metrics, same,
)
from ncd.multiverse import GraphWorld, Term


def world():
    return GraphWorld(
        seed=99, split="dev",
        graph=((0, 1, 0), (0, 0, 0), (0, 0, 0)),
        equations=((), (Term("linear", (0,), .75),), ()),
        noise_family="gaussian", noise_scale=.35, samples=32,
        scales=(.8, 1.4, 1.0), family="linear_gaussian",
    )


def test_independent_paired_do_replay_agrees_at_all_conditions():
    target = world()
    _, exogenous = target.sample(seed=77, samples=32, return_exogenous=True)
    predictors = [
        lambda observed: np.full(len(observed), .1),
        lambda observed: observed[:, 0] * .75 * (1.4 / .8),
        lambda observed: np.zeros(len(observed)),
    ]
    normalizers = np.array([.8, 1.4, 1.])
    expected = evaluate_predictors(target, predictors, exogenous, normalizers)
    observed = independent_metrics(target, predictors, exogenous, normalizers)
    same(expected, observed)
    cut = next(row for row in observed["conditions"]
               if row["interventions"] == {"1": 1.})
    assert cut["nodes"][1]["executed"] is False


def test_read_only_comparison_rejects_subtle_metric_tampering():
    target = world()
    _, exogenous = target.sample(seed=78, samples=32, return_exogenous=True)
    predictors = [lambda observed: np.zeros(len(observed)) for _ in range(3)]
    expected = independent_metrics(target, predictors, exogenous, np.ones(3))
    changed = copy.deepcopy(expected)
    changed["max_executed_normalized_mse"] += 1e-5
    with pytest.raises(ValueError, match="numeric value changed"):
        same(expected, changed)
    changed = copy.deepcopy(expected)
    changed["conditions"][0]["nodes"][0]["executed"] = False
    with pytest.raises(ValueError, match="value changed"):
        same(expected, changed)
