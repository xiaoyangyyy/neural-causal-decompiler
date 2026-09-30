"""Evaluator checks for paired do semantics and skipped cut equations."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from evaluate_interventional_mechanism_dev_v1 import (
    evaluate_predictors, intervention_masks, true_deterministic,
)
from ncd.multiverse import GraphWorld, Term


def world():
    return GraphWorld(
        seed=1, split="dev",
        graph=((0, 1, 0), (0, 0, 0), (0, 0, 0)),
        equations=((), (Term("linear", (0,), 1.),), ()),
        noise_family="gaussian", noise_scale=.35,
        samples=32, scales=(1., 1., 1.),
        family="linear_gaussian",
    )


def test_exact_mechanisms_have_zero_error_under_paired_interventions():
    target = world()
    u = np.random.default_rng(17).normal(size=(32, 3))
    predictors = [
        lambda x: np.zeros(len(x)),
        lambda x: x[:, 0],
        lambda x: np.zeros(len(x)),
    ]
    metrics = evaluate_predictors(target, predictors, u, np.ones(3))
    assert len(metrics["conditions"]) == len(intervention_masks(3)) == 9
    assert metrics["max_executed_normalized_mse"] == 0.
    assert metrics["max_paired_contrast_normalized_abs_error"] == 0.
    do_target = next(row for row in metrics["conditions"]
                     if row["interventions"] == {"1": 1.})
    assert do_target["nodes"][1]["executed"] is False
    assert do_target["nodes"][1]["reason"] == "structural_equation_cut_by_do"


def test_wrong_mechanism_exposes_two_do_contrast_error():
    target = world()
    u = np.random.default_rng(19).normal(size=(32, 3))
    predictors = [lambda x: np.zeros(len(x)) for _ in range(3)]
    metrics = evaluate_predictors(target, predictors, u, np.ones(3))
    row = next(value for value in metrics["paired_contrasts"]
               if value["source"] == 0 and value["target"] == 1)
    assert row["true_contrast"] == 2.
    assert row["predicted_contrast"] == 0.
    assert row["normalized_abs_error"] == 2.
    observed = target.sample(exogenous=u, samples=32)
    np.testing.assert_allclose(true_deterministic(target, observed, 1),
                               observed[:, 0])