"""Split and checkpoint checks for intervention-augmented mechanism training."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from interventional_mechanism_fit_v1 import (
    ObservationBatch, fit, prepare, save_checkpoint,
)
from ncd.mechanisms import load_mechanism, neural_values


def batch(source, n, seed, interventions):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    y = 1.1 * x0 + rng.normal(scale=.1, size=n)
    x2 = rng.normal(size=n)
    data = np.column_stack((x0, y, x2))
    if 0 in interventions:
        data[:, 0] = interventions[0]
        data[:, 1] = 1.1 * data[:, 0] + rng.normal(scale=.1, size=n)
    if 1 in interventions:
        data[:, 1] = interventions[1]
    return ObservationBatch(
        source, data, tuple(f"{source}:{i}" for i in range(n)), interventions)


def splits():
    fitting = [
        batch("fit-observation", 128, 1, {}),
        batch("fit-do-root", 64, 2, {0: 1.}),
        batch("fit-do-target", 64, 3, {1: 1.}),
    ]
    held_out = [
        batch("val-observation", 64, 4, {}),
        batch("val-do-root", 64, 5, {0: -1.}),
    ]
    return fitting, held_out


def test_target_intervention_excluded_and_frozen_checkpoint_reloads(tmp_path):
    training, validation = splits()
    x_fit, x_val, record = prepare(training, validation, 1, (0,), 3)
    assert len(x_fit) == 192 and len(x_val) == 128
    assert record["fit_coverage"][2]["reason"] == "target_intervened"
    assert not record["true_graph_used"] and not record["sampling_independence_proved"]
    model, fitted = fit(training, validation, 1, (0,), 3,
                        epochs=4, width=8)
    path = tmp_path / "mechanism_1.pt"
    saved = save_checkpoint(model, fitted, path)
    loaded = load_mechanism(path)
    np.testing.assert_allclose(
        neural_values(model, x_val), neural_values(loaded, x_val), rtol=0, atol=0)
    assert len(saved["checkpoint_sha256"]) == 64


def test_overlap_wrong_do_value_and_budget_are_rejected():
    training, validation = splits()
    duplicate = ObservationBatch("val-copy", training[0].data.copy(),
                                 tuple(f"copy:{i}" for i in range(128)), {})
    with pytest.raises(ValueError, match="identical observations"):
        prepare(training, [duplicate], 1, (0,), 3)
    wrong_do = ObservationBatch("bad-do", training[1].data,
                                training[1].row_ids, {0: -1.})
    with pytest.raises(ValueError, match="disagrees"):
        prepare([wrong_do], validation, 1, (0,), 3)
    oversized = batch("too-many", 513, 8, {})
    with pytest.raises(ValueError, match="budget"):
        prepare([oversized], validation, 1, (0,), 3)
    reused_source = batch("fit-observation", 64, 9, {})
    with pytest.raises(ValueError, match="source IDs overlap"):
        prepare(training, [reused_source], 1, (0,), 3)