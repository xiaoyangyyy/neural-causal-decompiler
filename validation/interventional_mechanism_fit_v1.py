"""Development-only training of a neural SCM mechanism from allowed do data.

The caller supplies an inferred parent set, observed-coordinate batches, and
disjoint observation IDs. True graph/equation metadata is never an input.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import copy

import numpy as np
import torch

from ncd.mechanisms import NeuralMechanism
from ncd.model import set_seed


@dataclass(frozen=True)
class ObservationBatch:
    source: str
    data: np.ndarray
    row_ids: tuple
    interventions: dict


def _validate_batches(batches, split, nodes):
    if not batches:
        raise ValueError(split + " requires observation batches")
    ids = set()
    rows = set()
    accepted = []
    for batch in batches:
        if not isinstance(batch, ObservationBatch) or not isinstance(batch.source, str) or not batch.source:
            raise ValueError("Invalid batch source")
        x = np.asarray(batch.data, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != nodes or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Invalid observed-coordinate batch")
        if not isinstance(batch.row_ids, (tuple, list)) or len(batch.row_ids) != len(x):
            raise ValueError("Row IDs must match observations")
        if any(not isinstance(item, str) or not item for item in batch.row_ids):
            raise ValueError("Row IDs must be nonempty strings")
        if not isinstance(batch.interventions, dict):
            raise ValueError("Interventions must be declared")
        for target, value in batch.interventions.items():
            if type(target) is not int or not 0 <= target < nodes or not np.isfinite(value):
                raise ValueError("Invalid intervention target/value")
            if not np.allclose(x[:, target], value, rtol=0, atol=1e-7):
                raise ValueError("Intervention value disagrees with observations")
        for index, row_id in enumerate(batch.row_ids):
            if row_id in ids:
                raise ValueError("Duplicate observation row ID")
            ids.add(row_id)
            signature = sha256(np.asarray(x[index], dtype="<f8").tobytes()).hexdigest()
            if signature in rows:
                raise ValueError("Duplicate observed row values")
            rows.add(signature)
        accepted.append((batch, np.ascontiguousarray(x)))
    return accepted, ids, rows


def prepare(fit_batches, validation_batches, target, parents, nodes):
    if type(nodes) is not int or not 1 <= nodes <= 8:
        raise ValueError("Supported node count is 1..8")
    if type(target) is not int or not 0 <= target < nodes:
        raise ValueError("Invalid target node")
    if (not isinstance(parents, (tuple, list))
            or any(type(p) is not int or not 0 <= p < nodes or p == target for p in parents)
            or len(set(parents)) != len(parents)):
        raise ValueError("Invalid inferred parent set")
    if not isinstance(fit_batches, (tuple, list)) or not isinstance(validation_batches, (tuple, list)):
        raise ValueError("Batch splits must be sequences")
    fit, fit_ids, fit_rows = _validate_batches(fit_batches, "fit", nodes)
    validation, val_ids, val_rows = _validate_batches(validation_batches, "validation", nodes)
    if sum(len(data) for _, data in (*fit, *validation)) > 512:
        raise ValueError("Mechanism observation budget exceeds 512 rows")
    if {batch.source for batch, _ in fit} & {batch.source for batch, _ in validation}:
        raise ValueError("Fit and validation source IDs overlap")
    if fit_ids & val_ids:
        raise ValueError("Fit and validation row IDs overlap")
    if fit_rows & val_rows:
        raise ValueError("Fit and validation contain identical observations")
    def select(batches):
        pieces, coverage = [], []
        for batch, data in batches:
            if target in batch.interventions:
                coverage.append({"source": batch.source, "rows": len(data),
                                 "used": False, "reason": "target_intervened"})
                continue
            pieces.append(data)
            coverage.append({"source": batch.source, "rows": len(data),
                             "used": True, "intervention_targets":
                             sorted(batch.interventions)})
        if not pieces:
            raise ValueError("No usable mechanism observations")
        joined = np.concatenate(pieces)
        if len(joined) < 32:
            raise ValueError("At least 32 usable rows are required")
        return joined, coverage
    x_fit, fit_coverage = select(fit)
    x_val, val_coverage = select(validation)
    return x_fit, x_val, {
        "schema": "ncd.intervention-mechanism-split.v1",
        "status": "development-only",
        "target": target, "inferred_parents": list(parents),
        "fit_rows": int(len(x_fit)), "validation_rows": int(len(x_val)),
        "fit_coverage": fit_coverage, "validation_coverage": val_coverage,
        "fit_row_id_sha256": sha256(json.dumps(sorted(fit_ids)).encode()).hexdigest(),
        "validation_row_id_sha256": sha256(json.dumps(sorted(val_ids)).encode()).hexdigest(),
        "true_graph_used": False, "true_equations_used": False,
        "sampling_independence_proved": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def fit(fit_batches, validation_batches, target, parents, nodes, *,
        seed=8100, epochs=120, width=48):
    if type(epochs) is not int or not 1 <= epochs <= 120:
        raise ValueError("Epoch budget must be 1..120")
    if type(width) is not int or not 1 <= width <= 48:
        raise ValueError("Width budget must be 1..48")
    x_fit, x_val, record = prepare(
        fit_batches, validation_batches, target, parents, nodes)
    torch.set_num_threads(1)
    set_seed(seed)
    model = NeuralMechanism(tuple(parents), width)
    train = torch.as_tensor(x_fit, dtype=torch.float32)
    dev = torch.as_tensor(x_val, dtype=torch.float32)
    with torch.no_grad():
        y_train = train[:, target]
        model.ymean.copy_(y_train.mean())
        model.ystd.copy_(y_train.std(unbiased=False).clamp_min(.05))
        if parents:
            px = train[:, list(parents)]
            model.mean.copy_(px.mean(0))
            model.std.copy_(px.std(0, unbiased=False).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.0005)
    best = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = (((model(train) - train[:, target]) / model.ystd) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(((model(dev) - dev[:, target]) ** 2).mean())
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {**record, "seed": seed, "epochs": epochs,
                   "best_epoch": best_epoch, "validation_mse": best,
                   "target_training_scale": float(model.ystd)}


def save_checkpoint(model, record, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parents": tuple(model.parents), "width": model.width,
                "state_dict": model.state_dict()}, output)
    return {**record, "checkpoint_sha256": sha256(output.read_bytes()).hexdigest()}