"""Post-hoc, non-confirmatory intervention diagnostic for continuous SCM noise.

The candidate sees observations and frozen neural models only. The true world
metadata and exogenous draws are used solely by the evaluator.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

from continuous_noise_candidate_v1 import draw, propose
from ncd.mechanisms import ExplicitSCM, load_mechanism, neural_values
from ncd.multiverse import GraphWorld


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def observation_residuals(observations, mechanisms):
    """Uses no true graph, equation or exogenous noise."""
    return np.column_stack([
        observations[:, j] - neural_values(model, observations)
        for j, model in enumerate(mechanisms)
    ])


def symbolic_residuals(observations, scm):
    """Calibrate noise against the actual executable symbolic equations."""
    rows = len(observations)
    return np.column_stack([
        observations[:, j] - np.broadcast_to(
            scm.equations[j].evaluate(observations), (rows,))
        for j in range(len(scm.equations))
    ])


def empirical_joint_energy(x, y):
    """Energy distance between finite empirical joint laws, in observed units."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Joint samples must have matching node dimensions")
    if len(x) < 2 or len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Joint samples must be finite and nonempty")
    xy = cdist(x, y, metric="euclidean").mean()
    xx = 2 * pdist(x, metric="euclidean").sum() / len(x) ** 2
    yy = 2 * pdist(y, metric="euclidean").sum() / len(y) ** 2
    return float(max(0., 2 * xy - xx - yy))


def evaluate(world, scm, mechanisms, proposal, eval_observations, eval_u, rows):
    """Truth access starts here, after model selection has finished."""
    scales = np.asarray(world.scales or (1.,) * world.nodes)
    true_noise = eval_u * scales
    candidate_u = draw(proposal["node_models"], rows, 9103)
    rng = np.random.default_rng(9104)
    empirical_u = np.column_stack([
        rng.choice(values, rows) for values in scm.noise_samples
    ])
    neural_residuals = observation_residuals(eval_observations, mechanisms)
    explicit_residuals = symbolic_residuals(eval_observations, scm)
    noise_metrics = []
    for j in range(world.nodes):
        noise_metrics.append({
            "node": j,
            "candidate_w1": float(wasserstein_distance(true_noise[:, j], candidate_u[:, j])),
            "empirical_w1": float(wasserstein_distance(true_noise[:, j], empirical_u[:, j])),
            "neural_residual_vs_true_noise_mae": float(np.mean(
                np.abs(neural_residuals[:, j] - true_noise[:, j]))),
            "symbolic_residual_vs_true_noise_mae": float(np.mean(
                np.abs(explicit_residuals[:, j] - true_noise[:, j]))),
        })
    interventions = [{}, {0: -1.}, {0: 1.}]
    if world.nodes > 1:
        interventions.extend([{1: 1.}, {0: 1., 1: -1.}])
    rows_out = []
    for mask in interventions:
        truth = world.sample(interventions=mask, exogenous=eval_u, samples=rows)
        candidate = scm.sample(rows, interventions=mask, noise_values=candidate_u)
        empirical = scm.sample(rows, interventions=mask, noise_values=empirical_u)
        local_gap = symbolic_residuals(truth, scm) - true_noise
        local_mae = [
            None if j in mask else float(np.mean(np.abs(local_gap[:, j])))
            for j in range(world.nodes)
        ]
        candidate_dist = [float(wasserstein_distance(truth[:, j], candidate[:, j]))
                          for j in range(world.nodes)]
        empirical_dist = [float(wasserstein_distance(truth[:, j], empirical[:, j]))
                          for j in range(world.nodes)]
        rows_out.append({
            "interventions": {str(k): v for k, v in mask.items()},
            "symbolic_mechanism_mae_at_true_parents": local_mae,
            "candidate_marginal_w1": candidate_dist,
            "empirical_marginal_w1": empirical_dist,
            "candidate_mean_marginal_w1": float(np.mean(candidate_dist)),
            "empirical_mean_marginal_w1": float(np.mean(empirical_dist)),
            "candidate_joint_energy": empirical_joint_energy(truth, candidate),
            "empirical_joint_energy": empirical_joint_energy(truth, empirical),
        })
    true_graph = np.asarray(world.graph, dtype=bool)
    estimated_graph = np.asarray(scm.source_graph, dtype=bool)
    return {
        "true_vs_estimated_source_graph_exact": bool(
            np.array_equal(true_graph, estimated_graph)),
        "true_vs_estimated_source_graph_edge_errors": int(
            np.count_nonzero(true_graph ^ estimated_graph)),
        "noise_diagnostics": noise_metrics,
        "intervention_diagnostics": rows_out,
    }


def run(world_path, model_dir, rows):
    if rows < 32:
        raise ValueError("At least 32 rows per split required")
    world_path, model_dir = Path(world_path), Path(model_dir)
    world = GraphWorld.from_dict(json.loads(world_path.read_text(encoding="utf-8")))
    scm_path = model_dir / "structured" / "explicit_scm.json"
    scm = ExplicitSCM.from_dict(json.loads(scm_path.read_text(encoding="utf-8")))
    if world.nodes != len(scm.equations):
        raise ValueError("True/evaluated SCM dimensions disagree")
    mechanisms = [load_mechanism(model_dir / "baseline" / f"mechanism_{j}.pt")
                  for j in range(world.nodes)]
    fit = world.sample(seed=9100, samples=rows)
    selection = world.sample(seed=9101, samples=rows)
    fit_residuals = symbolic_residuals(fit, scm)
    selection_residuals = symbolic_residuals(selection, scm)
    prefix = world.identity
    proposal = propose(
        fit_residuals, selection_residuals,
        [f"{prefix}:9100:{i}" for i in range(rows)],
        [f"{prefix}:9101:{i}" for i in range(rows)],
    )
    evaluation, eval_u = world.sample(
        seed=9102, samples=rows, return_exogenous=True)
    diagnostics = evaluate(
        world, scm, mechanisms, proposal, evaluation, eval_u, rows)
    return {
        "schema": "ncd.continuous-noise-posthoc-diagnostic.v1",
        "status": "posthoc-development-only",
        "world_id": world.identity,
        "world_sha256": fingerprint(world_path),
        "scm_sha256": fingerprint(scm_path),
        "mechanism_sha256": [
            fingerprint(model_dir / "baseline" / f"mechanism_{j}.pt")
            for j in range(world.nodes)],
        "source_sha256": {
            "runner": fingerprint(__file__),
            "candidate": fingerprint(ROOT / "validation" /
                                     "continuous_noise_candidate_v1.py"),
        },
        "split_seeds": {"fit": 9100, "selection": 9101, "evaluation": 9102},
        "split_rows": rows,
        "same_world_across_splits": True,
        "independent_worlds_for_inference": 0,
        "candidate_residual_target": "observation-minus-executable-symbolic-equation",
        "candidate": proposal,
        "evaluation": diagnostics,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()
    result = run(args.world, args.model_dir, args.rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()