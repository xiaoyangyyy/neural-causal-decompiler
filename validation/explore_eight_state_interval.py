"""Diagnostic search for an eight-interval simulation cover; not a certificate."""
import json
import numpy as np
from scipy.optimize import differential_evolution

M = 8
EPS = 0.101
WIDTH = 2 * EPS
LAMBDA = 0.5
BETA = 0.4


def uncovered(intervals):
    boxes = [(max(0.0, low), min(1.0, high)) for low, high in intervals]
    boxes = sorted((low, high) for low, high in boxes if high > low)
    covered = 0.0
    end = 0.0
    for low, high in boxes:
        covered += max(0.0, high - max(low, end))
        end = max(end, high)
    return max(0.0, 1.0 - covered)


def score(raw):
    left = np.r_[0.0, np.sort(raw), 1.0 - WIDTH]
    right = left + WIDTH
    initial_gap = uncovered(zip(left, right))
    action_gaps = []
    for source in range(M):
        actions = []
        for target in range(M):
            action_low = (left[target] - LAMBDA * left[source]) / BETA
            action_high = (right[target] - LAMBDA * right[source]) / BETA
            actions.append((action_low, action_high))
        action_gaps.append(uncovered(actions))
    return max(initial_gap, *action_gaps)


def main():
    result = differential_evolution(
        score, [(0.0, 1.0 - WIDTH)] * (M - 2),
        seed=735, popsize=12, maxiter=220,
        polish=True, tol=1e-8,
    )
    left = np.r_[0.0, np.sort(result.x), 1.0 - WIDTH]
    print(json.dumps({
        "claim": "diagnostic only; nonzero score cannot prove impossibility",
        "best_uncovered_fraction": float(result.fun),
        "left_endpoints": left.tolist(),
        "evaluations": result.nfev,
    }, indent=2))


if __name__ == "__main__":
    main()

