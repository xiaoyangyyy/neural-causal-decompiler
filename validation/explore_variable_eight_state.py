"""Non-certifying variable-width eight-state interval search."""
import json
import numpy as np
from scipy.optimize import differential_evolution

STATES = 8
OUTPUT_TOLERANCE = 0.101
WIDTH_LIMIT = 2 * OUTPUT_TOLERANCE
LAMBDA = 0.5
BETA = 0.4


def uncovered(intervals):
    clipped = sorted((max(0.0, low), min(1.0, high))
                     for low, high in intervals if high > 0 and low < 1)
    last = 0.0
    total = 0.0
    for low, high in clipped:
        if high > last:
            total += high - max(low, last)
            last = high
    return max(0.0, 1.0 - total)


def intervals(parameters):
    weights = np.asarray(parameters[:STATES - 1])
    widths = np.asarray(parameters[STATES - 1:])
    final_left = 1.0 - widths[-1]
    left = np.r_[0.0, final_left * np.cumsum(weights) / np.sum(weights)]
    return left, left + widths


def score(parameters):
    left, right = intervals(parameters)
    initial = uncovered(zip(left, right))
    outside = max(0.0, float(np.max(right - 1.0)))
    action_gaps = []
    for source in range(STATES):
        admissible = [
            ((left[target] - LAMBDA * left[source]) / BETA,
             (right[target] - LAMBDA * right[source]) / BETA)
            for target in range(STATES)
        ]
        action_gaps.append(uncovered(admissible))
    return max(initial, outside, *action_gaps)


def main():
    result = differential_evolution(
        score,
        [(0.01, 1.0)] * (STATES - 1)
        + [(0.05, WIDTH_LIMIT)] * STATES,
        seed=821, popsize=12, maxiter=350,
        polish=True, tol=1e-9,
    )
    left, right = intervals(result.x)
    print(json.dumps({
        "claim": "diagnostic only; a nonzero search result is not an impossibility proof",
        "best_uncovered_fraction": float(result.fun),
        "intervals": [[float(x), float(y)] for x, y in zip(left, right)],
        "evaluations": result.nfev,
    }, indent=2))


if __name__ == "__main__":
    main()

