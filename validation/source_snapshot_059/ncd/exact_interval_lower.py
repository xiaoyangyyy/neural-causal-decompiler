"""Exact, solver-independent replay of scalar interval-chain exclusions.

The proof stream lists each branch in canonical DFS order. A Farkas vector
closes a branch; null descends to the next source. No numerical solver is used
by the verifier.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from .transition_overlap_lower import benchmark_transition_lower_system, _parameters
from .continuous_separation import _digest


def _benchmark():
    system = benchmark_transition_lower_system()
    lam, beta, offset = _parameters(system)
    if offset != 0:
        raise ValueError("Exact interval-chain theorem requires zero transition offset")
    return system, lam, beta


def base_constraints(states: int, epsilon: Fraction):
    if not isinstance(states, int) or states < 1 or epsilon <= 0:
        raise ValueError('Invalid exact interval problem')
    m = states
    rows = []
    def add(items, bound):
        row = [Fraction(0)] * (2*m)
        for index, coefficient in items:
            row[index] += coefficient
        rows.append((tuple(row), bound))
    for i in range(m):
        add(((i, Fraction(1)),), Fraction(1))
        add(((i, Fraction(-1)),), Fraction(0))
        add(((m+i, Fraction(1)),), Fraction(1))
        add(((m+i, Fraction(-1)),), Fraction(0))
        add(((i, Fraction(1)), (m+i, Fraction(-1))), Fraction(0))
        add(((m+i, Fraction(1)), (i, Fraction(-1))), 2*epsilon)
    for i in range(m-1):
        add(((i, Fraction(1)), (i+1, Fraction(-1))), Fraction(0))
        add(((m+i, Fraction(1)), (m+i+1, Fraction(-1))), Fraction(0))
        add(((i+1, Fraction(1)), (m+i, Fraction(-1))), Fraction(0))
    add(((0, Fraction(1)),), Fraction(0))
    add(((2*m-1, Fraction(1)),), Fraction(1))
    add(((2*m-1, Fraction(-1)),), Fraction(-1))
    return tuple(rows)


def chain_constraints(states: int, source: int, chain: tuple[int, ...],
                      lam: Fraction, beta: Fraction):
    m = states
    if (not 0 <= source < m or not chain or any(not 0 <= j < m for j in chain)
            or any(j >= k for j, k in zip(chain, chain[1:]))):
        raise ValueError('Invalid target chain')
    rows = []
    def add(items, bound):
        row = [Fraction(0)] * (2*m)
        for index, coefficient in items:
            row[index] += coefficient
        rows.append((tuple(row), bound))
    first, last = chain[0], chain[-1]
    add(((first, Fraction(1)), (source, -lam)), Fraction(0))
    add(((m+source, lam), (m+first, Fraction(-1))), Fraction(0))
    add(((last, Fraction(1)), (source, -lam)), beta)
    add(((m+source, lam), (m+last, Fraction(-1))), -beta)
    for left, right in zip(chain, chain[1:]):
        add(((right, Fraction(1)), (m+left, Fraction(-1)),
             (m+source, lam), (source, -lam)), Fraction(0))
    return tuple(rows)


def verify_farkas(rows, witness):
    if not isinstance(witness, list) or not witness:
        raise ValueError('Empty Farkas witness')
    n = len(rows[0][0])
    weighted = [Fraction(0)] * n
    bound = Fraction(0)
    previous = -1
    for item in witness:
        if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], int) or isinstance(item[0], bool)
                or not previous < item[0] < len(rows)
                or not isinstance(item[1], str)):
            raise ValueError('Malformed Farkas support')
        index = item[0]
        weight = Fraction(item[1])
        if weight <= 0:
            raise ValueError('Nonpositive Farkas weight')
        previous = index
        row, rhs = rows[index]
        for j, coefficient in enumerate(row):
            weighted[j] += weight * coefficient
        bound += weight * rhs
    if any(weighted) or bound >= 0:
        raise ValueError('Invalid Farkas identity')
    return True


def verify_interval_chain_exclusion(certificate: dict):
    system, lam, beta = _benchmark()
    if (certificate.get('schema') != 'ncd.exact-interval-chain-lower.v1'
            or certificate.get('system_sha256') != _digest(system.to_dict())
            or certificate.get('epsilon') != repr(0.101)):
        raise ValueError('Interval-chain certificate mismatch')
    m = certificate.get('excluded_states')
    if not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= 16:
        raise ValueError('Invalid excluded state count')
    tokens = certificate.get('proof')
    if not isinstance(tokens, list):
        raise ValueError('Missing proof stream')
    epsilon = Fraction(0.101)
    base = base_constraints(m, epsilon)
    chains = [c for length in range(1, m+1)
              for c in combinations(range(m), length)]
    order = [k//2 if k % 2 == 0 else m-1-k//2 for k in range(m)]
    cursor = 0
    closed = 0
    internal = 0
    def walk(depth, rows):
        nonlocal cursor, closed, internal
        if depth == m:
            raise ValueError('Unclosed complete branch')
        source = order[depth]
        for chain in chains:
            if cursor >= len(tokens):
                raise ValueError('Truncated proof stream')
            token = tokens[cursor]
            cursor += 1
            candidate = rows + chain_constraints(m, source, chain, lam, beta)
            if token is None:
                internal += 1
                walk(depth+1, candidate)
            else:
                verify_farkas(candidate, token)
                closed += 1
    walk(0, base)
    if cursor != len(tokens):
        raise ValueError('Trailing proof tokens')
    return {'status': 'verified', 'excluded_states': m,
            'closed_branches': closed, 'internal_branches': internal,
            'proof_tokens': cursor}


def verify_exact_nine_state_upper():
    """Check a rational nine-state inductive realization for every real action."""
    _, lam, beta = _benchmark()
    m = 9
    epsilon = Fraction(0.101)
    radius = Fraction('0.1005')
    centers = tuple(Fraction(i, 10) for i in range(1, 10))
    lows = tuple(max(Fraction(0), c-radius) for c in centers)
    highs = tuple(min(Fraction(1), c+radius) for c in centers)
    point = lows + highs
    if (any(sum((a*x for a,x in zip(row, point)), Fraction(0)) > bound
            for row, bound in base_constraints(m, epsilon))
            or any(max(abs(c-lo), abs(hi-c)) > epsilon
                   for c, lo, hi in zip(centers, lows, highs))):
        raise ValueError('Nine-state initial/output interval proof failed')
    all_chains = [chain for length in range(1, m+1)
                  for chain in combinations(range(m), length)]
    chosen = []
    for source in range(m):
        for chain in all_chains:
            if all(sum((a*x for a, x in zip(row, point)), Fraction(0)) <= bound
                   for row, bound in chain_constraints(m, source, chain, lam, beta)):
                chosen.append(chain)
                break
        else:
            raise ValueError('Nine-state action cover failed')
    return {'status': 'verified', 'states': 9,
            'epsilon': repr(0.101), 'radius': str(radius),
            'target_chains': [list(chain) for chain in chosen],
            'horizon': 'unbounded', 'action_domain': '[0,1]'}


def exact_nine_initial(state):
    """Choose a containing abstract interval for an exact rational state."""
    x = Fraction(state)
    if not 0 <= x <= 1:
        raise ValueError('State outside unit domain')
    radius = Fraction('0.1005')
    for i in range(1, 10):
        center = Fraction(i, 10)
        if max(Fraction(0), center-radius) <= x <= min(Fraction(1), center+radius):
            return i-1
    raise ValueError('Nine-state initial cover failed')


def exact_nine_step(source: int, action):
    """Choose a target covering the entire source relation interval."""
    if not isinstance(source, int) or isinstance(source, bool) or not 0 <= source < 9:
        raise ValueError('Invalid abstract state')
    a = Fraction(action)
    if not 0 <= a <= 1:
        raise ValueError('Action outside unit domain')
    _, lam, beta = _benchmark()
    radius = Fraction('0.1005')
    center = Fraction(source+1, 10)
    low = max(Fraction(0), center-radius)
    high = min(Fraction(1), center+radius)
    next_low, next_high = lam*low+beta*a, lam*high+beta*a
    for target in range(9):
        c = Fraction(target+1, 10)
        if (max(Fraction(0), c-radius) <= next_low
                and next_high <= min(Fraction(1), c+radius)):
            return target
    raise ValueError('Nine-state transition cover failed')
