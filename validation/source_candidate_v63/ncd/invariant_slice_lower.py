"""Invariant-slice lift of a scalar transition lower bound to separable systems.

Each invariant second-coordinate slice carries an independent copy of the
scalar lower-bound problem. Distinct slices with outputs separated by more
than 2 epsilon cannot share any abstract state, even at later times.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from math import ceil
from pathlib import Path

from .affine_observability import _affine_network, Unresolved as AffineUnresolved
from .continuous_separation import ContinuousReLUSystem, _digest
from .io import read_json, save_json
from .multiswitch_lower import MultiSwitchConfig, certified_multiswitch_lower, verify_multiswitch_lower
from .exact_interval_lower import verify_interval_chain_exclusion


class Unresolved(Exception):
    """Frozen model does not satisfy the separable invariant-slice theorem."""


def _rational(x):
    return str(x)


def _parameters(scalar: ContinuousReLUSystem, product: ContinuousReLUSystem):
    if (scalar.state_dim != 1 or scalar.action_dim != 1
            or product.state_dim != 2 or product.action_dim != 2):
        raise Unresolved("Expected one scalar and one two-coordinate controlled system")
    sf, sb, _ = _affine_network(scalar.transition)
    so, sc, _ = _affine_network(scalar.observation)
    pf, pb, _ = _affine_network(product.transition)
    po, pc, _ = _affine_network(product.observation)
    lam, beta = sf[0]
    offset = sb[0]
    if (so != ((Q(1),),) or sc != (Q(0),)
            or po != ((Q(1), Q(0)), (Q(0), Q(1)))
            or pc != (Q(0), Q(0))
            or pf != ((lam, Q(0), beta, Q(0)),
                      (Q(0), lam, Q(0), beta))
            or pb != (offset, offset)):
        raise Unresolved("Product network is not two exact copies of the scalar system")
    if lam < 0 or lam >= 1 or beta <= 0 or offset < 0 or lam + beta + offset > 1:
        raise Unresolved("Scalar dynamics are not invariant contractions")
    return lam, beta, offset


def _compute(scalar: ContinuousReLUSystem, product: ContinuousReLUSystem,
             epsilon: str = "0.101", max_states: int = 20):
    base = {
        "schema": "ncd.invariant-slice-lower.v1",
        "scalar_sha256": _digest(scalar.to_dict()),
        "product_sha256": _digest(product.to_dict()),
        "epsilon": epsilon,
        "max_states": max_states,
        "domain": "full unit initial states and continuous unit actions",
        "horizon": "all finite action words",
    }
    try:
        lam, beta, offset = _parameters(scalar, product)
        e = Q(float(epsilon))
        if e <= 0:
            raise ValueError("epsilon must be positive")
        scalar_certificate = certified_multiswitch_lower(
            scalar, MultiSwitchConfig(epsilon=float(e), max_states=max_states))
        scalar_result = verify_multiswitch_lower(scalar, scalar_certificate)
        scalar_lower = scalar_result["lower_bound"]
        fixed_low = max(Q(0), offset / (1 - lam))
        fixed_high = min(Q(1), (offset + beta) / (1 - lam))
        if fixed_low > fixed_high:
            raise Unresolved("No unit-domain invariant fixed-point slice")
        span = fixed_high - fixed_low
        count = max(1, ceil(span / (2 * e)))
        if count == 1:
            centers = [fixed_low]
        else:
            centers = [fixed_low + i * span / (count - 1)
                       for i in range(count)]
        actions = [((1 - lam) * c - offset) / beta for c in centers]
        if (any(not 0 <= a <= 1 for a in actions)
                or any(centers[i + 1] - centers[i] <= 2 * e
                       for i in range(count - 1))):
            raise Unresolved("Invariant slice packing did not separate")
        return {
            **base,
            "status": "certified-lower-bound",
            "lambda": _rational(lam),
            "beta": _rational(beta),
            "offset": _rational(offset),
            "epsilon_exact_float": _rational(e),
            "scalar_lower_bound": scalar_lower,
            "scalar_certificate_sha256": _digest(scalar_certificate),
            "fixed_interval": [_rational(fixed_low), _rational(fixed_high)],
            "slices": [{"second_state": _rational(c),
                        "fixed_second_action": _rational(a)}
                       for c, a in zip(centers, actions)],
            "slice_count": count,
            "lower_bound": count * scalar_lower,
            "proof": "disjoint invariant output slices times scalar transition lower bound",
        }
    except (Unresolved, AffineUnresolved, ValueError) as error:
        return {**base, "status": "unresolved", "reason": str(error)}


def certify_slice_lower(scalar: ContinuousReLUSystem,
                        product: ContinuousReLUSystem, *,
                        epsilon: str = "0.101", max_states: int = 20):
    return _compute(scalar, product, epsilon=epsilon, max_states=max_states)


def verify_slice_lower(scalar: ContinuousReLUSystem,
                       product: ContinuousReLUSystem, certificate: dict):
    if certificate.get("schema") != "ncd.invariant-slice-lower.v1":
        raise ValueError("Unsupported invariant-slice certificate")
    expected = _compute(scalar, product, epsilon=certificate["epsilon"],
                        max_states=certificate["max_states"])
    if certificate != expected:
        raise ValueError("Invariant-slice certificate replay mismatch")
    return {"status": expected["status"],
            "lower_bound": expected.get("lower_bound"),
            "slice_count": expected.get("slice_count")}


def run_case(scalar: ContinuousReLUSystem, product: ContinuousReLUSystem,
             output: Path, *, epsilon: str = "0.101"):
    output = Path(output)
    certificate = certify_slice_lower(scalar, product, epsilon=epsilon)
    save_json(output / "certificate.json", certificate)
    return verify_case(scalar, product, output)


def verify_case(scalar: ContinuousReLUSystem, product: ContinuousReLUSystem,
                output: Path):
    return verify_slice_lower(
        scalar, product, read_json(Path(output) / "certificate.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar_model", type=Path)
    parser.add_argument("product_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epsilon", default="0.101")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    scalar = ContinuousReLUSystem.from_dict(read_json(args.scalar_model))
    product = ContinuousReLUSystem.from_dict(read_json(args.product_model))
    result = (verify_case(scalar, product, args.output) if args.verify else
              run_case(scalar, product, args.output, epsilon=args.epsilon))
    print(result)


if __name__ == "__main__":
    main()


def certify_exact_slice_lower(scalar: ContinuousReLUSystem,
                              product: ContinuousReLUSystem,
                              scalar_exclusion: dict) -> dict:
    """Lift a replayed exact eight-state scalar exclusion to the 2D product."""
    old = _compute(scalar, product)
    if old["status"] != "certified-lower-bound" or old["scalar_lower_bound"] != 7:
        raise ValueError("Historical scalar/slice proof did not replay")
    excluded = verify_interval_chain_exclusion(scalar_exclusion)
    if (excluded["excluded_states"] != 8
            or scalar_exclusion["system_sha256"] != _digest(scalar.to_dict())):
        raise ValueError("Exact scalar exclusion does not match the slice model")
    scalar_lower = 9
    return {
        "schema": "ncd.exact-invariant-slice-lower.v1",
        "scalar_sha256": _digest(scalar.to_dict()),
        "product_sha256": _digest(product.to_dict()),
        "epsilon": old["epsilon"],
        "scalar_exclusion_sha256": _digest(scalar_exclusion),
        "historical_slice_sha256": _digest(old),
        "slices": old["slices"],
        "slice_count": old["slice_count"],
        "scalar_lower_bound": scalar_lower,
        "lower_bound": old["slice_count"] * scalar_lower,
        "proof": "disjoint invariant output slices times certified exact scalar exclusion",
    }


def verify_exact_slice_lower(scalar: ContinuousReLUSystem,
                             product: ContinuousReLUSystem,
                             scalar_exclusion: dict, certificate: dict) -> dict:
    expected = certify_exact_slice_lower(scalar, product, scalar_exclusion)
    if certificate != expected:
        raise ValueError("Exact invariant-slice certificate replay mismatch")
    return {"status": "verified", "scalar_lower_bound": 9,
            "slice_count": expected["slice_count"],
            "lower_bound": expected["lower_bound"]}
