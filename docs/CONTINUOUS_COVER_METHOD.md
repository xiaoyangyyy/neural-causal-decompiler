# Global continuous behavioral-cover method

## Objective

This stage asks for the minimum number of response representatives needed to
approximate every initial state in the complete continuous unit cube. The
metric is finite-horizon behavioral distance:

D(x,y) = sup over allowed action words of the maximum-time L-infinity distance
between the two neural response traces.

An epsilon cover contains representatives r_k such that every initial state x
has some representative with D(x,r_k) at most epsilon.

## Certified upper bound

Each unit-cube coordinate is divided into five intervals. Their Cartesian
products form 5^d closed boxes. The representative of each box is its center.

For every box, a continuous-region certificate proves that every state in the
box stays within epsilon of its representative for every action word in the
complete control box. The verifier checks all cell geometries against the
declared grid, so the cells cover the complete unit cube without relying on
sampling.

This produces upper bound K <= 5^d.

## Certified lower bound

The packing grid contains five points per coordinate at 0, 0.25, 0.5, 0.75,
and 1. Every distinct pair differs by at least 0.25 in one coordinate. Stored
point-state certificates prove behavioral distance greater than 2*epsilon for
every pair.

By the triangle inequality, two such packing points cannot share one
epsilon-accurate response representative. The complete pair set therefore
proves K >= 5^d.

The formal epsilon is 0.101, so the cell radius 0.1 is inside the cover
tolerance and packing spacing 0.25 is strictly greater than 2*epsilon = 0.202.

## Neural benchmark

The ReLU system has coordinate-wise dynamics

x_next = 0.5*x + 0.4*a

and identity observation. It is represented as a two-layer ReLU network using
strictly positive hidden preactivations on the complete unit state/action
domain. The action dimension equals the state dimension. Every certificate
covers all action words in [0,1]^(d*horizon), with horizon three.

## Verification

The independent verifier checks:

- system hashes and exact unit-cube grid coverage;
- every cell-to-representative region upper certificate;
- the complete packing point list and every unordered pair;
- every separation witness and numerical bound;
- every child proof is bound to the complete declared action box;
- matching lower and upper counts;
- artifact hashes and deterministic full regeneration.

## Claim boundary

The result is an exact minimum finite-horizon behavioral cover number for the
declared systems and domains. It is not a transition-closed quotient: a
representative's next state is not required to be another representative.
It also makes no infinite-horizon, arbitrary-network, or general nonlinear
state-space claim.
