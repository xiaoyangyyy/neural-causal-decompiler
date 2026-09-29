# Infinite-horizon continuous realization method

## Scope and metric

The frozen benchmark is the two-layer ReLU network in
`ncd.continuous_cover.benchmark_cover_system`. On the unit state and action
cubes its real-valued semantics are coordinate-wise
`x_next = 0.5*x + 0.4*a`, with identity observation. The verification
procedure requires an exact match of the complete serialized network, so the
factorized argument does not silently apply to a coupled or phase-crossing
network.

An abstract state is a vector of grid indices. Its output is the corresponding
grid center. The encoder sends every state in the unit cube to the center of
its grid cell. At every time, the concrete output and abstract output must
differ by at most epsilon in L-infinity norm for every action sequence in
`[0,1]^(d*t)`, with no bound on t.

## Inductive certificate

There are ten representative centers per axis, at 0.05, 0.15, ..., 0.95.
The epsilon is 0.101. The verifier proves two facts using outward-rounded
interval propagation through the actual ReLU network:

1. Every initial interval of width 0.1 has observation error at most epsilon
   relative to its representative.
2. For every representative center, the continuous action interval [0,1] is
   partitioned at nominal nearest-grid thresholds. For each closed segment,
   all concrete states in the epsilon-ball about the center, intersected with
   [0,1], map to the epsilon-ball about the stored target center.

Adjacent action segments share endpoints; the deterministic executable model
uses the right segment on a cut. Both closed segments are certified, so this
tie rule is sound. The certificate stores each target index and numerical
error upper bound. Verification regenerates the exact cuts, target choices,
and interval bounds; it rejects incomplete or altered partitions.

The frozen network is coordinate separable. Once its full serialization is
checked, the one-axis interval results compose by Cartesian product under
L-infinity distance. The initial cover establishes the relation at time zero,
and the transition checks preserve it under every control value. Induction
therefore proves the observation error for arbitrarily many steps, including
infinite control sequences. The concrete transition maps the unit cube into
itself.

## Size interval

The realization has 10^d states. The existing five-point-per-coordinate
packing is rechecked directly at the initial neural output. Every pair has
initial L-infinity distance strictly greater than 2 epsilon, so no single
epsilon-accurate abstract initial output can represent both. Thus any such
realization needs at least 5^d states. The certified intervals are 5 to 10
in one dimension and 25 to 100 in two dimensions. They do not establish
minimality.

This is a transition-closed approximate simulation, not an exact causal
quotient or a general method for arbitrary continuous neural networks.

