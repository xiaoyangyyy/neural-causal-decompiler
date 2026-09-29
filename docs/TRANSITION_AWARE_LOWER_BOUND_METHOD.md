# Transition-aware exact lower-bound method

The 0.32 and 0.33 upper certificates use finite symbolic abstraction.
That general idea has substantial prior work, including Pola, Girard and
Tabuada, *Approximately bisimilar symbolic models for nonlinear control
systems* (2008), https://arxiv.org/abs/0706.0246, and Pola and Tabuada,
*Symbolic Models for Nonlinear Control Systems: Alternating Approximate
Bisimulations* (2007), https://arxiv.org/abs/0707.4205. The project's
scientific target is a checked lower bound on the minimum number of states
of a fixed neural computation, alongside a checked executable upper bound.

A pairwise response packing can miss an obstruction imposed by deterministic
transition closure. Consider a one-dimensional system on [0,1],

    F(x,a) = lambda*x + beta*a,  O(x) = x,

with a in [0,1], positive lambda and beta, and a finite deterministic
realization with m states, fixed state outputs, and uniform error epsilon
for all times and action sequences.

For each abstract state q, collect every concrete state that can be paired
with q along any run starting in the complete initial domain; call this set
B_q. Let I_q be its closed interval hull. Output fidelity puts I_q inside
the epsilon-ball of the abstract output, so length(I_q) <= 2*epsilon.
The I_q cover [0,1], since all initial states must be encoded.

When m=5 and epsilon=0.101, the five hull lengths sum to at most 1.01.
Consequently, every pairwise intersection has length at most 0.01:
the sum of interval lengths exceeds the measure of their union by at most
0.01. At least one hull has length D >= 1/5. Under one fixed abstract state
and a fixed action a, deterministic transition closure forces the full
moving interval J(a)=lambda*I_q + beta*a into one target hull. Its length
is at least lambda/5=0.1.

As a varies continuously, the admissible action sets for target hulls form
a finite closed cover of [0,1]. No single target can handle every action,
because the full sweep has length at least beta+lambda/5=0.5, exceeding
2*epsilon=0.202. Hence two target action sets meet. At their meeting
action, both target hulls contain J(a), so they overlap by at least 0.1.
That contradicts the 0.01 maximum overlap.

For m<5, the elementary length cover bound m*2*epsilon < 1 already rules
out a realization. Thus this argument yields K >= 6, improving the
pairwise packing lower bound 5. The full size interval becomes 6 <= K <= 10
for the frozen 1D continuous benchmark. This is a transition-aware
obstruction: it does not depend on a particular encoder, representative
grid, or candidate abstract transition table.

The implementation in `ncd/transition_overlap_lower.py` extracts the affine
coefficients from the serialized two-layer ReLU network after proving that
all hidden units stay strictly active on the complete input cube. It checks
identity observation and unit-domain invariance. Every exclusion inequality
is then evaluated with exact rational arithmetic on the stored binary
floating-point coefficients, and a separate verifier checks each excluded
size. The formal run is
`runs/certified_transition_lower_seed12701`; its joint acceptance with the
independently replayed 10-state upper realization is
`validation/certified_transition_lower_acceptance.json`.

This certificate rules out every deterministic 1–5-state realization,
regardless of encoder, abstract output placement, or transition table.
It does not rule out 6–9 states. The exact minimum and a transition-aware
lower method for general nonlinear coupled networks remain open.

