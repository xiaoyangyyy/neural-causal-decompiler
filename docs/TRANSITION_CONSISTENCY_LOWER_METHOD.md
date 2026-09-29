# A whole-machine lower bound from transition consistency

All eight frozen neural systems require at least 82 states in any
accurate deterministic machine with a fixed state-only output map.
This is independent of a chosen grid, partition or simulation relation.
It uses continuum coverage plus an exact one-step witness, rather than
only packing pairwise-separated initial responses.

## Machine semantics and scope

Let q_0=I(x), q_(t+1)=T(q_t,u_t), and y_t=O(q_t). I is any selector,
T any deterministic time-homogeneous transition, and O any fixed output
map. Require L-infinity error <= e=17/100 for every initial x in [0,1]^d
and every control word in the unit action cube, including times zero
and one. All persistent memory, including a phase label or clock used by
O, must be counted in q. Outputs with access to the original continuous
state or an external uncounted time variable are outside this theorem.
No regularity of I, T or O is assumed.

Observation equals the first four coordinates on the initial cube. This
property is proved from actual network coefficients and fixed phases.
Future observations below are evaluated using the actual neural network,
without assuming its transition is affine or consulting teacher data.

## Lemma: an 81-state machine has forced decoder bands

The 81 packing anchors {0,1/2,1}^4 (hidden coordinates zero) are separated
by 1/2 > 2e. They exhaust every state of a hypothetical 81-state machine.
Each state's decoder is within e of its unique anchor at time zero.

Fix an observed coordinate j and the other three coordinates at any
packing-anchor triple. Vary x_j continuously in [0,1]. A label belonging
to a different anchor on another coordinate has error at least
1/2-e > e there. Consequently only the three labels associated with
self anchors 0,1/2,1 can cover this entire axis slice.

Write their decoder coordinates a_0,a_1,a_2. Initial anchor accuracy
gives their order and a_0<=e, a_2>=1-e. Coverage of the whole slice by
[a_k-e,a_k+e] forces both adjacent gaps <=2e; otherwise an interior
point of [0,1] is uncovered. Hence

    a_0 >= a_2-4e >= 1-5e,
    a_1 >= a_2-2e >= 1-3e,
    a_1 <= a_0+2e <= 3e,
    a_2 <= a_0+4e <= 5e.

For every state and every observed coordinate, decoder values therefore
lie in one of the three necessary bands

    low:    [0.15,0.17]
    middle: [0.49,0.51]
    high:   [0.83,0.85].

In particular, no decoder coordinate lies in the open gap (0.17,0.49).
This conclusion relies on continuum coverage, not just the 81 anchors.

## Lemma: a forced common successor lands in that forbidden gap

Use observed axis j=0 for the accepted models, action u=(1,0), and
t=319/1000. Initial state A has x_(d-1)=1 and all other coordinates zero.
State B has the same predecessor coordinate and additionally x_0=t.
Their other observed coordinates are identical packing anchors.

Because t<1-4e=0.32, the necessary decoder bands force both A and B to
use the same label with low self anchor and those fixed other anchors.
In the 8/32/64/128D family the predecessor is hidden, so this label is
the all-zero observed anchor. Given that same label and the same action,
a deterministic machine must produce the same successor state.

Let L=H_0(F(A,u)) and U=H_0(F(B,u)), computed as exact rationals from the
serialized neural weights. Every accepted model satisfies

    L < 1-4e = 0.32,   U > 2e = 0.34.

A common successor decoder must lie in [U-e,L+e]. Its lower end exceeds
0.17, and its upper end is below 0.49. The whole required interval is
inside the forbidden decoder gap (or empty). No one of the 81 labels
can serve as that successor. This contradicts the assumed machine.

Packing already excludes fewer than 81 states, and this argument excludes
81, so the whole-machine minimum is at least 82. The certified upper
program still uses only 81 initial labels: transition consistency demands
more total states than initialization accuracy alone.

For seed 6101 at 128D, L is approximately 0.2882043989431 and U is
approximately 0.368413843240764. The common decoder would need to lie
approximately in [0.198413843240764,0.4582043989431], inside that gap.
These decimals are illustrations; certificates store exact rationals.

## Replay and limits

`ncd.transition-consistency-lower.v1` binds the actual network, packing,
necessary bands, both concrete states, their observations, common action,
one-step observations and strict gap margins. Replay recomputes everything.
A constant-zero transition with the same initial observation admits an
81-state midpoint machine; a negative regression test requires the
checker to report no witness for that case. Tests also reject forged
bands and changed observation maps.

Replay: `python -m scripts.acceptance_transition_consistency_lower --verify`.
The joint general intervals are [82,132], [82,139], [82,135], [82,136].
The exact global state minimum, shortest program, arbitrary deep-network
families and original end-to-end causal requirements remain open.


An additional independent-domain check uses an affine 4D system with one
control and an observed predecessor. Exact witness replay still proves
the 82-state lower; this checks that the theorem implementation does not
silently require a hidden predecessor or two-action separated dynamics.
Evidence: validation/transition_lower_independent_domain.json. It is not
a substitute for the eight-case installed-package acceptance.
