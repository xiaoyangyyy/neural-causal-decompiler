# Generic certified grid realization for continuous ReLU systems

## Contract

Input is a serialized `ContinuousReLUSystem` with a deterministic ReLU
transition `F(x,a)` and observation `O(x)`. The declared state and action
domains are full unit cubes. The construction does not compare network
parameters against a frozen benchmark. A system hash binds every certificate
to its actual network. The current implementation uses a uniform finite state
grid and a uniform continuous-action box partition.

The candidate model has one state for each grid center `r_q`. Its output is
`O(r_q)`. The encoder maps an initial state to its grid-cell center.
For an action, the executable transition table chooses a target `q'`
according to the action box containing it. Shared action-box boundaries use
the right box; each adjacent closed box is separately certified.

## Proof obligations

Let `delta` be the internal state relation radius and `epsilon` the output
error tolerance. Keeping them separate matters: even for identity observation,
an outward-rounded interval bound for a delta-radius box may exceed delta by a
few floating-point ulps.

The verifier independently checks:

1. Every initial grid cell lies inside the delta-ball of its representative.
2. For every representative, interval propagation through the actual
   observation network bounds the output difference between the full
   delta-ball and `O(r_q)` by epsilon.
3. For every representative and every continuous action box, interval
   propagation through the actual transition network places the whole image
   of `B_delta(r_q) intersect [0,1]^d` inside both the unit cube and the
   delta-ball of the stored target.
4. The state grid, action partition, target indices, interval endpoints,
   stored bounds, system hash, and aggregate status match the declared
   network and domains.
5. Every pair in the optional initial-output packing is checked from the
   actual observation network. A pair is counted only when its certified
   distance exceeds `2*epsilon`.

Items 1–3 establish an inductive simulation relation. Thus the neural and
finite-machine outputs differ by at most epsilon for every initial state and
every finite prefix of every continuous action sequence, with no horizon
limit. The verifier returns `unresolved`, with no upper-bound claim, whenever
a grid fails; this is not a proof of impossibility.

The independent checker evaluates the stored transition targets directly.
The generator proposes targets from nominal grid quantization, but the
checker does not rely on that heuristic. Artifact replay additionally
regenerates the candidate and checks byte-level file hashes.

## Nonlinear coupled demonstration

The two-state-coordinate benchmark uses

`F_i(x,a)=0.1+0.25*x_i+0.10*x_j+0.20*a_i+0.05*ReLU(x_i-x_j)`,

where `j` is the other coordinate. Its first output depends on the second
state coordinate, and `x_i=x_j` is a ReLU phase boundary crossed by
certified state boxes. The network is evaluated as a two-layer ReLU MLP,
rather than replaced by the displayed formula during certification.

With ten state bins and ten action bins per coordinate, delta is 0.115 and
epsilon is 0.12. The verifier checks 100 initial cells, 100 observation
relations, and 10,000 full state-action transition boxes. The maximum
outward-rounded transition bound is 0.1092500000000016 < delta.
Five packing points per coordinate give 25 pairwise separated points and
a lower bound of 25; the candidate has 100 states.

## Boundary

The checker accepts any network represented by the supported ReLU MLP class,
but a fixed grid may return `unresolved`. The successful example is a
constructed, two-dimensional neural system. The method is exponential in
state and action dimension, the size interval does not close, and no
trained-network or general minimality claim follows.

