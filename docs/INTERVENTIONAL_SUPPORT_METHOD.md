# Interventional support certificates for frozen ReLU transitions

Let a fixed recurrent transition be F: [0,1]^(d+u) -> R^d, where the first d
inputs are declared state coordinates and the remaining u are declared action
coordinates. The one-step intervention support contains j -> i exactly when
there exist two inputs in the full unit cube differing only in coordinate j
whose F_i values differ. This is a coordinate-level, one-step counterfactual
property. It is not a recovered latent SCM or a claim about reachable states.

The workflow has a deliberately asymmetric trust boundary.

1. `propose_support` receives only a callable `step(state, action)` and the
   dimensions. It evaluates five rational levels per coordinate while holding
   every other coordinate at 1/2. It stores the first observed difference for
   each proposed output-input pair. The oracle stage never reads weights.
2. `certify_support` independently reads the frozen serialized ReLU network.
   It evaluates each proposed pair with exact rational arithmetic, interpreting
   every serialized binary float coefficient as an exact rational. A nonzero
   difference proves a support edge. It then propagates Boolean nonzero-weight
   reachability through every layer. A pair with no path is provably absent on
   the entire cube; a pair with a path but no exact witness stays unresolved.
3. `verify_case` regenerates the behavior-query proposal from the model's
   callable transition, compares it byte-for-byte as structured data, then
   regenerates and compares the certificate. The stored model digest binds
   evidence to the frozen network.

The distinction between presence and absence is necessary. For any finite
set of queried levels, a small piecewise-linear ReLU tent can be placed
between them. It is zero at every queried level yet depends on that input
coordinate elsewhere. Therefore no finite behavioral query transcript alone
can certify absence for the unrestricted ReLU class. Structural zero-path
checking is a sound sufficient absence rule, though not a complete one:
two nonzero paths can cancel, leaving a truly absent edge unresolved.

The support graph becomes a complete certified graph only when every
structurally possible pair has an exact intervention witness. This criterion
does not require assuming that every nonzero weight path is functionally
active. It also refuses to call a zero response at the five probe levels
a proof of absence.

The certificate targets the mathematical ReLU function with exact values
of the serialized float coefficients. A floating-point runtime may round
near a boundary. Oracle differences merely propose candidates; the exact
checker decides which edges are established. Proposal regeneration can be
platform-sensitive if a floating result lies near the query threshold; the
exact witness and zero-path certificate do not rely on that threshold.
Query-call counts describe the fixed probe protocol, not an
information-theoretic optimum.

Run a model with:

    python -m ncd.interventional_support path/to/system.json output/dir
    python -m ncd.interventional_support path/to/system.json output/dir --verify

Formal study: `python -m scripts.acceptance_interventional_support`.
Evidence lives in `runs/interventional_support_global_v1` and
`validation/interventional_support_acceptance.json`.

The next research problem is to turn unresolved cases into stronger
function-level absence certificates, for example via exact ReLU region
reasoning or proof-producing solver search, without assuming that network
sparsity equals causal support. More importantly, the declared coordinates
remain fixed; recovering latent causal variables and minimal quotients from
general neural representations remains open.
