# Next target: certify phase-aware 81-cell initial handoff

The overall original goal remains open. This next step strengthens the
certified-realization path without redefining completion.

## Current evidence

`validation/phase_handoff_probe.json` checks the frozen seed-6101 128D
network and all 128 recurrent relation inequalities. An 81-cell initial
grid (three bins on the first four observed axes, one on every hidden
axis) has positive exact handoff slack about 0.007903950876. The probe
extracts state/action separation from actual nonzero hidden/output weights,
requires two state inputs per output, splits each state-cell rectangle
into exact ReLU phases, and computes extrema from affine polygon forms.
Control quantization uses the maximum exact signed-gradient l1 norm over
action phases. This is exploratory evidence, not a new accepted machine.

The v0.56 108-cell minimum theorem remains valid for its original absolute-
weight sensitivity proof predicate. Exclusion by that predicate does not
imply a concrete unsafe trajectory; this tighter phase calculation
supports the possibility of an 81-cell handoff under the same relation.

## Required implementation and verification

1. Verify the observation is exactly the first four coordinates, zero
   bias, and the recurrent proof is valid at epsilon=17/100.
2. Derive sparse state/action separability and state-support axes from the
   actual frozen network. Reject mixed unsupported hidden units, rather
   than reading a teacher or assuming local ground-truth mechanisms.
3. Prove exact state-cell output extrema by rational phase polygons, cache
   per-output pair-cell patterns, and combine them with the exact action
   phase Lipschitz bound and recurrent quantization radius.
4. Store and replay all coordinate inequalities, phase metadata and hashes
   in a distinct initial-handoff certificate schema. Four directly observed
   unit axes and the independent 81-point packing establish minimum initial
   label count 81 when this grid passes; this is not whole-machine minimality.
5. Make nonlinear closure/program export accept either verified initial
   proof class through explicit schema dispatch. Preserve replay of the
   already released optimal-sensitivity certificates and v0.57 programs.
6. Recompute the full eight-case closure with the new 81-cell initial grid,
   export source-network-free programs, and report every frozen case.
7. Test phase boundaries, unsupported separation, coefficient tampering,
   runtime equality and complete installed-wheel replay before claiming
   any new state bound. Existing v0.57 uppers remain authoritative meanwhile.

Global finite-machine/program minimality, general deep networks and
original end-to-end R4/R5/R8/R9/R10 remain unproved.


The planned proof is now implemented and accepted for all eight models.
See RESULTS_PHASE_INITIAL_HANDOFF.md and the two new method documents.
The original full objective remains open.
