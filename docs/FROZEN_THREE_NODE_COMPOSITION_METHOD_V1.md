# Frozen three-node neural mechanism composition, version 1

This method composes three actual frozen mechanism checkpoints from `active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline`. It concerns the *learned deterministic neural mechanism system* and an explicit program extracted from it. No true-world graph, equation, or noise metadata enters program construction or verification. The graph in the theorem is read from the frozen checkpoints: nodes 1 and 2 are roots, and both are declared parents of node 0.

For each root `j = 1, 2`, the verifier exports exact rational values of the stored floating weights and proves that a constant CDIR expression `P_j` differs from the ideal real evaluation of frozen neural mechanism `H_j` by at most `eta_j` on the entire closed box `[-1,1]^3`. The interval proof also encloses both the root neural value and root program value inside `[-1,1]`. The root certificates are generated from the checkpoint bytes and independently recomputed.

The child uses the already accepted version-8 piecewise affine CDIR certificate for the *same* frozen checkpoint. Its full 7,373-node cover, including 3,687 proved leaves and every split boundary, must be independently rechecked. The piecewise program is recomputed from the checked certificate rather than trusted as a separate declaration. The maximum accepted leaf error times the frozen output scale supplies `eta_0`, an absolute local bound over every point of `[-1,1]^3`.

For the child network, let `W_1`, `W_2`, `W_3` be its three stored weight matrices, `s_i > 0` its input normalization scales, and `s_y > 0` its output scale. An exact rational coordinatewise Lipschitz bound is the corresponding row of `s_y |W_3| |W_2| |W_1| diag(1/s_i)`. This follows from `|tanh'(z)| <= 1` over all real inputs. The verifier recomputes every matrix product from the checkpoint export; the program's piecewise branches need not be Lipschitz for the following induction.

For any compatible simultaneous intervention set `S subset {0,1,2}` with observed do values in `[-1,1]`, compare the neural system and extracted program under the *same* intervention values. A replaced node has paired error zero. For a free root, `e_j = eta_j`. For the free child,

`e_0 = eta_0 + L_01 e_1 + L_02 e_2`.

The neural and program parent states both lie in the child proof box, so the triangle inequality first compares the child neural mechanism at the two paired parent states, then applies the local child program certificate at the program parent state. This proves the pointwise coordinate bounds and joint `l1` bound for every one of the eight intervention masks, not just sampled interventions. The verifier checks all eight rows and takes their maximum. It checks the compiled program, root certificates, checkpoint hashes, accepted child proof hashes, and installed historical checker source hashes again from a portable bundle.

The proof uses mathematical real arithmetic on exact serialized weight values and mathematical Tanh. The CLI executor uses floating operations; its finite device rounding is outside the theorem. The theorem also excludes exogenous noise, distributional intervention fidelity, true causal graph recovery, exact true mechanisms, other worlds or graph sizes, and MDL minimality. It therefore supplies a scoped deterministic mechanism-composition result relevant to R10, without proving R4 discovery-network fidelity, R5 internal read/write alignment, or closing any original R0-R13 atomic claim.

The run completed and passed its independent installed-wheel replay and
strict acceptance gate. The exact worst joint l1 upper over the eight masks is
254962767348555191970506817544303392944972042344865871077/
50216813883093446110686315385661331328818843555712276103168,
about 0.00507723903. The portable bundle is
runs/frozen_three_node_composition_v1 and the acceptance receipt is
validation/frozen_three_node_acceptance_v1.json. The paired truth-only
evaluator subsequently proved strict incompatibility at both root nodes;
see docs/FROZEN_THREE_NODE_TRUTH_GAP_METHOD_V1.md. Neither result closes the
original R0-R13 ledger.
