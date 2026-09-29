# Original atomic proof obligations

Requirements snapshot SHA-256: 04d608c94e83c8b77adaca81fcf4942826d3aa7d5d61152d6e895b010fcc724c.

All obligations are retained. Scoped evidence does not silently reduce an original quantifier.
See `runs/original_proof_milestone_v1/ledger.json` for machine-readable target/checkpoint bindings.

## R0.objects (unresolved)

Separate discovery network, discovery program, true SCM and neural mechanisms

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all allowed SCMs and observation laws, including linear Gaussian models.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- acyclic SCM with explicitly stated external noise law; additional identification assumptions require separate evidence

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R0.observational_unique_direction (refuted)

Every allowed observational linear Gaussian SCM has a uniquely recoverable direction

Evidence class: true_causal_correctness.

Quantifier: any observational estimator on the allowed linear Gaussian family.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all sample sizes including population observations.

Interventions: none observed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- independent centered Gaussian noises
- no confounding
- acyclic

Uncovered: Additional identifying assumptions or informative interventions are separate claims.

## R0.uniform_finite_sample_direction (refuted)

Some observational estimator recovers every allowed linear Gaussian direction with error strictly below one half

Evidence class: true_causal_correctness.

Quantifier: any observational estimator on the allowed linear Gaussian family.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all sample sizes including population observations.

Interventions: none observed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- independent centered Gaussian noises
- no confounding
- acyclic

Uncovered: Additional identifying assumptions or informative interventions are separate claims.

## R0.equivalence_contract (unresolved)

Specify fidelity domains and distinguish mathematical from device semantics

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all allowed SCMs and observation laws, including linear Gaussian models.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- acyclic SCM with explicitly stated external noise law; additional identification assumptions require separate evidence

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R1.world_generator (unresolved)

SCM sampling, noises and interventions implement the declared benchmark

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all allowed SCMs and observation laws, including linear Gaussian models.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- acyclic SCM with explicitly stated external noise law; additional identification assumptions require separate evidence

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R1.noise_law (unresolved)

Explicitly justify each declared noise independence and law

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: allowed true SCM family, frozen networks and executable programs as separate objects.

Domain: all allowed SCMs and observation laws, including linear Gaussian models.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identifiability or exact object/semantics specification, not benchmark mean accuracy.

Assumptions and outstanding premises:

- acyclic SCM with explicitly stated external noise law; additional identification assumptions require separate evidence

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R2.sample_invariance (unresolved)

Discovery is invariant to sample order on its declared domain

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R2.variable_equivariance (unresolved)

Variable relabeling commutes with discovery

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R2.discovery_guarantee (unresolved)

State a justified discovery accuracy or identification guarantee

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R3.typed_semantics (unresolved)

CDIR operators, protected operations and branches have explicit executable semantics

Evidence class: program_semantics.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: CDIR operators and frozen finite operator/constant/search grammar.

Domain: all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: semantic equivalence; deterministic serialization does not imply semantic uniqueness.

Assumptions and outstanding premises:

- frozen operator and constant sets
- explicit finite search bounds for class-relative MDL
- unsupported operations remain unresolved

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R3.compositionality (unresolved)

Composed program execution follows the declared semantics

Evidence class: program_semantics.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: CDIR operators and frozen finite operator/constant/search grammar.

Domain: all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: semantic equivalence; deterministic serialization does not imply semantic uniqueness.

Assumptions and outstanding premises:

- frozen operator and constant sets
- explicit finite search bounds for class-relative MDL
- unsupported operations remain unresolved

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R4.program_fidelity (unresolved)

Automatically extract a program with verified fidelity on the declared discovery domain

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure
- candidate search reads only allowed data and frozen network
- shorter candidates must be exhausted to claim bounded-language MDL minimality

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R4.cegis (unresolved)

Use checked counterexamples to refine candidates

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure
- candidate search reads only allowed data and frozen network
- shorter candidates must be exhausted to claim bounded-language MDL minimality

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R4.mdl (unresolved)

Prove minimality only under the explicitly fixed language and search bounds

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire admissible discovery input domain; a local proof box is only scoped evidence.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: classification label agreement for fidelity; graph accuracy reported separately.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure
- candidate search reads only allowed data and frozen network
- shorter candidates must be exhausted to claim bounded-language MDL minimality

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R5.interchange (unresolved)

Verify the fixed neural/program mapping under all declared compatible interventions

Evidence class: internal_mechanism_alignment.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: fixed neural read/write map, executed intermediate variables and program steps.

Domain: all reachable base states, source states and branch paths in the declared abstraction domain.

Interventions: all declared compatible combinations, including independent source worlds per variable.

Error/acceptance: numeric intermediate and final-output bounds, collateral preservation and guard agreement.

Assumptions and outstanding premises:

- fixed mapping before validation
- branch accessibility and local relations must be certified
- a failed map does not exclude its mapping family

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R5.intermediate_numeric (unresolved)

Verify numeric intermediates and collateral preservation

Evidence class: internal_mechanism_alignment.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: fixed neural read/write map, executed intermediate variables and program steps.

Domain: all reachable base states, source states and branch paths in the declared abstraction domain.

Interventions: all declared compatible combinations, including independent source worlds per variable.

Error/acceptance: numeric intermediate and final-output bounds, collateral preservation and guard agreement.

Assumptions and outstanding premises:

- fixed mapping before validation
- branch accessibility and local relations must be certified
- a failed map does not exclude its mapping family

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R5.composition (unresolved)

Compose local relations across multiple computation steps

Evidence class: internal_mechanism_alignment.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: fixed neural read/write map, executed intermediate variables and program steps.

Domain: all reachable base states, source states and branch paths in the declared abstraction domain.

Interventions: all declared compatible combinations, including independent source worlds per variable.

Error/acceptance: numeric intermediate and final-output bounds, collateral preservation and guard agreement.

Assumptions and outstanding premises:

- fixed mapping before validation
- branch accessibility and local relations must be certified
- a failed map does not exclude its mapping family

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R6.separate_targets (unresolved)

Report network fidelity, truth accuracy, graph and mechanism errors separately

Evidence class: evaluation_contract.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen neural/program pair, independent true SCM evaluator and world-level statistics.

Domain: all registered experiments, including failed runs and confirmation worlds.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: separate network fidelity, graph, true mechanism, noise and intervention errors; familywise confidence 99 percent.

Assumptions and outstanding premises:

- independent world units
- no reuse of evaluator truth in candidate search

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R6.evaluation_guarantee (unresolved)

Replay comparisons and justify world-level statistical units

Evidence class: evaluation_contract.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen neural/program pair, independent true SCM evaluator and world-level statistics.

Domain: all registered experiments, including failed runs and confirmation worlds.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: separate network fidelity, graph, true mechanism, noise and intervention errors; familywise confidence 99 percent.

Assumptions and outstanding premises:

- independent world units
- no reuse of evaluator truth in candidate search

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R7.counterexamples (unresolved)

Return independently checked concrete counterexamples

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire declared closed input domain including branch boundaries and degenerate inputs.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: rigorous point counterexample or full-domain worst-case enclosure; no omitted hard regions.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure
- strict rational numerical enclosures
- search exhaustion is unresolved, not a universal refutation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R7.worst_case (unresolved)

Certify any claimed global worst-case bound on its entire declared domain

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery network and extracted discovery program.

Domain: entire declared closed input domain including branch boundaries and degenerate inputs.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: rigorous point counterexample or full-domain worst-case enclosure; no omitted hard regions.

Assumptions and outstanding premises:

- a frozen target, declared input semantics and domain must be bound before closure
- strict rational numerical enclosures
- search exhaustion is unresolved, not a universal refutation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R8.function_ood (unresolved)

Justify claims on the declared function shift family

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery and mechanism networks, programs and true benchmark SCMs.

Domain: declared function-shift family; unbounded universal OOD claims are not substituted by a finite benchmark.

Interventions: declared node interventions and compatible combinations within that family.

Error/acceptance: truth accuracy and neural/program fidelity separately; numeric and distributional intervention errors.

Assumptions and outstanding premises:

- function, noise, scale, intervention support and sampling law frozen before confirmation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R8.noise_ood (unresolved)

Justify claims on the declared noise shift family

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery and mechanism networks, programs and true benchmark SCMs.

Domain: declared noise-shift family; unbounded universal OOD claims are not substituted by a finite benchmark.

Interventions: declared node interventions and compatible combinations within that family.

Error/acceptance: truth accuracy and neural/program fidelity separately; numeric and distributional intervention errors.

Assumptions and outstanding premises:

- function, noise, scale, intervention support and sampling law frozen before confirmation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R8.scale_ood (unresolved)

Justify claims on the declared scale shift family

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery and mechanism networks, programs and true benchmark SCMs.

Domain: declared scale-shift family; unbounded universal OOD claims are not substituted by a finite benchmark.

Interventions: declared node interventions and compatible combinations within that family.

Error/acceptance: truth accuracy and neural/program fidelity separately; numeric and distributional intervention errors.

Assumptions and outstanding premises:

- function, noise, scale, intervention support and sampling law frozen before confirmation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R8.intervention_ood (unresolved)

Justify claims on the declared intervention family

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen discovery and mechanism networks, programs and true benchmark SCMs.

Domain: declared intervention-shift family; unbounded universal OOD claims are not substituted by a finite benchmark.

Interventions: declared node interventions and compatible combinations within that family.

Error/acceptance: truth accuracy and neural/program fidelity separately; numeric and distributional intervention errors.

Assumptions and outstanding premises:

- function, noise, scale, intervention support and sampling law frozen before confirmation

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R9.graph_recovery (unresolved)

Recover an identified DAG or justified graph equivalence class on 3/5/8 nodes

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: 3/5/8-node discovery networks, extracted graph computation and true SCM evaluator.

Domain: full declared 3/5/8-node observation or intervention task, separated by information access.

Interventions: none for observational task; recorded node-do coverage for interventional task.

Error/acceptance: identified DAG or valid equivalence class; network/program graph labels checked separately.

Assumptions and outstanding premises:

- observational directions require identification premises
- DAG completion and oracle diagnostics do not prove identification

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R9.sepsets (unresolved)

Tie separations and orientation decisions to evidence rather than inserted teacher truth

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: 3/5/8-node discovery networks, extracted graph computation and true SCM evaluator.

Domain: full declared 3/5/8-node observation or intervention task, separated by information access.

Interventions: none for observational task; recorded node-do coverage for interventional task.

Error/acceptance: identified DAG or valid equivalence class; network/program graph labels checked separately.

Assumptions and outstanding premises:

- observational directions require identification premises
- DAG completion and oracle diagnostics do not prove identification

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R9.neural_program (unresolved)

Extract and verify the network graph-discovery computation

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: 3/5/8-node discovery networks, extracted graph computation and true SCM evaluator.

Domain: full declared 3/5/8-node observation or intervention task, separated by information access.

Interventions: none for observational task; recorded node-do coverage for interventional task.

Error/acceptance: identified DAG or valid equivalence class; network/program graph labels checked separately.

Assumptions and outstanding premises:

- observational directions require identification premises
- DAG completion and oracle diagnostics do not prove identification

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R10.end_to_end (unresolved)

Recover an explicit SCM using inferred parents without oracle repairs

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen conditional neural mechanisms and inferred explicit SCM including graph, equations and exogenous laws.

Domain: entire declared mechanism/intervention domain for the inferred parent sets.

Interventions: all declared node-do values and compatible simultaneous interventions.

Error/acceptance: neural-symbolic mechanism error <=1/100 of frozen training scale; joint intervention distribution distance requires certified noise coupling.

Assumptions and outstanding premises:

- correct parent sets or explicit conditional theorem
- local Lipschitz and domain coverage
- noise law, independence and coupling distance separately justified

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R10.mechanism (unresolved)

Certify the frozen neural-to-symbolic mechanism approximation

Evidence class: network_fidelity.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen conditional neural mechanisms and inferred explicit SCM including graph, equations and exogenous laws.

Domain: entire declared mechanism/intervention domain for the inferred parent sets.

Interventions: all declared node-do values and compatible simultaneous interventions.

Error/acceptance: neural-symbolic mechanism error <=1/100 of frozen training scale; joint intervention distribution distance requires certified noise coupling.

Assumptions and outstanding premises:

- correct parent sets or explicit conditional theorem
- local Lipschitz and domain coverage
- noise law, independence and coupling distance separately justified

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R10.noise (unresolved)

Justify exogenous laws and independence rather than silently assuming empirical residual validity

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen conditional neural mechanisms and inferred explicit SCM including graph, equations and exogenous laws.

Domain: entire declared mechanism/intervention domain for the inferred parent sets.

Interventions: all declared node-do values and compatible simultaneous interventions.

Error/acceptance: neural-symbolic mechanism error <=1/100 of frozen training scale; joint intervention distribution distance requires certified noise coupling.

Assumptions and outstanding premises:

- correct parent sets or explicit conditional theorem
- local Lipschitz and domain coverage
- noise law, independence and coupling distance separately justified

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R10.intervention_distribution (unresolved)

Certify intervention effects and distribution bounds with verified premises

Evidence class: true_causal_correctness.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: frozen conditional neural mechanisms and inferred explicit SCM including graph, equations and exogenous laws.

Domain: entire declared mechanism/intervention domain for the inferred parent sets.

Interventions: all declared node-do values and compatible simultaneous interventions.

Error/acceptance: neural-symbolic mechanism error <=1/100 of frozen training scale; joint intervention distribution distance requires certified noise coupling.

Assumptions and outstanding premises:

- correct parent sets or explicit conditional theorem
- local Lipschitz and domain coverage
- noise law, independence and coupling distance separately justified

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R11.ablations (unresolved)

Preserve and independently replay all declared ablations without selective reporting

Evidence class: engineering_reproduction.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: all declared source, wheel, checkpoint, ablation and executable-program artifacts.

Domain: all registered release/ablation cases, including failures and isolated installed execution.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identity and independent replay; passing reproduction does not establish causal truth.

Assumptions and outstanding premises:

- frozen hashes and requirements
- external extension tasks require their own specifications

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R12.semantic_equivalence (unresolved)

Certify equivalence in the declared operator subset

Evidence class: program_semantics.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: CDIR operators and frozen finite operator/constant/search grammar.

Domain: all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: semantic equivalence; deterministic serialization does not imply semantic uniqueness.

Assumptions and outstanding premises:

- frozen operator and constant sets
- explicit finite search bounds for class-relative MDL
- unsupported operations remain unresolved

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R12.canonicalization (unresolved)

Separate deterministic serialization from semantic uniqueness

Evidence class: program_semantics.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: CDIR operators and frozen finite operator/constant/search grammar.

Domain: all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: semantic equivalence; deterministic serialization does not imply semantic uniqueness.

Assumptions and outstanding premises:

- frozen operator and constant sets
- explicit finite search bounds for class-relative MDL
- unsupported operations remain unresolved

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R12.shortest_unique (unresolved)

Justify any uniqueness or globally shortest-program claim

Evidence class: program_semantics.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: CDIR operators and frozen finite operator/constant/search grammar.

Domain: all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: semantic equivalence; deterministic serialization does not imply semantic uniqueness.

Assumptions and outstanding premises:

- frozen operator and constant sets
- explicit finite search bounds for class-relative MDL
- unsupported operations remain unresolved

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R13.reproduction (unresolved)

Provide portable artifacts, source bindings and independent installed-package replay

Evidence class: engineering_reproduction.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: all declared source, wheel, checkpoint, ablation and executable-program artifacts.

Domain: all registered release/ablation cases, including failures and isolated installed execution.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identity and independent replay; passing reproduction does not establish causal truth.

Assumptions and outstanding premises:

- frozen hashes and requirements
- external extension tasks require their own specifications

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.

## R13.extensions (unresolved)

Distinguish defined external tasks from architecture visions

Evidence class: engineering_reproduction.

Quantifier: all instances in the stated original scope; instance binding alone does not reduce this quantifier.

Target: all declared source, wheel, checkpoint, ablation and executable-program artifacts.

Domain: all registered release/ablation cases, including failures and isolated installed execution.

Interventions: observational input changes; no identifying intervention assumed.

Error/acceptance: identity and independent replay; passing reproduction does not establish causal truth.

Assumptions and outstanding premises:

- frozen hashes and requirements
- external extension tasks require their own specifications

Uncovered: Historical module/test existence or a scoped certificate does not close this original scientific claim.
