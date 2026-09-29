# Unified proof registry, isolated 0.60.0.dev1 candidate

The live core remains 0.59 while the original confirmation and subsequent
piecewise proof use their frozen sources. The candidate is in
`validation/source_candidate_v60` and independently installed into
`validation/wheel_v60_env`. It is not promoted to the live source tree.

## Public entry points

The candidate `ncd prove --config ... [--resume]`, `ncd verify-proof ...`, and
`ncd audit-requirements ... --require-closed` accept a new
`ncd.unified-proof-plan.v2` protocol. Legacy v1 entry points remain dispatched
to the original workflow. The registered v2 plan has seven jobs:

- original six-job milestone and its independently derived original ledger;
- five scoped proof-extension jobs;
- four scoped workbench jobs;
- actual normalization-prefix composition and 96 diagnostic interventions;
- exact equality of the binary-constant expression to 32*x0;
- exact non-equality of that expression to zero;
- the pending actual two-parent mechanism certificate on the full declared box.

A collection's `verified` result is artifact replay, not a scientific conclusion
that all its candidates succeeded. The nested results retain each proved,
refuted or unresolved scoped conclusion. Unsupported polynomial operators or
exhausted certificate bounds have explicit unresolved records. The native
polynomial budget is at most 64 AST nodes and degree 8, with 1-8 variables.

## Scope and completeness

The protocol binds the unchanged FULL_REQUIREMENTS, every candidate module,
each historical manifest, and the original target mechanism specification.
The bundle binds both its copied configuration and its external frozen protocol.
A missing/reordered job, changed source/weight identity, omitted original atom,
or a local-result promotion to original closure is rejected.

Historical replay runs in a separate `python -I` process, with the approved
0.59 source snapshot and its independently installed packages. The runner checks
the exact module sets and installed/source byte identity. Three approved source
manifest hashes are fixed in the candidate verifier; a caller cannot substitute
an arbitrary snapshot source list. This preserves the legacy counterexample
against the old checker while the new core uses corrected semantics.

The original ledger is derived from the independently replayed milestone. Only
the two already justified Gaussian observational refutations close original atoms.
New records attach as scoped evidence with `entails_original_claim=false`. The
38 original atoms remain present, including all 36 unresolved ones. Strict
completion audit therefore returns 1. This registry does not manufacture a
scope-matched proof for those remaining claims.

Resume retains a complete prior bundle revision before re-evaluating open jobs.
Closed records are independently replayed. A retained unresolved record never
automatically adopts a later stronger result; an explicit resumed generation is
needed. Its open conclusion remains conservative when new evidence appears.

## Correctness repairs in the candidate

CDIR `to_sympy` now converts a binary64 constant through `as_integer_ratio`.
The old decimal spelling conversion erased the exact 32*x0 term in
`((.1+.2-.3)*2**60)*x0`. The new native polynomial certificate independently
recomputes rational coefficient normal forms; it distinguishes mathematical
function equality from device execution and internal causal equivalence.
The general symbolic API is not claimed to decide all transcendentals.

The historical Rule proof now lowers its features through the actual
`statistics.extract_one` semantics: the correct moment/variance floors,
train-only quantile/RBF basis, ridge regularization and response normalization.
Its proof schema is v2. CDIR crossfit_prediction remains a different operator.
Sorting/rank/branch uncertainty remains unresolved. Old v1 certificates replay
through the old snapshot; the corrected actual-statistics revalidation is retained.

## Historical floating reproducibility

The original CEGIS verifier also regenerated each fitted candidate exactly.
A historical checkpoint matches the saved candidate under six inference threads;
1, 2 and 4 threads change fitted coefficients by about 1e-8. This was detected,
not hidden with a tolerance. The failure, initial candidate source and protocol
are retained under `validation/source_candidate_v60_attempt_0000` and the
initial `runs/original_proof_registry_v2` attempt. The successful r1 protocol
explicitly binds historical inference to six threads. No neural training occurs
in this historical reproduction. New confirmation training stays at two threads.
`validation/historical_cegis_thread_diagnostic_v1.json` records the comparisons.
Candidate numerical certificate validity is a separate issue from exact search
process reproducibility.

## Resources and validation

Independent workers enter a Windows job before being resumed, so the venv
launcher and its real Python descendants share the configured 8 GiB committed
memory limit. Closing that job after a timeout terminates only that owned process
family. Controlled tests show a descendant's 256 MiB allocation rejected under
a 128 MiB job limit and confirm descendant cleanup on timeout. Windows peak
counters are recorded as reported; the orchestration also treats any reported
peak over budget as unresolved. The API contract is described by
[Microsoft's job-limit documentation](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
The previous confirmation guard is not retroactively changed or relabeled.

Sixteen targeted source/installed tests pass. The isolated candidate runs all
three public proof commands, and its strict audit correctly refuses completion.
The complete core regression is separately frozen at 249 historical tests plus
14 new registry tests, and waits until both real heavy process families end.
Its stage permits one worker, two training threads, 8 GiB memory/artifacts and
12 hours. Process observation uses explicit UTF-8; the first failed pre-start
queue observation is archived, with no test result overwritten.

```
validation/wheel_v60_env/Scripts/python.exe -I -m ncd prove --config validation/original_proof_registry_protocol_v2r1.json --resume
validation/wheel_v60_env/Scripts/python.exe -I -m ncd verify-proof runs/original_proof_registry_v2r1
validation/wheel_v60_env/Scripts/python.exe -I -m ncd audit-requirements runs/original_proof_registry_v2r1 --require-closed
```

Installed candidate evidence is `validation/wheel_v60_run/status.json`.
The full-regression queue is `validation/full_regression_v60_launch/launch.json`.
Neither a queued test suite nor this limited candidate validation constitutes
full release acceptance or original-project completion.
