# GitHub checkout and evidence scope

This repository contains source, tests, research documentation, validation
protocols and receipts, frozen source snapshots, and selected independent
proof bundles. Large historical experiment datasets, installed environments,
caches and built wheels remain local. No experiment files are deleted.

## Scientific status

The production package declares version `0.63.0.dev1`. Retained directories
`validation/source_candidate_v64r2` and `validation/source_candidate_v65` are
experimental candidates, not a production-version change. Their stage
receipts report 323 and 343 regression tests respectively, with independent
proof replay. Collection verification does not imply all claims are proved.

The latest inspected original ledger contains 38 atomic claims: 0 proved,
2 refuted and 36 unresolved. The original causal-decompilation objective is
incomplete. Scoped certificates retain their own declared assumptions.
`validation/scm_population_package_v1` remains unaccepted research work.

## Install

Use Python 3.11 or newer in a fresh environment:

```sh
python -m pip install -e ".[test]"
ncd --help
ncd prove --help
ncd verify-proof --help
ncd audit-requirements --help
```

Historical integration tests and full replays can require excluded datasets
or separately installed proof packages. Restore the declared artifacts and
package versions or rerun the corresponding experiment scripts. Preserve
original protocols when creating portable revisions; do not edit frozen,
hash-bound files in place. A checkout does not reproduce the original Windows
execution environment by itself.

## Preserve evidence bytes

`.gitattributes` disables newline conversion because proof manifests bind
SHA-256 hashes of source and artifact bytes. Validate external artifact hashes
after transfer. Historical absolute workspace and installed-environment paths
record execution context, not credentials or portable installation paths.

`.gitignore` retains selected independent proof bundles and frozen validation
sources/receipts. Acceptance receipts referring to excluded external files
retain their hashes; a receipt alone does not verify a missing artifact.
