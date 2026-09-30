# Historical SCM second moments: all 300 observational worlds classified

The five Student5 worlds previously left unresolved by a sufficient bound now have independently replayed strict tail certificates. Combined with the accepted 295 finite-moment proofs, the historical 300-world **ideal observational** family contains **295 worlds with finite second moments and five with an infinite second moment at the listed node**.

| Historical world | Pivot root noise | Divergent output | Degree | Exact cutoff T | Leading magnitude lower, decimal display |
|---|---:|---:|---:|---:|---:|
| seed_8101_n5_test_noise_7 | 1 | 4 | 3 | 145 | 0.0616566015 |
| seed_8101_n8_test_noise_9 | 7 | 6 | 4 | 447 | 0.0122682786 |
| seed_8102_n3_test_noise_2 | 2 | 1 | 4 | 614 | 0.00770425944 |
| seed_8102_n3_test_noise_4 | 0 | 1 | 3 | 49 | 0.109232944 |
| seed_8102_n5_test_noise_3 | 3 | 0 | 4 | 620 | 0.00689783046 |

The displayed leading magnitudes are rounded for reading; the certificate and verifier use exact rational values. A positive-probability event bounds all other independent noises, and a positive rational density lower bound for the pivot Student5 variable converts each nonzero cubic or quartic term into an unbounded lower sequence for E[X_j^2]. The proof and its limits are detailed in docs/SCM_SECOND_MOMENT_TAIL_METHOD_V1.md.

The frozen protocol is validation/scm_second_moment_protocol_v1.json (SHA-256 ff9bfa5c322a9a158834280f4d2fa93419cabb426f0df3ee8e680974894b63cf). The portable proof is runs/scm_second_moment_completion_v1 (manifest SHA-256 df509b37135e5d7377367336c58a00f7cfa6a82a92080cf02a3e4770ca5849ea). The independent wheel SHA-256 is c88579885550fb8c0eaa22c8cce0363e4f279af888d63d74088aa149205e18b5.

Three source tests passed, including archive tampering and false model-premise rejection. Installed-wheel generation and a separate installed replay both passed in Windows Jobs with an 8 GiB descendant memory cap and a 15-minute stage limit. Their peak committed Job memory was 19,943,424 and 18,759,680 bytes. The completed bundle is 880,408 bytes. The strict acceptance receipt is validation/scm_second_moment_acceptance_v1.json.

The accepted finite-support result for six empirical SCMs in each of the five worlds yields a further observational corollary: every coupling has infinite expected squared coordinate cost in these 30 true/estimated pairings. This is an extended quadratic transport statement; finite-valued Wasserstein-2 is outside its ordinary domain here, and no analogous Wasserstein-1 failure follows.

The prior population proof's 295 finite claims apply under its declared compatible intervention bounds; the five new infinite claims are observational only. The acceptance receipt field worlds_with_any_intervention_claim=0 counts only the five new divergence witnesses, not the inherited 295 finite certificates. No claim is made about finite device PRNG output, other worlds, unbounded general SCM families, true-noise inference from samples, or full causal decompilation. The original R0-R13 ledger remains 0 proved, 2 refuted, 36 unresolved.