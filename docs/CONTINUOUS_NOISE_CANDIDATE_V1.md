# Continuous noise candidate v1

This development module proposes a continuous, full-support marginal noise law per
SCM node. It addresses the finite empirical-residual support obstruction but does
not prove that the recovered law is the true exogenous noise law.

Input to validation/continuous_noise_candidate_v1.py:

- Residual matrices from observations used only to fit family parameters and from
  separate observations used only to select a family. The caller must provide
  unique observation IDs across both splits; overlap is rejected.
- Gaussian, Laplace and Student-t with five degrees of freedom are the fixed
  candidate families. Parameters are fitted only on the fit split; the selection
  split maximizes held-out mean log density.
- Each selected law has positive scale and a sampler. The module records hashes
  of both residual matrices and observation-ID sequences.

A matching ID check does not prove that separately labeled rows are statistically
independent. Residuals of a fitted mechanism are not automatically samples from
the true exogenous noise. The current module therefore reports candidate-only,
with joint independence, true family, sampling independence and the original
claim all unproved.

For R10 distribution recovery, the next admissible experiment needs a frozen
mechanism and disjoint fit, selection and final evaluation worlds, with complete
provenance. It must independently test the joint residual law, account for
mechanism and graph error, and bound the distance between the recovered joint law
and the true exogenous law under declared interventions. Marginal held-out
likelihood alone is insufficient.

The synthetic tests check the three candidate families and reject reused or
malformed split IDs. They are development checks, not confirmation evidence.