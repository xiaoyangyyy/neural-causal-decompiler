# Fixed-world Gaussian noise transport, version 1

The historical fixed test_id world in
validation/continuous_noise_posthoc_package_v1 has three independent true
Gaussian exogenous coordinates with stored binary64 scale 0.35, unit
observed-coordinate scales and no root shift. The archived candidate selected
a Gaussian location and scale for each node. Its ideal sampler draws each
node's coordinate separately. This certificate compares the declared
candidate product law with the oracle true exogenous law for this one world.

Use common independent standard Gaussians Z_j. For true noise
U_j = sigma Z_j and candidate noise V_j = mu_j + tau_j Z_j,
the coupling is valid for both declared joint laws. Since E|Z_j| <= 1,

    E|U_j - V_j| <= |mu_j| + |sigma - tau_j|.

The joint Wasserstein-1 distance under l1 cost is no greater than the sum
of these three rational bounds. Every location and scale is interpreted as
the exact rational value of the stored binary64 number. The archive hashes,
candidate source hashes, Gaussian family choice and original bundle
manifest identity are checked before calculation.

The resulting bound is exactly

    35430223889551975 / 288230376151711744
    = 0.12292328228064023...

The certificate is
validation/fixed_gaussian_noise_transport_v1.json. Replaying
validation/fixed_gaussian_noise_transport_v1.py without arguments recomputes
it from the frozen bundle and rejects any changed number or scope flag.
Eleven targeted tests pass, including a candidate payload whose internal
manifest was rehashed after mutation. The first preseal attempt lacked the
fixed original-manifest anchor; its certificate remains at
validation/fixed_gaussian_noise_transport_v1_preseal_attempt0000.json and
is rejected by the corrected verifier.

This is a bound between two *declared ideal noise laws*. The oracle true
law comes from benchmark metadata; it was not inferred from observations.
The archived residual data do not establish independence of the unknown
true residual process. The candidate mechanism's uniform error on the
unbounded Gaussian support is not proved, so this noise result cannot be
propagated to a certified intervention-outcome distribution bound for the
historical SCM. It is one fixed development world, not the independent
confirmation protocol. R10.noise and R10.intervention_distribution remain
unresolved in the original ledger.

A second checker,
validation/check_fixed_gaussian_noise_transport_v1.py, does not import the
certificate generator. It independently reads the archived world and
candidate result, checks the original manifest and primary source SHA-256,
reconstructs each stored binary64 rational, and recomputes the joint bound.
Its sealed replay receipt is
validation/fixed_gaussian_noise_transport_verification_v1.json. This
additional replay confirms the exact noise-only result and keeps the
same-world, oracle-metadata, and missing-mechanism-premise flags visible.
