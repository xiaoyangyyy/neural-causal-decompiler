# Exact multi-switch lower certificate: scalar K is at least seven

The 0.34 lower certificate ruled out one through five states. Version 0.35
counts every target switch forced by a continuous action sweep. For the
frozen scalar ReLU system, the strongest checked result is now

    7 <= K_N(0.101, all continuous action words) <= 10.

The upper side is the already verified executable ten-state model.
The lower side does not assume its encoder, representative grid, outputs,
or transition choices.

For any hypothetical m-state deterministic realization, the concrete state
sets paired with its abstract states have interval hulls. Their total
available overlap is at most `m*2*epsilon-1`. A source hull of width at
least `1/m` moves under a continuous action. One target hull can contain
that moving interval for only a bounded action range. At m=6, covering the
entire action interval requires at least four distinct targets and three
overlaps of at least `(0.5/6)` each. The forced total overlap is `0.25`,
but the available overlap is about `0.212`. Six states are impossible.

The serialized ReLU network is checked to be affine over the full declared
domain, and every target-count and overlap inequality is checked using
exact rational arithmetic on its stored coefficients. The formal run is
`runs/certified_multiswitch_lower_seed13701`; the joint acceptance file is
`validation/certified_multiswitch_lower_acceptance.json`. The complete
derivation is in `docs/MULTISWITCH_LOWER_BOUND_METHOD.md`.

This improves a transition-aware lower bound beyond ordinary pairwise
packing. It does not show whether seven, eight, or nine states suffice.
The theorem currently applies to scalar affine systems with identity
observation and cannot yet certify minimality for the coupled nonlinear
benchmark or trained high-dimensional networks.


Version 0.35.0 release verification: 159 regression tests passed.
The isolated wheel matched all source modules and completed all 24 certified
generation/replay CLI commands. Wheel SHA-256:
533f19e99f879a197d61080ecae0579c1ce1b3a56e1341f684fb9c2e3e7cfaaa.
Evidence: validation/wheel_v35_run/status.json.