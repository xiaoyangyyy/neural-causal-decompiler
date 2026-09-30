# Frozen Tanh mechanism: verified origin-point fidelity failure

For the archived three-node `test_id` development world, the saved child-1
Tanh checkpoint and the archived symbolic child-1 equation are compared
at the exact parent input (x0,x2)=(0,0). The third full-data coordinate
is also set to 0; it is not a parent of this mechanism.

The existing rational interval backend exports the stored float32
weights as exact rationals, evaluates real tanh with checked rational
enclosures, and verifies a single-leaf proof tree for the singleton
domain `[[0,0],[0,0],[0,0]]`. The normalized
(neural minus symbolic) error lies strictly within the certified interval

    [3381667802333073452781/151115727451828646838272,
     13526671209332293811253/604462909807314587353088].

Both endpoints are approximately 0.02237800035, and the lower
endpoint is strictly greater than 1/100. The normalizer is the
frozen checkpoint's saved output training scale, not a scale chosen
after seeing the witness.

The generator `validation/fixed_tanh_origin_fidelity_v1.py` stores the
network export, candidate AST and one-leaf proof in
`validation/fixed_tanh_origin_fidelity_v1.json`. The separate
`validation/check_fixed_tanh_origin_fidelity_v1.py` reopens the
original archived checkpoint and AST, checks the pinned archive and
proof-backend hashes, and recomputes the interval proof without
importing the generator. Its receipt is
`validation/fixed_tanh_origin_fidelity_verification_v1.json`.
Six targeted tests include numerical, conclusion and scope tampering
and a rehashed archive mutation.

This is a direct frozen-network fidelity failure of **one fixed
candidate** at one exact input. It uses no true SCM equations or noise
oracle. The certificate does not establish that the original
declared intervention/input domain contains the origin, so it does
not close the universal R10.mechanism claim. It is a real-arithmetic
certificate for exact stored parameters; device float32 inference
rounding is not certified. Any declared domain containing this point
cannot give this candidate a uniform 1/100-scale fidelity guarantee.
