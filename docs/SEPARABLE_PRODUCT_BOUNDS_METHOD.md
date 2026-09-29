# Exact composition of finite-state bounds for separable ReLU products

This result uses the exact scalar minimum `K_1=9` certified by the
[interval-chain proof](EXACT_INTERVAL_CHAIN_METHOD.md). Let the serialized
`d`-dimensional ReLU network represent, over the full unit state/action
cube,

    x'_j = lambda*x_j + beta*a_j,       O(x)=x,
    lambda = 1/2, beta = exact binary64 value of 0.4.

The observation error is measured in the maximum coordinate norm at the
exact binary64 tolerance `epsilon=0.101`. The claim is about arbitrary
initial states, arbitrary continuous action sequences, and every finite
time horizon. Floats in the serialized network are interpreted as exact real
coefficients for the theorem.

The checker first extracts the exact affine function represented by each
serialized ReLU network. For every hidden unit it bounds the preactivation
over the full unit cube and accepts only a fixed active or inactive phase;
a phase-changing unit is unresolved. It then checks that the extracted
transition has precisely `lambda` on its matching state coordinate and
`beta` on its matching action coordinate, zero offsets and zero
cross-coordinate terms, and that the extracted observation is the identity.
This is a *functional* check: hidden neurons may be permuted or otherwise
reparameterized without changing the conclusion, provided the checker can
prove the phases globally. All serialized weights and biases participate
in the extraction. The frozen template has `8*d^2+5*d` such coefficients.
A coupled transition, mixed observation, or changed realized function fails
closed. The exact serialized template has zero transition offset because
`1/2 + binary64(0.4) - binary64(0.9) = 0`.

For the upper bound, take the Cartesian product of the exact rational
nine-state scalar realization. Each coordinate has an inductive interval
relation and error at most `epsilon`; the product relation has maximum
coordinate error at most `epsilon`. The abstract state space has `9^d`
states, represented implicitly without materializing them.

For the lower bound, the scalar map has controlled fixed points

    c_k = k*beta / (3*(1-lambda)),  k=0,1,2,3,

maintained by actions `k/3`. Their pairwise gaps exceed `2*epsilon`.
Fix each of coordinates 2 through `d` to one of these four values and
hold its action at the matching fixed action. This creates
`4^(d-1)` invariant slices. On each slice, the free first coordinate
and its action form the complete scalar problem, so at least nine
abstract states are needed. Different slices cannot share an abstract
state: a differing fixed output coordinate would then have to be
within `epsilon` of two values more than `2*epsilon` apart. Hence

    9 * 4^(d-1) <= K_d <= 9^d.

`ncd.separable_product_bounds` checks the realized affine function, replays
the 40,545-record scalar rational exclusion, checks the exact scalar
upper, and regenerates the dimension certificate. The acceptance freezes
and replays 2, 8, 32, and 128 dimensional networks from JSON. A second
eight-dimensional network with the transition and observation hidden layers
permuted and positively rescaled is certified with identical bounds. This
demonstrates invariance under that representation change. A coupled 2D variant and a separately trained 128D network
are rejected. The exponential numbers describe mathematical state-space
complexity; the certificate and verifier operate on polynomial-size
network weights and symbolic integer bounds.

The theorem exploits exact coordinate separability of the realized function.
The current extraction requires globally fixed hidden ReLU phases;
a function that is affine only through phase cancellations may remain
unresolved. A functionally unused phase-changing hidden unit is an explicit
negative control for that boundary. The theorem does not give these
bounds for trained coupled networks, identify a unique causal coordinate
system, or close the gap between its product lower and upper bounds when
`d>1`.
