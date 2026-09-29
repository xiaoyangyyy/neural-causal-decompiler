# Exact all-horizon behavioral covering number for affine rings

This theorem covers the four frozen globally affine ReLU ring networks of
dimension 8, 32, 64 and 128, with full unit-cube initial states and
continuous unit-cube actions at `epsilon=binary64(0.17)`. A
**behavioral cover** is a fixed set of concrete initial states such that
every concrete initial state has one representative whose true output
trajectory stays within `epsilon` under every common action word and
at every time. The representative is chosen from the initial state;
it does not depend on the future actions.

The construction uses three centers `1/6,1/2,5/6` on each directly
observed coordinate `x_0,...,x_3`, two centers `1/4,3/4` on the
feedback coordinate `x_(d-1)`, and center `1/2` on every other
coordinate. There are `3^4*2=162` concrete representatives, all inside
the full state cube. Nearest-center selection bounds initial observed
error by `1/6`, feedback error by `1/4`, and all other hidden
coordinate errors by `1/2`.

The exact checker extracts the affine transition `x'=Ax+Bu+b` after
proving every ReLU phase fixed on the full input cube. Differences of
runs under the same action word obey `delta x'=A delta x`:
intervention and offset terms cancel. Exact invariance checks ensure
every trajectory stays inside the cube.

Let `a=|A[0,0]|`, `b=|A[0,d-1]|`, `g` be the largest absolute row
sum among observed rows 1..3, and `h` the largest among hidden rows.
The exact support check proves row 0 depends only on `x_0` and
`x_(d-1)`; rows 1..3 depend only on observed coordinates. The
first-step error bounds are

    output 0: a/6 + b/4  ~= 0.150,
    outputs 1..3: g/6    ~= 0.083,
    hidden coordinates: h/2 ~= 0.250.

Initial observed error is `1/6 < epsilon`. From time 1 onward,
the rectangle `M<=epsilon, H<=h/2` is forward invariant, where
`M` and `H` are maximum observed and hidden coordinate errors.
The checker verifies, with exact binary64 fractions,

    a*epsilon + b*(h/2) <= epsilon,  ~= 0.1515 < 0.17
    g*epsilon           <= epsilon, ~= 0.0850 < 0.17
    h*max(epsilon,h/2)   <= h/2,     ~= 0.1250 < 0.25.

Thus every initial state is covered by one of the 162 *true concrete
trajectories* for every continuous action sequence and all times.
No discretization of the action cube is needed.

The preceding exact all-horizon 162-point packing certificate gives
the matching lower bound: two points whose outputs separate by more
than `2*epsilon` under the same word cannot share one behavioral
representative. The verifier replays all 13,041 packing pairs and the
analytic all-horizon capacity theorem, so the minimum behavioral cover
number is **exactly 162**.

This is a passive, action-uniform trajectory cover, not an executable
162-state deterministic realization. After an action, a representative
may move outside the fixed representative set. Rounding it to a new
representative could accumulate errors, and the current proof does
not establish a transition map that preserves an `epsilon` simulation
relation. The finite-machine minimum remains in `[162,27216]`.
The exact 128-dimensional behavioral quotient is also a different
notion: it distinguishes arbitrarily small, long-delayed effects.
