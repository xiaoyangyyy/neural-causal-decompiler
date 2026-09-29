# Exact all-horizon output-trace packing capacity for affine rings

This result applies to the four frozen affine ReLU ring networks of
dimension 8, 32, 64 and 128 at `epsilon=binary64(0.17)`. Their serialized
weights are checked by exact fractions. A full-cube ReLU phase proof
extracts one affine transition `x'=Ax+Bu+b`; a full-cube interval
check proves invariance for every continuous unit-cube action. The
observation is exactly `(x_0,x_1,x_2,x_3)`. Therefore, for two runs
under the **same** action sequence, all action and offset terms cancel
from their state difference.

Let `theta=2*epsilon`. A trace packing is a set of initial states
whose output traces are pairwise farther than `theta` in at least one
coordinate and time. Such a set lower-bounds the state count of every
deterministic realization with observation error at most `epsilon`.

The matching 162-point construction chooses six feedback pairs
`(x_0,x_(d-1))`:

    (0,0), (1/10,1), (9/20,0),
    (11/20,1), (9/10,0), (1,1).

Each of `x_1,x_2,x_3` independently takes `0,1/2,1`; all other
coordinates are zero. Different middle triples separate immediately.
The three close `x_0` pairs have first-step output-0 separation
`binary64(0.45)/10 + binary64(0.3)`, approximately `0.345`,
strictly above `theta`. The exact ReLU verifier checks all
`162*161/2=13,041` pairs: 12,960 first separate at time 0 and
81 at time 1.

No one-step packing can contain 163 points. Divide each of the four
observed initial coordinates into three intervals of width at most
`1/3 < theta`, giving 81 cells. Inside a cell, initial outputs are
close. The next outputs 1, 2 and 3 differ by at most
`(|A[i,i-1]|+|A[i,i]|)/3`, approximately `1/6 < theta`.
The next output 0 lies in an interval of diameter at most
`|A[0,0]|/3+|A[0,d-1]|`, approximately `0.45 < 2*theta`.
Three scalar values pairwise farther than `theta` would require
diameter greater than `2*theta`, so each cell holds at most two
packing points. The upper is `81*2=162`.

The same upper holds at **every horizon**. Write `M_t` for the maximum
difference in observed coordinates 0..3 and `H_t` for the maximum
difference in hidden coordinates 4..d-1. Exact row support and absolute
weights give

    M_(t+1) <= max(a*M_t + b*H_t, g*M_t),
    H_(t+1) <= h*max(M_t,H_t),

where `a=|A[0,0]|`, `b=|A[0,d-1]|`, `g` is the maximum
absolute row sum of observed rows 1..3, and `h` the maximum hidden
row sum. The checker derives all four quantities from the serialized
network; here they are approximately `0.45,0.3,0.5,0.5`.

For unit-cube initial states, `H_0<=1`. If a pair is unseparated at
times 0 and 1, then `M_0,M_1<=theta` and `H_1<=h`.
The exact checker verifies `theta<=1`, `h<=1`,
`a*theta+b*h<=theta`, `g*theta<=theta`, and
`h*max(theta,h)<=h`. Numerically, the three forward bounds are
about `0.303,0.170,0.250`, versus `theta?0.340` and `h?0.500`.
Thus the rectangle `M<=theta,H<=h` is forward invariant. Any pair
not separated by time 1 remains unseparated forever. Because actions
cancel, this holds for every common continuous action word.

Consequently the **maximum all-horizon common-action trace packing
cardinality is exactly 162**, achieved already at one step. This is
a ceiling on pairwise trace-packing lower bounds, not a proof that a
162-state deterministic machine exists. Transition consistency can
require more states even when all pairwise traces are close; the
general minimum-state interval is still `[162,27216]`.
The exact behavioral quotient remains 128-dimensional because it
concerns equality at arbitrarily fine precision, not this fixed
positive tolerance.
