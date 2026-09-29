# Exact replay of integer-grid refinement on the frozen affine 128D network

The previous finite-realization certificate for the frozen trained affine
128D ring had 40,824 abstract states at tolerance `17/100`. Its implicit
machine quantizes coordinates `(0,1,2,3,127)` into `(18,6,7,6,9)` bins;
the remaining coordinates use one bin each. It quantizes each continuous
action coordinate into 128 bins. The original certificate already proved
all-unit-domain initial coverage, full unit-cube transition invariance,
inductive relation closure, and observation error for *every* continuous
action word and time.

The weighted verifier's sufficient conditions are coordinatewise. If
`S` is the absolute-weight state sensitivity and `U` the action
sensitivity, bin halfwidths are `h_i=1/(2*n_i)`, action-bin count is `m`,
and relation radii are `r_i`, the inductive step requires

    S*r + U*1/(2*m) + h <= r.

The observation sensitivity applied to `r` must not exceed `epsilon`.
For a contractive numerical influence matrix, the least suggested radius
solves `(I-S)*r=h+U*1/(2*m)`. This floating calculation is used only
to propose bins and rounded radii; it is **not** a certificate.

`ncd.integer_grid_refinement` reads a previously certified baseline and
selects its nontrivial coordinates automatically. It enumerates a
bounded integer neighborhood of the four nonpivot active axes, computes
the smallest numerical pivot count satisfying the observation budget,
and ranks candidates by state count. A fixed-action search examines
4,851 tuples at the original 128 action bins. A joint search examines
9,702 tuples with action-bin multipliers 1 and 4. Neither count claims
that the chosen grid is globally minimal, even within coordinate grids.

Two proposals survived independent exact-rational verification:

| Case | State bins at `(0,1,2,3,127)` | Action bins per coordinate | Abstract states |
|---|---|---:|---:|
| Historical baseline | `(18,6,7,6,9)` | 128 | 40,824 |
| State-only refinement | `(16,6,6,6,8)` | 128 | 27,648 |
| State/action refinement | `(14,6,6,6,9)` | 512 | 27,216 |

The exact `verify_weighted` routine reconstructs every bound from the
serialized ReLU weights using fractions. It also checks complete unit-cube
invariance and the implicit machine's initial/output/transition
inequalities. The action partitions are represented algorithmically; the
machine does not enumerate `512^5` action cells. The refined upper is
valid at the exact rational tolerance `17/100`, slightly stricter than
the binary64 tolerance of the independently replayed 135-state lower
packing. Thus the same frozen model has certified interval
`[135,27216]` under the latter tolerance.

The search does not prove a global optimum or eliminate the large gap
between lower and upper. In particular, a non-coordinate abstract
state space could be smaller. The refinement also says nothing about
the separately trained nonlinear 128D models without their own exact
upper certificates.
