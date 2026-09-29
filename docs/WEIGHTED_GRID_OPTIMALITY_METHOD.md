# Exact optimality inside the weighted uniform-grid proof class

This theorem concerns the **class of certificates checked by**
`verify_weighted` for the frozen trained affine 128D network at exact
rational tolerance `17/100`. It does not give a lower bound on arbitrary
finite-state realizations. The class permits any positive integer number
of equal-width bins on each of the 128 state coordinates, any positive
integer uniform action-bin count, any positive coordinate relation radii,
and any packing-axes setting, provided the weighted verifier certifies
the full unit-cube model.

Write `n_i` for state bin counts, `h_i=1/(2n_i)`, `r` for relation
radii, `S` for the exact absolute-weight state sensitivity of the
serialized transition network, and `C` for the exact absolute-weight
observation sensitivity. The verifier requires

    r >= S*r + h + action_error,        C*r <= epsilon.

All matrices and vectors on the right are nonnegative. Therefore, for
any integer `T>=0`,

    epsilon >= C*r >= C*(I+S+...+S^T)*h.

This is a necessary condition for **every** certificate in the class,
regardless of how its radii or action bins are chosen. The exact checker
derives `S` and `C` by multiplying absolute serialized layer weights as
fractions. It uses only ten transition powers (`T=10`), so no infinite
series or floating point convergence assumption enters the exclusion.

Let `g_i` be row `i` of `C*(I+...+S^10)`. For each directly observed
middle axis `i=1,2,3`, the exact inequality
`g_i[i]/(2*5) > 17/100` forces `n_i>=6`. For the first observation,
discarding nonnegative contributions from other axes gives

    17/100 >= g_0[0]/(2*n_0) + g_0[127]/(2*n_127).

AM-GM implies
`n_0*n_127 >= ceil(g_0[0]*g_0[127]/epsilon^2)=63`.
Thus if **any** of the remaining 123 coordinates has more than one
bin, the complete state count is at least
`6^3 * 63 * 2 = 27,216`.

If every remaining coordinate has one bin, its exact contribution to
row 0 is the fixed positive background
`b=sum(g_0[j]/2 for j not in {0,1,2,3,127})`.
To exclude `n_0*n_127<=125`, it suffices to check the 125 possible
values `n_0=1,...,125` with the largest allowed
`n_127=floor(125/n_0)`. The verifier evaluates every case using exact
fractions. The weakest case is `(n_0,n_127)=(15,8)`; even there the
necessary row-0 bound exceeds `17/100` by about `0.001033434827566737`.
Hence `n_0*n_127>=126`, and the total is again at least
`6^3*126=27,216`.

The 0.47 state/action grid with bins `(14,6,6,6,9)` on the five
active coordinates is an independently verified 27,216-state upper
certificate, so the lower and upper match **within this certificate
class**. The verifier binds the proof to both the frozen model digest
and that upper certificate, recomputes the ten-step rational rows, and
rejects a changed conclusion or any failed inequality. A shorter
eight-step truncation fails the extra-axis exclusion, serving as a
negative control.

This result identifies a method boundary. More action bins, larger
search ranges, or merely moving uniform coordinate bin counts cannot
beat 27,216 while using the same weighted certificate conditions.
A smaller general realization would require a different abstraction
geometry, a sharper simulation relation, or a stronger proof method.
