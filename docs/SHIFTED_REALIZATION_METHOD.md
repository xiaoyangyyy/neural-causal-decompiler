# Nine-center infinite-horizon realization method

The frozen scalar ReLU system has effective dynamics
`x_next=0.5*x+0.4*a` and identity observation on the full unit state and
action intervals. The two-dimensional version is its coordinate-wise
product. Before using the product proof, the verifier checks the complete
serialized network against the frozen benchmark. Exact affine extraction
also proves that the neural transition maps the unit cube into itself.

The abstract centers per coordinate are

    0.1, 0.2, ..., 0.9.

There are nine states in one dimension and 81 product states in two
dimensions. The initial encoder uses nearest-center cells, with endpoints
0 and 1 included. The largest initial state error is 0.1.

The internal simulation relation is `|x-r_q|<=0.1005`, clipped to the
unit domain. The output tolerance is 0.101. The verifier propagates every
relation interval through the actual observation ReLU network and obtains
a maximum outward-rounded output bound of 0.10050000000000117.

For each abstract center, continuous actions are partitioned at nominal
nearest-center threshold crossings. The verifier propagates the full
concrete relation interval and each closed action segment through the
actual transition ReLU network. The maximum outward-rounded next-state
error from the chosen target center is 0.10025000000000153, below the
relation radius 0.1005. Adjacent action segments both include their
shared threshold and are both certified; execution chooses the right
segment on a threshold.

Initial coverage plus forward invariance proves the output error bound
by induction for every continuous action sequence and every time.
The product argument uses coordinate separability, checked against the
complete network parameters. The stored certificate also includes the
full initial-output packing lower proof.

The result improves the earlier 10/100-state construction to 9/81.
Together with the independent exact-rational multi-switch lower proof,
the one-dimensional minimum lies in [7,9]. The two-dimensional interval
remains [25,81]. The exact minimum in either dimension is unresolved.

