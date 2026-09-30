# Frozen independent-do linear graph confirmation

This protocol tests a graph candidate using **independently sampled** node-do
responses, without the paired exogenous oracle required by the earlier
linear and nonlinear exact graph theorems. It is restricted to the
linear-Gaussian members of the actual 3/5/8-node GraphWorld generator.
It does not decompile the frozen discovery network or identify nonlinear
SCMs. Original R9 remains open regardless of the finite result.

The rule was chosen **after** inspecting the previous seed-8301/8302
confirmation archive. A posthoc diagnostic on its 60 linear-Gaussian worlds
found 44/60 exact graphs at threshold 0.2, versus 11/60 for the frozen
observational graph teacher. The exploratory 0.05 and 0.1 thresholds had
lower exact counts. These numbers are rule selection evidence, not a new
confirmation. The diagnostic is bound in
`validation/independent_do_linear_graph_diagnostic_v1.json`.

Before examining new results, the protocol fixes seeds 8401/8402; three
node sizes; all five existing environments; and linear-Gaussian indices
0 and 5 from each 10-world cell. This is 60 new independent worlds.
Per world, the candidate gets 384 independent node-do rows, split across
`do(X_i=-1)` and `do(X_i=+1)`, from the prior 512-row mechanism-observation
budget. It computes each column of a total-effect matrix by the two
sample-mean differences divided by 2, inverts the matrix, thresholds each
off-diagonal direct-effect estimate at 0.2, and reports cycles as failures.
The frozen graph teacher sees 96 observational rows. This comparison
measures the value of **additional intervention access**; it does not
compare equally informed algorithms or establish network-to-program
fidelity. Truth graphs and equations enter only the separate evaluator.

The predeclared relative gate requires at least 30/60 candidate exact
graphs, at least 20 more exact worlds than the teacher, at least 10/20
candidate exact graphs at each node size, and a one-sided exact paired
sign-test p-value at most 0.01. The sign test treats each independent
world as one paired outcome and tests conditional fairness of discordance
directions; it does not imply 99% accuracy or a 99% all-instance theorem.
The frozen plan also fixes source/checkpoint hashes, distinct seeds, a
12-hour stage limit, one Torch thread and 8 GiB artifact/memory budgets.

The protocol is
`validation/independent_do_linear_confirmation_protocol_v1.json`.
`validation/confirm_independent_do_linear_graph_v1.py --preflight`
checks the freeze and historical seed separation. Its `--run` operation
writes candidate-only observations, truth-only metadata, world results,
and the exact copied protocol under
`runs/independent_do_linear_confirmation_v1`. The separate
`validation/verify_independent_do_linear_graph_v1.py --write` must
regenerate every world, sample group and teacher prediction, independently
recompute the candidate matrix/graph, and calculate the frozen gate.
A failed or missing unit remains unresolved; no truth-based graph repair
is allowed.