# Certified nonlinear shared-action closure and standalone programs

The previous initial-grid certificate proves output accuracy and the
one-step handoff into a weighted recurrent simulation relation. It keeps
216 cells in a conservative full-cube transition-image rectangle. This
method computes a smaller closed graph from the actual frozen network and
compiles it into a portable finite-state program.

## Exact geometry from actual network weights

The supported transition has one hidden ReLU layer and two continuous
actions. At any finite-state midpoint, each relevant hidden preactivation
is affine in those actions. The compiler extracts sparse nonzero weights
directly; it does not read teacher coefficients, training targets or a
known ground-truth mechanism. Units can be omitted only when their output
coefficients on all active coordinates are zero. Inactive grid axes have
one cell and their midpoint is exactly one half.

Split the unit action square by every relevant phase-crossing
preactivation. Every split uses exact rational half-plane clipping; the
two closed halves cover their parent polygon. A midpoint determines the
phase, and vertex extrema independently check its sign on the whole
convex polygon. The resulting output affine forms are the corresponding
sums of actual hidden-layer forms and output weights. State phase changes
are handled separately at every abstract source center.

All eight frozen models have four interior action boundaries and five
polygons at every checked source. This fact is established from their
weights, rather than assumed by the algorithm.

## Joint successors and the complete least fixed point

For every active output, exact polygon vertex extrema give coordinate
bounds. Floor-and-clamp quantization yields a finite candidate range;
then each candidate target cell is checked by intersecting each action
polygon with all of its output-cell inequalities simultaneously. Closed
intersections that are points or segments are retained. This preserves
shared-action correlations across all five active coordinates, including
boundary contacts.

Enumerate all 81/108 initial centers, collect their successors, and run a
breadth-first least fixed point over recurrent centers. Every successor
must remain in the prior certified image rectangle. The checker replays
the whole ordered initial graph and recurrent graph, rather than checking
only traces or the reported count. The old initial handoff and recurrent
simulation inequalities then prove the smaller machine's error bound
for the full initial/action cubes and every finite horizon.

The graph remains conservative: it uses the whole action cube rather
than only the 128-by-128 quantized action midpoints, and closed target
cells can introduce extra edges at quantization ties. Least closure is
minimal for this computed relation, not for all possible finite machines
or the exact concrete reachable set.

## A source-network-free executable artifact

On each action region `P_r`, the active transition has the exact form

\[
 F_i(c(q),u)=d_i(q)+a_{r,i}u_0+b_{r,i}u_1+e_{r,i}.
\]

The exporter factors the source-dependent offsets `d_i(q)` from the
region affine forms and deduplicates the normalized region templates.
For each of the eight models one template suffices for every initial and
recurrent source. A program stores that template, the discrete state
rows, offsets, output values and successor sets. It stores no source
network weight matrices and requires no source model file at execution.

`program_initial`, `program_output` and `program_step` accept only this
program plus points/states/actions. A step quantizes actions, selects a
closed phase polygon, evaluates its five rational output forms plus the
source offsets, quantizes the target and checks graph membership. On
shared polygon boundaries adjacent affine forms agree because the
corresponding ReLU preactivation is zero.

A standalone execution example from the project root is:

```python
from ncd.io import read_json
from ncd.nonlinear_action_closure import program_initial, program_output, program_step

program = read_json("runs/nonlinear_action_closure_v1/seed_6101/d_128/program.json")
state = program_initial(program, [1] * program["state_dim"])
state = program_step(program, state, [0, 1])
print(program_output(program, state))
```

Program verification reconstructs every row/template from the verified
network and closure. Hence its transitions equal those of the original
finite realization for every quantized action, while the inherited
simulation proof relates it to the continuous neural dynamics. Tests
also disable network evaluation/compilation functions during execution.
This supplies the executable artifact together with the state-bound
certificate. It does not prove globally shortest source
code, globally minimum state count or a unique causal interpretation.

## Evidence and replay

```powershell
python -m scripts.acceptance_nonlinear_action_closure --verify
python -m pytest tests/test_nonlinear_action_closure.py
```

The eight complete certificates and portable programs are stored under
`runs/nonlinear_action_closure_v1`. Acceptance binds source models,
weighted and initial certificates, graph certificates and portable program
file hashes. Exact state/control nonlinearity witnesses, the independent
81-state packing lower, all graph/initial proofs and program templates are
replayed. Three runtime traces with mixed actions check each program
against original neural observations.

These networks still have a structured fixed feature dictionary and
learned output coefficients on synthetic traffic-inspired data. Tolerance
is exactly `17/100`. General deep networks, arbitrary intervention
families, global minimality and original R4/R5/R8/R9/R10 remain open.
