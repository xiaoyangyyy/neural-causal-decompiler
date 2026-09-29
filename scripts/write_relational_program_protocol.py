from pathlib import Path
Path('docs/RELATIONAL_PROGRAM_PROTOCOL.md').write_text('''# Relational program protocol

This protocol was frozen before generating evaluation worlds for seeds 693 and 694.
It addresses a specific representational mismatch found in the 493/494 audit: the
teacher attends across incident edge tokens while the extracted rule receives one
edge token at a time.

## Frozen inputs and candidates

- Teachers and extraction datasets: both modes from `runs/relational_seed493`.
- Node sizes: 3, 5, 8. Extraction labels are frozen teacher argmax labels.
- Program budget: six splits, beam width three, MDL penalty 0.001.
- `local_composed`: the existing 20 local graph primitives with arithmetic search.
- `local_primitive`: the same local primitives without arithmetic compositions.
- `relational_primitive`: local primitives plus means over four explicit incidence
  sets: same source, same target, successor, and predecessor. It uses no arithmetic
  compositions, avoiding a larger combinatorial budget than the baseline.
- All variants use uniform extraction row weights. The failed weighting variants
  from seed 593 are excluded before evaluation.

## Evaluation fixed in advance

For each seed 693 and 694, generate 64 unseen worlds for every combination of 3/5/8
nodes and ID/function/noise/scale/intervention environments. Report teacher-program
all-pair, active-pair, per-class and exact-graph fidelity, plus teacher and program
truth accuracy. The primary comparison is paired exact-graph fidelity of
`relational_primitive` versus `local_composed`, pooled only within each seed and
reported with a conservative Hoeffding interval over worlds. Per-size and per-family
results are descriptive. No variant is selected and refit from these worlds.

This experiment tests behavioral decompilation of fixed teachers. It cannot establish
that a relation primitive is encoded at a specific neural site or improve the frozen
teacher's causal accuracy.
''',encoding='utf-8')
