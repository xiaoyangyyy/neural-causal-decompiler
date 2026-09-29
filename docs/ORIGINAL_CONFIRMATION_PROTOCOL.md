# Frozen original-project confirmation protocol

Development seed: 8100. Confirmation seeds: 8101 and 8102. Preserve all
historical networks, negative results and acceptance thresholds.

Each confirmation seed covers 3/5/8 nodes and the five existing environments
(ID, function, noise, scale, intervention), ten worlds per cell: 150 per seed,
300 in total. Train paired observational-padded and true-intervention graph
teachers using the existing architecture and existing training configuration:
96 samples, 256 training and 64 development worlds per node size, 40 epochs,
width 48. Freeze after existing development selection. Test results never repair
parents, select equations or update teachers.

Per world use 96 discovery samples, 512 mechanism observations, 120 mechanism
epochs, 768 distillation queries and 512 evaluation rows. Symbolic fitting uses
at most eight terms and beam width 32. Observational, active and oracle-graph
branches remain separate; oracle is diagnostic only. The existing intervention
interface uses simulator do-samples with shared exogenous draws (paired controls).
The teacher receives response features, not equations or true parent sets.
This information regime is stronger than independent observational samples and
requires paired/resettable interventions for a physical interpretation.

Retain the historical numerical gates: both seeds must improve exact graph
accuracy and SHD, pooled exact graph gain >=20 percentage points, pooled truth
NMSE and intervention MAE reductions >=20%, symbolic-to-neural NMSE reduction
>=15% with improvement in each seed, no more symbolic atoms, truth error <=1.75x
oracle and intervention error <=2x oracle. Scientific acceptance additionally
requires all 300 declared worlds and complete independent replay. Failure does
not become a universal impossibility claim.

World-level exact graph indicators receive separate 99% familywise Hoeffding
bounds; node edges and repeated source pairs are not independent units. Report
unbounded NMSE/MAE as measured metrics without unjustified Hoeffding guarantees.

The coordinator runs one heavy worker at a time, uses two Torch threads, watches
an 8 GiB memory budget and an 8 GiB new-artifact budget, and stops an incomplete
worker at the 12-hour stage deadline. The memory guard samples working set;
it is not an OS-wide memory reservation. Every completed world has a manifest,
frozen teacher identities and a world/protocol binding. Failed or interrupted
attempts remain on disk; resume creates a fresh attempt instead of overwriting
partial artifacts. Completed worlds are validated and reused.

Commands:

```powershell
python -m scripts.confirm_original --protocol validation/original_confirmation_protocol.json
python -m scripts.confirm_original --protocol validation/original_confirmation_protocol.json --resume
python -m scripts.confirm_original --protocol validation/original_confirmation_protocol.json --verify
```

The provided background supervisor uses one shared 12-hour stage deadline:
generation gets at most six hours, then installed-package independent replay
gets the remaining time. It runs from a directory outside the source import
path, so replay workers also use the installed package. CLI commands can also
be given a shorter budget with `--seconds`. A partial summary enumerates every
still-unresolved declared unit and keeps
`science_passed=false`. Even a complete empirical pass does not prove true
noise independence, full-domain internal mechanism equivalence, or general
causal identification. These remain separately tracked proof obligations.

Background status and phase logs: `validation/original_confirmation_launch/`.
The supervisor is `validation/run_original_confirmation_stage.py`; it preserves
partial results and never reports overall original-project completion.
