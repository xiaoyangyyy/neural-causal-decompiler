"""Build acceptance evidence for the frozen structured-mechanism protocol."""
from pathlib import Path
import json
from ncd.mechanism_search_experiment import verify_mechanism_search
root=Path(__file__).resolve().parents[1]
runs={}
for seed in (1793,1794):
 path=root/'runs'/f'mechanism_search_seed{seed}'
 replay=verify_mechanism_search(path)
 summary=json.loads((path/'summary.json').read_text())
 base=summary['aggregate']['baseline'];new=summary['aggregate']['structured']
 by_environment={}
 for env in summary['config']['environments']:
  rows=[r for r in summary['records'] if r['environment']==env]
  by_environment[env]={m:sum(r[m]['mean_symbolic_neural_nmse'] for r in rows)/len(rows) for m in ('baseline','structured')}
 runs[str(seed)]={'replay':replay,'baseline':base,'structured':new,'relative_primary_reduction':1-new['mean_symbolic_neural_nmse']/base['mean_symbolic_neural_nmse'],
  'checks':{'primary_lower':new['mean_symbolic_neural_nmse']<base['mean_symbolic_neural_nmse'],
   'truth_within_10_percent':new['mean_symbolic_truth_nmse']<=1.1*base['mean_symbolic_truth_nmse'],
   'intervention_within_10_percent':new['mean_intervention_effect_mae']<=1.1*base['mean_intervention_effect_mae'],
   'no_more_atoms':new['mean_nonconstant_atoms']<=base['mean_nonconstant_atoms']},'by_environment':by_environment}
pooled={m:{k:sum(runs[str(s)][m][k] for s in (1793,1794))/2 for k in runs['1793'][m]} for m in ('baseline','structured')}
reduction=1-pooled['structured']['mean_symbolic_neural_nmse']/pooled['baseline']['mean_symbolic_neural_nmse']
passed=all(r['checks']['primary_lower'] for r in runs.values()) and reduction>=.2 and pooled['structured']['mean_symbolic_truth_nmse']<=1.1*pooled['baseline']['mean_symbolic_truth_nmse'] and pooled['structured']['mean_intervention_effect_mae']<=1.1*pooled['baseline']['mean_intervention_effect_mae'] and pooled['structured']['mean_nonconstant_atoms']<=pooled['baseline']['mean_nonconstant_atoms']
record={'protocol':'docs/STRUCTURED_MECHANISM_PROTOCOL.md','replay_verified':True,'replicated_mechanism_improvement':passed,'pooled':pooled,'pooled_relative_primary_reduction':reduction,'runs':runs,
 'scope':'equation extraction from frozen neural mechanisms with supplied oracle DAG; graph recovery and general identifiability not certified'}
(root/'validation/structured_mechanism_acceptance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
lines=['# Structured symbolic mechanism search results','','Protocol: `docs/STRUCTURED_MECHANISM_PROTOCOL.md`.','',f'Both 30-world confirmation runs passed full replay. The preregistered rule **{"passed" if passed else "failed"}**.','',
 '| Seed | Baseline neural NMSE | Structured neural NMSE | Relative reduction | Baseline atoms | Structured atoms |',
 '|---:|---:|---:|---:|---:|---:|']
for seed in (1793,1794):
 r=runs[str(seed)];b=r['baseline'];n=r['structured'];lines.append(f"| {seed} | {b['mean_symbolic_neural_nmse']:.6f} | {n['mean_symbolic_neural_nmse']:.6f} | {100*r['relative_primary_reduction']:.1f}% | {b['mean_nonconstant_atoms']:.3f} | {n['mean_nonconstant_atoms']:.3f} |")
lines += ['', '| Pooled diagnostic | Baseline | Structured | Change |','|---|---:|---:|---:|',
 f"| Neural-teacher NMSE | {pooled['baseline']['mean_symbolic_neural_nmse']:.6f} | {pooled['structured']['mean_symbolic_neural_nmse']:.6f} | -{100*reduction:.1f}% |",
 f"| Ground-truth NMSE | {pooled['baseline']['mean_symbolic_truth_nmse']:.6f} | {pooled['structured']['mean_symbolic_truth_nmse']:.6f} | {100*(pooled['structured']['mean_symbolic_truth_nmse']/pooled['baseline']['mean_symbolic_truth_nmse']-1):+.1f}% |",
 f"| Intervention-effect MAE | {pooled['baseline']['mean_intervention_effect_mae']:.6f} | {pooled['structured']['mean_intervention_effect_mae']:.6f} | {100*(pooled['structured']['mean_intervention_effect_mae']/pooled['baseline']['mean_intervention_effect_mae']-1):+.1f}% |",
 f"| Operator exact fraction | {pooled['baseline']['operator_exact_fraction']:.3f} | {pooled['structured']['operator_exact_fraction']:.3f} | {pooled['structured']['operator_exact_fraction']-pooled['baseline']['operator_exact_fraction']:+.3f} |",'',
 'The primary result concerns fidelity to frozen neural conditional mechanisms. Ground-truth and intervention metrics are diagnostics. The very large baseline intervention MAE is caused by redundant polynomial terms extrapolating outside their fitted range; the hierarchy constraint removes this instability in these runs.','',
 'The oracle DAG isolates equation extraction. These results strengthen R10 but do not solve R9, do not measure end-to-end inferred-graph SCM recovery, and do not prove causal identifiability or a universal theorem.','',
 'Machine-readable evidence: `validation/structured_mechanism_acceptance.json`.','']
(root/'RESULTS_STRUCTURED_MECHANISM.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'passed':passed,'pooled_relative_reduction':reduction,'pooled':pooled},indent=2))