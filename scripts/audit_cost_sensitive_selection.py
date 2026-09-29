"""Post-hoc audit of every frozen cost-sensitive candidate; never used for selection."""
from pathlib import Path
import json,numpy as np
from ncd.factorized_graph_program import FactorizedGraphProgram
from ncd.factorized_graph_experiment import _pooled_metrics
from ncd.rules import Rule
runs=[Path('runs/cost_sensitive_graph_seed2593'),Path('runs/cost_sensitive_graph_seed2594')];out={}
for root in runs:
 s=json.loads((root/'summary.json').read_text());seed=str(s['config']['seed']);out[seed]={}
 for mode in ('without_relations','with_relations'):
  syn=s['synthesis'][mode];orientation=Rule.from_dict(syn['orientation_program']);teacher=[];truth=[];features=[]
  for d in sorted((root/'datasets').iterdir()):
   with np.load(d/f'{mode}.npz') as z:teacher.append(z['teacher'])
   with np.load(d/'samples.npz') as z:truth.append(z['target_graph'])
   with np.load(d/'features.npz') as z:features.append(z['features'])
  rows=[]
  for weight,sd in syn['skeleton_programs'].items():
   skeleton=Rule.from_dict(sd)
   for agg in ('and','or'):
    p=FactorizedGraphProgram(skeleton,orientation,agg);pred=[np.stack([p.predict(x) for x in chunk]) for chunk in features];m=_pooled_metrics(teacher,pred,truth);rows.append({'positive_weight':float(weight),'aggregation':agg,'exact_graph_fidelity':m['exact_graph_fidelity'],'active_pair_fidelity':m['active_pair_fidelity'],'truth_exact_graph_accuracy':m['program_exact_graph_accuracy']})
  oracle=max(rows,key=lambda r:(r['exact_graph_fidelity'],r['active_pair_fidelity']));chosen=syn['selected'];out[seed][mode]={'selected':chosen,'confirmation_candidates':rows,'posthoc_oracle':oracle,'selection_regret':oracle['exact_graph_fidelity']-next(r['exact_graph_fidelity'] for r in rows if r['positive_weight']==chosen['positive_weight'] and r['aggregation']==chosen['aggregation'])}
Path('validation/cost_sensitive_selection_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({seed:{m:{'selected_weight':v['selected']['positive_weight'],'oracle':v['posthoc_oracle'],'regret':v['selection_regret']} for m,v in modes.items()} for seed,modes in out.items()},indent=2))
