from pathlib import Path
import json
r=json.loads(Path('runs/weighted_rules_seed593/summary.json').read_text(encoding='utf-8'))
for mode,variants in r['results'].items():
 print(mode,r['extraction'][mode])
 for name,v in variants.items():
  strata=v['strata'];tot=sum(x['worlds'] for x in strata.values())
  agg=lambda key:sum(x['worlds']*x[key] for x in strata.values())/tot
  print(name,'complexity',v['complexity'],'all',round(agg('teacher_program_all_pair_fidelity'),3),'active',round(agg('teacher_program_active_pair_fidelity'),3),'graph_fid',round(agg('teacher_program_exact_graph_fidelity'),3),'graph_truth',round(agg('program_truth_exact_graph_accuracy'),3))
  for n in (3,5,8):
   rows=[x for k,x in strata.items() if k.startswith(f'n{n}_test_id_')]
   total=sum(x['worlds'] for x in rows)
   print(' n',n,'graph_fid',round(sum(x['worlds']*x['teacher_program_exact_graph_fidelity'] for x in rows)/total,3),'graph_truth',round(sum(x['worlds']*x['program_truth_exact_graph_accuracy'] for x in rows)/total,3))
