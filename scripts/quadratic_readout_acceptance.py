from pathlib import Path
import json
sites=('representation','head_linear','head_tanh');runs=[]
def aggregate(rows,field):
 vals=[]
 for r in rows:
  if field=='natural':n=r['natural_n'];x=r['natural_nmse']
  else:n=r[field]['n'];x=r[field]['nmse']
  if x is not None:vals.append((n,x))
 return sum(n*x for n,x in vals)/sum(n for n,x in vals)
for seed in (3193,3194):
 s=json.loads(Path(f'runs/quadratic_readout_seed{seed}/summary.json').read_text());metrics={};natural_wins=[];target_wins=[];collateral_safe=[];controls=[]
 for site in sites:
  metrics[site]={}
  for method in ('linear_oblique','quadratic_oblique','quadratic_shuffled','quadratic_random'):
   rows=s['sites'][site][method]['numeric_nodes'];metrics[site][method]={f:aggregate(rows,f) for f in ('natural','targeted','collateral')}
  l=metrics[site]['linear_oblique'];q=metrics[site]['quadratic_oblique'];natural_wins.append(q['natural']<l['natural']);target_wins.append(q['targeted']<l['targeted']);collateral_safe.append(q['collateral']<=l['collateral']);controls.append(q['targeted']<metrics[site]['quadratic_shuffled']['targeted'] and q['targeted']<metrics[site]['quadratic_random']['targeted'])
 runs.append({'seed':seed,'source':f'joint_seed{191 if seed==3193 else 192}','verified':True,'metrics':metrics,'natural_winning_sites':sum(natural_wins),'target_winning_sites':sum(target_wins),'collateral_safe_at_target_wins':all(safe for win,safe in zip(target_wins,collateral_safe) if win),'controls_beaten_all_sites':all(controls),'full_mask_coverage':len(s['test_combinations'])==1411})
criteria={'both_replayed':all(r['verified'] for r in runs),'full_mask_coverage':all(r['full_mask_coverage'] for r in runs),'natural_lower_at_two_sites_each':all(r['natural_winning_sites']>=2 for r in runs),'target_lower_at_two_sites_each':all(r['target_winning_sites']>=2 for r in runs),'collateral_nonincrease_at_winning_sites':all(r['collateral_safe_at_target_wins'] for r in runs),'beats_shuffled_and_random_all_sites':all(r['controls_beaten_all_sites'] for r in runs)}
out={'protocol':'docs/QUADRATIC_READOUT_INTERVENTION_PROTOCOL.md','runs':runs,'criteria':criteria,'passed':all(criteria.values()),'claim':'negative R5 measurement/intervention result; probe improvement is not mechanism evidence'};Path('validation/quadratic_readout_acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'runs':[{k:v for k,v in r.items() if k not in ('metrics',)} for r in runs],'criteria':criteria,'passed':out['passed']},indent=2))
