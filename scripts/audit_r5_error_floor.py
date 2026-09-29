from pathlib import Path
import json,numpy as np
for seed in (1193,1194):
 s=json.loads((Path('runs')/f'oblique_seed{seed}'/'summary.json').read_text());print(seed)
 for site,v in s['sites'].items():
  rows=v['oblique_numeric']['numeric_nodes'];nat=[];tar=[];before=[];coll=[]
  for r in rows:
   if r['natural_nmse'] is not None:nat.extend([r['natural_nmse']]*r['natural_n'])
   if r['targeted']['nmse'] is not None:tar.extend([r['targeted']['nmse']]*r['targeted']['n']);before.extend([r['targeted']['no_intervention_nmse']]*r['targeted']['n'])
   if r['collateral']['nmse'] is not None:coll.extend([r['collateral']['nmse']]*r['collateral']['n'])
  print(site,{'natural_weighted':float(np.mean(nat)),'target':float(np.mean(tar)),'no_intervention':float(np.mean(before)),'collateral':float(np.mean(coll))})
