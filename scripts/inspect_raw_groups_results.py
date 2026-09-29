from pathlib import Path
import json,numpy as np
for seed in (793,794):
 s=json.loads(Path(f'runs/raw_numeric_seed{seed}/summary.json').read_text());print('\nseed',seed)
 labels=[]
 for g in s['groups']:
  a=g['ast'];op=a['op']
  if op in ('mean','var_stat') and a['args'][0]['op']=='var':label=op+'_'+('x' if a['args'][0]['index']==0 else 'y')
  elif op=='sqrt':label='std_'+('x' if a['args'][0]['args'][0]['index']==0 else 'y')
  elif op=='div':label='variance_ratio'
  else:label='corr'
  labels.append(label)
 print(labels)
 for site in s['sites']:
  rows=s['sites'][site]['numeric']['numeric_nodes']
  print(site,[(labels[i],round(r['natural_nmse'],2),round(r['targeted']['nmse'],2),round(r['targeted']['no_intervention_nmse'],2)) for i,r in enumerate(rows)])
