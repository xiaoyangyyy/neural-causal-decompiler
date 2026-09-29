import json
from pathlib import Path
for seed in (3193,3194):
 s=json.loads(Path(f'runs/quadratic_readout_seed{seed}/summary.json').read_text());print(seed)
 for site,v in s['sites'].items():
  print(site,end=' ')
  for method in ('linear_oblique','quadratic_oblique','quadratic_shuffled','quadratic_random'):
   rows=v[method]['numeric_nodes'];vals=[]
   for field in ('natural','targeted','collateral'):
    weighted=[]
    for r in rows:
     n=r['natural_n'] if field=='natural' else r[field]['n'];x=r['natural_nmse'] if field=='natural' else r[field]['nmse']
     if x is not None:weighted.append((n,x))
    vals.append(sum(n*x for n,x in weighted)/sum(n for n,x in weighted))
   print(method+':'+'/'.join(f'{x:.3f}' for x in vals),end='  ')
  print()
