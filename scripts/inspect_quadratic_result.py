import json
from pathlib import Path
s=json.loads(Path('runs/quadratic_readout_seed3193/summary.json').read_text())
for site,v in s['sites'].items():
 print(site)
 for method in ('linear_oblique','quadratic_oblique','quadratic_shuffled','quadratic_random'):
  rows=v[method]['numeric_nodes'];vals={}
  for field in ('natural','targeted','collateral'):
   total=0.;count=0
   for r in rows:
    if field=='natural':n=r['natural_n'];x=r['natural_nmse']
    else:n=r[field]['n'];x=r[field]['nmse']
    if x is not None:total+=n*x;count+=n
   vals[field]=total/count
  print(' ',method,vals)
