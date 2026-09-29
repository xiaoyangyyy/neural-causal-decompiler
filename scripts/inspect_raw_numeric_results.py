from pathlib import Path
import json
for seed in (793,794):
 s=json.loads(Path(f'runs/raw_numeric_seed{seed}/summary.json').read_text())
 print('seed',seed,'groups',len(s['groups']))
 for site,methods in s['sites'].items():
  for name,row in methods.items():
   nodes=row['numeric_nodes'];tn=sum(n['targeted']['n'] for n in nodes);cn=sum(n['collateral']['n'] for n in nodes)
   weighted=lambda part,n:sum(x[part]['n']*x[part]['nmse'] for x in nodes if x[part]['n'])/n if n else None
   b=row['behavioral']
   print(site,name,'target',round(weighted('targeted',tn),3),'coll',round(weighted('collateral',cn),3),'natural',round(sum(n['natural_n']*n['natural_nmse'] for n in nodes)/sum(n['natural_n'] for n in nodes),3),'informative',b['informative_pairs'],'acc',b['informative_accuracy'])
