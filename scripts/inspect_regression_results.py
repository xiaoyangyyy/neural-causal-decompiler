from pathlib import Path
import json
for seed in (993,994):
 s=json.loads(Path(f'runs/regression_numeric_seed{seed}/summary.json').read_text());print('\nseed',seed,'groups',len(s['groups']),'heldout',len(s['held_out_combinations']))
 for site,methods in s['sites'].items():
  for method,row in methods.items():
   ns=row['numeric_nodes'];tn=sum(x['targeted']['n'] for x in ns);cn=sum(x['collateral']['n'] for x in ns)
   ag=lambda part,key,n:sum(x[part]['n']*x[part][key] for x in ns if x[part]['n'])/n
   nat=sum(x['natural_n']*x['natural_nmse'] for x in ns)/sum(x['natural_n'] for x in ns);b=row['behavioral']
   print(site,method,'nat',round(nat,3),'target',round(ag('targeted','nmse',tn),3),'base',round(ag('targeted','no_intervention_nmse',tn),3),'coll',round(ag('collateral','nmse',cn),3),'info',b['informative_pairs'],'acc',b['informative_accuracy'])
