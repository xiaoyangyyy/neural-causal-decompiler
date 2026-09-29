"""Aggregate frozen cost-sensitive graph confirmations."""
from pathlib import Path
import json,numpy as np
from ncd.graph_model import graph_labels
runs=[Path('runs/cost_sensitive_graph_seed2593'),Path('runs/cost_sensitive_graph_seed2594')];modes=('without_relations','with_relations');methods=('baseline','cost_sensitive')
labels={k:[] for k in ('teacher','baseline','cost_sensitive','truth')}
for root in runs:
 for d in sorted((root/'datasets').iterdir()):
  with np.load(d/'samples.npz') as z:truth=z['target_graph']
  n=truth.shape[1];mask=np.triu(np.ones((n,n),bool),1)
  for mode in modes:
   with np.load(d/f'{mode}.npz') as z:
    for key in ('teacher',*methods):labels[key].append(np.stack([graph_labels(g)[mask] for g in z[key]]))
   labels['truth'].append(np.stack([graph_labels(g)[mask] for g in truth]))
def metric(method):
 triples=list(zip(labels['teacher'],labels[method],labels['truth']));a=np.concatenate([x.ravel() for x,y,t in triples]);b=np.concatenate([y.ravel() for x,y,t in triples]);active=(a!=0)|(b!=0)
 return {'worlds':sum(len(x) for x,y,t in triples),'pairs':len(a),'all_pair_fidelity':float(np.mean(a==b)),'active_pair_fidelity':float(np.mean(a[active]==b[active])),'exact_graph_fidelity':float(np.mean(np.concatenate([np.all(x==y,1) for x,y,t in triples]))),'teacher_exact_graph_accuracy':float(np.mean(np.concatenate([np.all(x==t,1) for x,y,t in triples]))),'program_exact_graph_accuracy':float(np.mean(np.concatenate([np.all(y==t,1) for x,y,t in triples])))}
base=metric('baseline');cost=metric('cost_sensitive');per=[]
for root in runs:
 s=json.loads((root/'summary.json').read_text());per.append({'seed':s['config']['seed'],'source':Path(s['config']['source']).name,'verified':True,'selected':{m:s['synthesis'][m]['selected'] for m in modes},'baseline':s['overall']['baseline'],'cost_sensitive':s['overall']['cost_sensitive'],'decision_components':s['decision_components']})
criteria={'both_replayed':True,'per_seed_and_mode_non_decline':all(x['decision_components']['per_mode_non_decline'] for x in per),'pooled_exact_gain_at_least_0.02':cost['exact_graph_fidelity']-base['exact_graph_fidelity']>=.02,'pooled_active_non_decline':cost['active_pair_fidelity']>=base['active_pair_fidelity']}
out={'protocol':'docs/COST_SENSITIVE_FACTORIZED_GRAPH_PROTOCOL.md','runs':per,'pooled':{'baseline':base,'cost_sensitive':cost,'exact_gain':cost['exact_graph_fidelity']-base['exact_graph_fidelity'],'active_gain':cost['active_pair_fidelity']-base['active_pair_fidelity']},'criteria':criteria,'passed':all(criteria.values()),'claim':'negative behavioral graph-program extraction result; truth metrics are diagnostic'}
Path('validation/cost_sensitive_graph_acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
