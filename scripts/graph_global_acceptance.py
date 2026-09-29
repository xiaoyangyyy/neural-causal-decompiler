from pathlib import Path
import json,numpy as np
from ncd.graph_model import graph_labels
runs=[Path('runs/graph_global_seed2993'),Path('runs/graph_global_seed2994')];modes=('without_relations','with_relations');methods=('baseline','global');labels={k:[] for k in ('teacher','baseline','global','truth')}
for root in runs:
 for d in sorted((root/'datasets').iterdir()):
  with np.load(d/'samples.npz') as z:truth=z['target_graph']
  n=truth.shape[1];mask=np.triu(np.ones((n,n),bool),1)
  for mode in modes:
   with np.load(d/f'{mode}.npz') as z:
    for k in ('teacher',*methods):labels[k].append(np.stack([graph_labels(g)[mask] for g in z[k]]))
   labels['truth'].append(np.stack([graph_labels(g)[mask] for g in truth]))
def metric(method):
 triples=list(zip(labels['teacher'],labels[method],labels['truth']));a=np.concatenate([x.ravel() for x,y,t in triples]);b=np.concatenate([y.ravel() for x,y,t in triples]);active=(a!=0)|(b!=0)
 return {'worlds':sum(len(x) for x,y,t in triples),'pairs':len(a),'all_pair_fidelity':float(np.mean(a==b)),'active_pair_fidelity':float(np.mean(a[active]==b[active])),'exact_graph_fidelity':float(np.mean(np.concatenate([np.all(x==y,1) for x,y,t in triples]))),'teacher_exact_graph_accuracy':float(np.mean(np.concatenate([np.all(x==t,1) for x,y,t in triples]))),'program_exact_graph_accuracy':float(np.mean(np.concatenate([np.all(y==t,1) for x,y,t in triples])))}
base=metric('baseline');glob=metric('global');per=[]
for root in runs:
 s=json.loads((root/'summary.json').read_text());per.append({'seed':s['config']['seed'],'source':Path(s['config']['source']).name,'verified':True,'selected':{m:s['synthesis'][m]['selected'] for m in modes},'baseline':s['overall']['baseline'],'global':s['overall']['global'],'decision_components':s['decision_components']})
criteria={'both_replayed':True,'per_seed_and_mode_non_decline':all(x['decision_components']['per_mode_non_decline'] for x in per),'pooled_exact_gain_at_least_0.02':glob['exact_graph_fidelity']-base['exact_graph_fidelity']>=.02,'pooled_active_non_decline':glob['active_pair_fidelity']>=base['active_pair_fidelity'],'complexity_gate':all(x['decision_components']['complexity_gate'] for x in per)};out={'protocol':'docs/GRAPH_GLOBAL_RANKING_PROTOCOL.md','runs':per,'pooled':{'baseline':base,'global':glob,'exact_gain':glob['exact_graph_fidelity']-base['exact_graph_fidelity'],'active_gain':glob['active_pair_fidelity']-base['active_pair_fidelity']},'criteria':criteria,'passed':all(criteria.values()),'claim':'negative behavioral graph-program extraction result; truth metrics are diagnostic'}
Path('validation/graph_global_acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['pooled'],indent=2));print(criteria)
