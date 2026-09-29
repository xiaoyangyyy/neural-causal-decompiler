import json
from pathlib import Path
import numpy as np
from ncd.graph_model import graph_labels
runs=[Path('runs/factorized_graph_seed2393'),Path('runs/factorized_graph_seed2394')]
modes=('without_relations','with_relations')
all_labels={k:[] for k in ('teacher','local','factorized','truth')}
for root in runs:
    for d in sorted((root/'datasets').iterdir()):
        with np.load(d/'samples.npz') as z: truth=z['target_graph']
        n=truth.shape[1]; mask=np.triu(np.ones((n,n),bool),1)
        for mode in modes:
            with np.load(d/f'{mode}.npz') as z:
                for key in ('teacher','local','factorized'):
                    all_labels[key].append(np.stack([graph_labels(g)[mask] for g in z[key]]))
            all_labels['truth'].append(np.stack([graph_labels(g)[mask] for g in truth]))
def metrics(method):
    pairs=[];ex=[];ta=[];pa=[]
    for a,b,t in zip(all_labels['teacher'],all_labels[method],all_labels['truth']):
        pairs.append((a.ravel(),b.ravel())); ex.append(np.all(a==b,1));ta.append(np.all(a==t,1));pa.append(np.all(b==t,1))
    a=np.concatenate([x for x,y in pairs]);b=np.concatenate([y for x,y in pairs]);active=(a!=0)|(b!=0)
    return {'worlds':sum(len(x) for x in ex),'pairs':len(a),'all_pair_fidelity':float(np.mean(a==b)),'active_pair_fidelity':float(np.mean(a[active]==b[active])),'exact_graph_fidelity':float(np.mean(np.concatenate(ex))),'teacher_exact_graph_accuracy':float(np.mean(np.concatenate(ta))),'program_exact_graph_accuracy':float(np.mean(np.concatenate(pa)))}
local=metrics('local');factorized=metrics('factorized')
per_seed=[]
for root in runs:
 s=json.loads((root/'summary.json').read_text());per_seed.append({'seed':s['config']['seed'],'source':Path(s['config']['source']).name,'verified':True,'local':s['overall']['local'],'factorized':s['overall']['factorized'],'decision_components':s['decision_components']})
accept={'protocol':'docs/FACTORIZED_GRAPH_PROGRAM_PROTOCOL.md','runs':per_seed,'pooled':{'local':local,'factorized':factorized,'exact_gain':factorized['exact_graph_fidelity']-local['exact_graph_fidelity'],'active_gain':factorized['active_pair_fidelity']-local['active_pair_fidelity']},'criteria':{'both_replayed':True,'no_exact_decline_each_seed':all(x['decision_components']['per_mode_non_decline'] for x in per_seed),'pooled_exact_gain_at_least_0.02':factorized['exact_graph_fidelity']-local['exact_graph_fidelity']>=.02,'pooled_active_non_decline':factorized['active_pair_fidelity']>=local['active_pair_fidelity']},'passed':False,'claim':'negative behavioral graph-program extraction result; truth metrics are diagnostic'}
Path('validation/factorized_graph_acceptance.json').write_text(json.dumps(accept,indent=2)+'\n')
print(json.dumps(accept,indent=2))

