"""Exploratory, frozen-teacher weighted program audit on fresh SCM worlds."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from ncd.graph_experiment import symbolic_probabilities
from ncd.graph_model import GRAPH_FEATURES,decode_graph,graph_probabilities,pair_features
from ncd.multiverse import generate_graph_worlds
from ncd.relational_graph import load_relational_graph
from ncd.rules import fit_rule

root=Path(__file__).resolve().parents[1]
out=root/'runs/weighted_rules_seed593'
if out.exists():raise FileExistsError('Preserve previous weighted audit')
out.mkdir(parents=True)
source=root/'runs/relational_seed493'
manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))['artifacts']
def checked(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==manifest[path.relative_to(source).as_posix()]
    return digest
source_hashes={}
for n in (3,5,8):source_hashes[f'n{n}']=checked(source/'datasets'/f'n{n}_extraction'/'features.npz')
for mode in ('without_relations','with_relations'):
    source_hashes[mode]=checked(source/'models'/mode/'graph_teacher.pt')
config={'seed':593,'source':'relational_seed493','modes':['without_relations','with_relations'],'nodes':[3,5,8],
        'variants':['uniform','equal_world','equal_world_and_teacher_class'],'evaluation_splits':['test_id','test_function','test_noise','test_scale','test_intervention'],
        'evaluation_worlds_per_size_split':24,'samples':96,'max_splits':6,'beam_width':3,
        'supervision':'frozen_teacher_only','weights_derived_only_from_extraction_teacher_labels':True,
        'claim':'exploratory comparison; new evaluation seeds are not confirmation after selecting among variants'}
(out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
(out/'status.json').write_text(json.dumps({'state':'running'}),encoding='utf-8')
try:
    evaluation={}
    for split in config['evaluation_splits']:
        for n in config['nodes']:
            worlds=generate_graph_worlds(split,24,n,593,96)
            features=np.stack([pair_features(w.sample()) for w in worlds])
            truth=np.stack([w.target_graph for w in worlds])
            evaluation[f'n{n}_{split}']={'features':features,'truth':truth,'ids':[w.identity for w in worlds],
                'family':[w.family for w in worlds]}
            print('generated',n,split,flush=True)
    results={};provenance={}
    for mode in config['modes']:
        model=load_relational_graph(source/'models'/mode/'graph_teacher.pt')
        xs=[];ys=[];worldweights=[]
        for n in config['nodes']:
            with np.load(source/'datasets'/f'n{n}_extraction'/'features.npz') as z:features=z['features']
            p=graph_probabilities(model,features)
            mask=~np.eye(n,dtype=bool)
            x=features[:,mask].reshape(-1,len(GRAPH_FEATURES));y=p[:,mask].argmax(-1).ravel()
            xs.append(x);ys.append(y);worldweights.append(np.full(len(y),1/(n*(n-1))))
        x=np.concatenate(xs);y=np.concatenate(ys);worldweight=np.concatenate(worldweights)
        mass=np.bincount(y,weights=worldweight,minlength=4)
        if np.any(mass<=0):raise ValueError('Frozen extraction teacher lacks a class; no class balancing defined')
        provenance[mode]={'teacher_class_counts':np.bincount(y,minlength=4).tolist(),'equal_world_class_mass':mass.tolist(),
                          'extraction_rows':int(len(y))}
        variants={'uniform':None,'equal_world':worldweight,'equal_world_and_teacher_class':worldweight/mass[y]}
        results[mode]={}
        for variant,weight in variants.items():
            print('fitting',mode,variant,flush=True)
            start=time.monotonic()
            rule,trace=fit_rule(x,y,GRAPH_FEATURES,max_splits=6,beam_width=3,sample_weight=weight)
            part={'fit_seconds':time.monotonic()-start,'complexity':rule.complexity,'tree':rule.to_dict(),
                  'trace':trace,'strata':{}}
            for key,data in evaluation.items():
                features=data['features'];n=features.shape[1];mask=np.triu(np.ones((n,n),bool),1)
                teacher=graph_probabilities(model,features)
                symbolic=symbolic_probabilities(rule,features)
                t=np.stack([decode_graph(p)[0] for p in teacher]);s=np.stack([decode_graph(p)[0] for p in symbolic])
                tl=(t.astype(int)+2*t.transpose(0,2,1))[:,mask]
                sl=(s.astype(int)+2*s.transpose(0,2,1))[:,mask]
                for family in sorted(set(data['family'])):
                    ids=np.array([f==family for f in data['family']]);a=tl[ids];b=sl[ids]
                    active=(a!=0)|(b!=0)
                    part['strata'][key+'_'+family]={'worlds':int(ids.sum()),'teacher_program_all_pair_fidelity':float(np.mean(a==b)),
                        'teacher_program_active_pair_fidelity':float(np.mean(a[active]==b[active])) if active.any() else None,
                        'teacher_program_exact_graph_fidelity':float(np.mean(np.all(a==b,axis=1))),
                        'teacher_truth_exact_graph_accuracy':float(np.mean(np.all(t[ids]==data['truth'][ids],axis=(1,2)))),
                        'program_truth_exact_graph_accuracy':float(np.mean(np.all(s[ids]==data['truth'][ids],axis=(1,2))))}
            results[mode][variant]=part
            (out/'status.json').write_text(json.dumps({'state':'running','last_finished':mode+'/'+variant},indent=2),encoding='utf-8')
    record={'config':config,'source_sha256':source_hashes,'extraction':provenance,'results':results,
            'boundary':'Only program extraction changed. Frozen teachers and test worlds did not enter fit weights. Exploratory comparison; no success claim from variant selection.'}
    (out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    (out/'status.json').write_text(json.dumps({'state':'completed'}),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print('weighted-rule audit completed',flush=True)
except BaseException as error:
    (out/'status.json').write_text(json.dumps({'state':'failed','error':str(error)}),encoding='utf-8')
    raise
