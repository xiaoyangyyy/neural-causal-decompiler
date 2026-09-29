"""Independent replay of the frozen-teacher weighted program comparison."""
from pathlib import Path
import hashlib,json
import numpy as np
from ncd.graph_experiment import symbolic_probabilities
from ncd.graph_model import GRAPH_FEATURES,decode_graph,graph_probabilities,pair_features
from ncd.multiverse import generate_graph_worlds
from ncd.relational_graph import load_relational_graph
from ncd.rules import fit_rule,Rule
root=Path(__file__).resolve().parents[1]
run=root/'runs/weighted_rules_seed593';source=root/'runs/relational_seed493'
summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
for name,digest in json.loads((run/'manifest.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,name
for name,digest in summary['source_sha256'].items():
    path=source/('models/'+name+'/graph_teacher.pt' if name in summary['config']['modes'] else 'datasets/'+name+'_extraction/features.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
assert json.loads((run/'status.json').read_text(encoding='utf-8'))['state']=='completed'
assert summary['config']['seed']==593 and summary['config']['evaluation_worlds_per_size_split']==24
for mode in summary['config']['modes']:
    teacher=load_relational_graph(source/'models'/mode/'graph_teacher.pt')
    xs=[];ys=[];weights=[]
    for n in (3,5,8):
        with np.load(source/'datasets'/f'n{n}_extraction'/'features.npz') as z:features=z['features']
        mask=~np.eye(n,dtype=bool)
        xs.append(features[:,mask].reshape(-1,len(GRAPH_FEATURES)))
        ys.append(graph_probabilities(teacher,features)[:,mask].argmax(-1).ravel())
        weights.append(np.full(len(ys[-1]),1/(n*(n-1))))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights)
    mass=np.bincount(y,weights=w,minlength=4)
    assert np.array_equal(np.bincount(y,minlength=4),summary['extraction'][mode]['teacher_class_counts'])
    assert np.allclose(mass,summary['extraction'][mode]['equal_world_class_mass'])
    for variant,sw in {'uniform':None,'equal_world':w,'equal_world_and_teacher_class':w/mass[y]}.items():
        saved=summary['results'][mode][variant]
        rule,trace=fit_rule(x,y,GRAPH_FEATURES,max_splits=6,beam_width=3,sample_weight=sw)
        assert rule.to_dict()==saved['tree']
        assert trace==saved['trace']
        assert Rule.from_dict(saved['tree']).to_dict()==rule.to_dict()
        for split in summary['config']['evaluation_splits']:
            for n in (3,5,8):
                worlds=generate_graph_worlds(split,24,n,593,96)
                feature=np.stack([pair_features(world.sample()) for world in worlds])
                truth=np.stack([world.target_graph for world in worlds])
                a=np.stack([decode_graph(p)[0] for p in graph_probabilities(teacher,feature)])
                b=np.stack([decode_graph(p)[0] for p in symbolic_probabilities(rule,feature)])
                mask=np.triu(np.ones((n,n),bool),1)
                al=(a.astype(int)+2*a.transpose(0,2,1))[:,mask]
                bl=(b.astype(int)+2*b.transpose(0,2,1))[:,mask]
                for family in sorted({world.family for world in worlds}):
                    ids=np.array([world.family==family for world in worlds]);t=al[ids];p=bl[ids]
                    active=(t!=0)|(p!=0)
                    expected={'worlds':int(ids.sum()),'teacher_program_all_pair_fidelity':float(np.mean(t==p)),
                              'teacher_program_active_pair_fidelity':float(np.mean(t[active]==p[active])) if active.any() else None,
                              'teacher_program_exact_graph_fidelity':float(np.mean(np.all(t==p,axis=1))),
                              'teacher_truth_exact_graph_accuracy':float(np.mean(np.all(a[ids]==truth[ids],axis=(1,2)))),
                              'program_truth_exact_graph_accuracy':float(np.mean(np.all(b[ids]==truth[ids],axis=(1,2))))}
                    observed=saved['strata'][f'n{n}_{split}_{family}']
                    assert observed==expected,(mode,variant,n,split,family)
        print('replayed',mode,variant,flush=True)
record={'state':'verified','variants':6,'independent_scored_strata':180,'run':'runs/weighted_rules_seed593',
        'scope':'source hashes, extraction weights, symbolic programs, fresh worlds, all reported stratum metrics; no scientific success claim'}
(root/'validation/weighted_rules_replay.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('independent weighted-rule replay passed',flush=True)
