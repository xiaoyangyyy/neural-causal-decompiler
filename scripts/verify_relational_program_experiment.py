"""Independent replay for the preregistered relational program experiment."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from ncd.graph_experiment import symbolic_probabilities
from ncd.graph_model import GRAPH_FEATURES,decode_graph,graph_probabilities,pair_features
from ncd.graph_program_features import RELATIONAL_PROGRAM_FEATURES,relational_program_features
from ncd.multiverse import generate_graph_worlds
from ncd.relational_graph import load_relational_graph
from ncd.rules import Rule,fit_rule
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/relational_program_693_694';SOURCE=ROOT/'runs/relational_seed493'
summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'));config=summary['config'];saved_worlds=json.loads((RUN/'worlds.json').read_text(encoding='utf-8'))
assert json.loads((RUN/'status.json').read_text())['state']=='completed'
for name,d in json.loads((RUN/'manifest.json').read_text())['artifacts'].items():assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==d,name
assert hashlib.sha256((ROOT/'docs/RELATIONAL_PROGRAM_PROTOCOL.md').read_bytes()).hexdigest()==summary['protocol_sha256']
for name,d in summary['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d,name
modes=('without_relations','with_relations');variants=('local_composed','local_primitive','relational_primitive');nodes=tuple(config['nodes'])
models={m:load_relational_graph(SOURCE/'models'/m/'graph_teacher.pt') for m in modes};xs={v:[] for v in variants};ys={m:[] for m in modes}
for n in nodes:
 with np.load(SOURCE/'datasets'/f'n{n}_extraction'/'features.npz') as z:base=z['features']
 mask=~np.eye(n,dtype=bool);context=relational_program_features(base)
 xs['local_composed'].append(base[:,mask].reshape(-1,len(GRAPH_FEATURES)));xs['local_primitive'].append(xs['local_composed'][-1]);xs['relational_primitive'].append(context[:,mask].reshape(-1,len(RELATIONAL_PROGRAM_FEATURES)))
 for m in modes:ys[m].append(graph_probabilities(models[m],base)[:,mask].argmax(-1).ravel())
xs={k:np.concatenate(v) for k,v in xs.items()};ys={k:np.concatenate(v) for k,v in ys.items()};rules={}
for m in modes:
 assert np.bincount(ys[m],minlength=4).tolist()==summary['extraction_teacher_class_counts'][m];rules[m]={}
 for v in variants:
  names=RELATIONAL_PROGRAM_FEATURES if v=='relational_primitive' else GRAPH_FEATURES
  rule,trace=fit_rule(xs[v],ys[m],names,max_splits=6,beam_width=3,arithmetic=v=='local_composed')
  saved=Rule.from_dict(json.loads((RUN/'programs'/m/f'{v}.json').read_text()))
  assert rule.to_dict()==saved.to_dict();assert trace==summary['program_traces'][m][v]['trace'];rules[m][v]=rule
  print('refit',m,v,flush=True)
def metric(a,b,t):
 n=a.shape[1];mask=np.triu(np.ones((n,n),bool),1);lab=lambda z:(z.astype(int)+2*z.swapaxes(-2,-1).astype(int))[:,mask]
 a,b,t=lab(a),lab(b),lab(t);active=(a!=0)|(b!=0);cm=np.zeros((4,4),int);np.add.at(cm,(a.ravel(),b.ravel()),1)
 return {'worlds':len(a),'all_pair_fidelity':float(np.mean(a==b)),'active_pair_fidelity':float(np.mean(a[active]==b[active])) if active.any() else None,
 'exact_graph_fidelity':float(np.mean(np.all(a==b,axis=1))),'teacher_exact_graph_accuracy':float(np.mean(np.all(a==t,axis=1))),
 'program_exact_graph_accuracy':float(np.mean(np.all(b==t,axis=1))),'teacher_class_recall':[float(np.mean(b[a==c]==c)) if np.any(a==c) else None for c in range(4)],
 'teacher_program_confusion':cm.tolist(),'per_world_exact_fidelity':np.all(a==b,axis=1).astype(int).tolist()}
observed={}
for seed in config['seeds']:
 observed[str(seed)]={}
 for split in config['splits']:
  for n in nodes:
   key=f'n{n}_{split}';worlds=generate_graph_worlds(split,config['worlds_per_cell'],n,seed,config['samples'])
   assert json.loads(json.dumps([w.metadata() for w in worlds]))==saved_worlds[str(seed)][key]
   base=np.stack([pair_features(w.sample()) for w in worlds]);context=relational_program_features(base);truth=np.stack([w.target_graph for w in worlds]);observed[str(seed)][key]={}
   for m in modes:
    teacher=np.stack([decode_graph(p)[0] for p in graph_probabilities(models[m],base)]);observed[str(seed)][key][m]={}
    for v in variants:
     feature=context if v=='relational_primitive' else base
     program=np.stack([decode_graph(p)[0] for p in symbolic_probabilities(rules[m][v],feature)])
     observed[str(seed)][key][m][v]=metric(teacher,program,truth)
   print('replayed',seed,key,flush=True)
assert observed==summary['evaluation']
primary=[]
for seed in config['seeds']:
 for m in modes:
  a=[];b=[]
  for key in observed[str(seed)]:a+=observed[str(seed)][key][m]['local_composed']['per_world_exact_fidelity'];b+=observed[str(seed)][key][m]['relational_primitive']['per_world_exact_fidelity']
  d=np.asarray(b)-a;radius=math.sqrt(2*math.log(2/.05)/len(d));primary.append({'seed':seed,'mode':m,'worlds':len(d),'local_composed':float(np.mean(a)),'relational_primitive':float(np.mean(b)),'paired_difference':float(d.mean()),'hoeffding95_interval':[float(d.mean()-radius),float(d.mean()+radius)]})
assert primary==summary['primary']
record={'state':'verified','programs':6,'worlds':len(config['seeds'])*len(config['splits'])*sum(nodes)*config['worlds_per_cell'],'world_units':len(config['seeds'])*len(config['splits'])*len(nodes)*config['worlds_per_cell'],'primary':primary,'science_not_certified':True}
(ROOT/'validation/relational_program_replay.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('verified',flush=True)
