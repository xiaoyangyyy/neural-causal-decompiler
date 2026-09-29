"""Prospective relational-context program extraction experiment."""
from pathlib import Path
import hashlib,json,math,shutil,time
import numpy as np
from ncd.graph_experiment import symbolic_probabilities
from ncd.graph_model import GRAPH_FEATURES,decode_graph,graph_probabilities,pair_features
from ncd.graph_program_features import RELATIONAL_PROGRAM_FEATURES,relational_program_features
from ncd.multiverse import generate_graph_worlds
from ncd.relational_graph import load_relational_graph
from ncd.rules import fit_rule
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'runs/relational_seed493';OUT=ROOT/'runs/relational_program_693_694'
MODES=('without_relations','with_relations');VARIANTS=('local_composed','local_primitive','relational_primitive')
SPLITS=('test_id','test_function','test_noise','test_scale','test_intervention');NODES=(3,5,8);SEEDS=(693,694);COUNT=64

def labels(a):return a.astype(int)+2*a.swapaxes(-2,-1).astype(int)
def checked(path,manifest):
 d=hashlib.sha256(path.read_bytes()).hexdigest();assert manifest[path.relative_to(SOURCE).as_posix()]==d;return d
def metrics(teacher,program,truth):
 n=teacher.shape[1];mask=np.triu(np.ones((n,n),bool),1);a=labels(teacher)[:,mask];b=labels(program)[:,mask];t=labels(truth)[:,mask]
 active=(a!=0)|(b!=0);cm=np.zeros((4,4),int);np.add.at(cm,(a.ravel(),b.ravel()),1)
 recall=[float(np.mean(b[a==c]==c)) if np.any(a==c) else None for c in range(4)]
 return {'worlds':len(a),'all_pair_fidelity':float(np.mean(a==b)),'active_pair_fidelity':float(np.mean(a[active]==b[active])) if active.any() else None,
 'exact_graph_fidelity':float(np.mean(np.all(a==b,axis=1))),'teacher_exact_graph_accuracy':float(np.mean(np.all(a==t,axis=1))),
 'program_exact_graph_accuracy':float(np.mean(np.all(b==t,axis=1))),'teacher_class_recall':recall,
 'teacher_program_confusion':cm.tolist(),'per_world_exact_fidelity':np.all(a==b,axis=1).astype(int).tolist()}
if OUT.exists():raise FileExistsError('Preserve prior relational program run')
OUT.mkdir(parents=True);(OUT/'status.json').write_text(json.dumps({'state':'running'}),encoding='utf-8')
try:
 manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))['artifacts'];hashes={};models={}
 for mode in MODES:
  path=SOURCE/'models'/mode/'graph_teacher.pt';hashes[path.relative_to(ROOT).as_posix()]=checked(path,manifest);models[mode]=load_relational_graph(path)
 extraction={variant:{'x':[],'names':None,'arithmetic':None} for variant in VARIANTS};teacher_labels={mode:[] for mode in MODES}
 for n in NODES:
  path=SOURCE/'datasets'/f'n{n}_extraction'/'features.npz';hashes[path.relative_to(ROOT).as_posix()]=checked(path,manifest)
  with np.load(path) as z:base=z['features']
  mask=~np.eye(n,dtype=bool);context=relational_program_features(base)
  extraction['local_composed']['x'].append(base[:,mask].reshape(-1,len(GRAPH_FEATURES)))
  extraction['local_primitive']['x'].append(base[:,mask].reshape(-1,len(GRAPH_FEATURES)))
  extraction['relational_primitive']['x'].append(context[:,mask].reshape(-1,len(RELATIONAL_PROGRAM_FEATURES)))
  for mode in MODES:teacher_labels[mode].append(graph_probabilities(models[mode],base)[:,mask].argmax(-1).ravel())
 extraction['local_composed'].update(names=GRAPH_FEATURES,arithmetic=True)
 extraction['local_primitive'].update(names=GRAPH_FEATURES,arithmetic=False)
 extraction['relational_primitive'].update(names=RELATIONAL_PROGRAM_FEATURES,arithmetic=False)
 for value in extraction.values():value['x']=np.concatenate(value['x'])
 teacher_labels={k:np.concatenate(v) for k,v in teacher_labels.items()}
 programs={};traces={}
 for mode in MODES:
  programs[mode]={};traces[mode]={}
  for variant in VARIANTS:
   value=extraction[variant];print('fit',mode,variant,flush=True);start=time.monotonic()
   rule,trace=fit_rule(value['x'],teacher_labels[mode],value['names'],max_splits=6,beam_width=3,arithmetic=value['arithmetic'])
   programs[mode][variant]=rule;traces[mode][variant]={'runtime_seconds':time.monotonic()-start,'trace':trace}
   path=OUT/'programs'/mode;path.mkdir(parents=True,exist_ok=True);(path/f'{variant}.json').write_text(json.dumps(rule.to_dict(),indent=2),encoding='utf-8')
 evaluations={};world_records={}
 for seed in SEEDS:
  evaluations[str(seed)]={};world_records[str(seed)]={}
  for split in SPLITS:
   for n in NODES:
    key=f'n{n}_{split}';print('evaluate',seed,key,flush=True);worlds=generate_graph_worlds(split,COUNT,n,seed,96)
    base=np.stack([pair_features(w.sample()) for w in worlds]);context=relational_program_features(base);truth=np.stack([w.target_graph for w in worlds])
    world_records[str(seed)][key]=[w.metadata() for w in worlds];evaluations[str(seed)][key]={}
    for mode in MODES:
     teacher=np.stack([decode_graph(p)[0] for p in graph_probabilities(models[mode],base)]);evaluations[str(seed)][key][mode]={}
     for variant in VARIANTS:
      feature=context if variant=='relational_primitive' else base
      program=np.stack([decode_graph(p)[0] for p in symbolic_probabilities(programs[mode][variant],feature)])
      evaluations[str(seed)][key][mode][variant]=metrics(teacher,program,truth)
 protocol_hash=hashlib.sha256((ROOT/'docs/RELATIONAL_PROGRAM_PROTOCOL.md').read_bytes()).hexdigest()
 primary=[]
 for seed in SEEDS:
  for mode in MODES:
   a=[];b=[]
   for key in evaluations[str(seed)]:
    a+=evaluations[str(seed)][key][mode]['local_composed']['per_world_exact_fidelity'];b+=evaluations[str(seed)][key][mode]['relational_primitive']['per_world_exact_fidelity']
   diff=np.asarray(b)-np.asarray(a);radius=math.sqrt(2*math.log(2/.05)/len(diff))
   primary.append({'seed':seed,'mode':mode,'worlds':len(diff),'local_composed':float(np.mean(a)),'relational_primitive':float(np.mean(b)),
    'paired_difference':float(diff.mean()),'hoeffding95_interval':[float(diff.mean()-radius),float(diff.mean()+radius)]})
 summary={'config':{'source':'runs/relational_seed493','seeds':list(SEEDS),'nodes':list(NODES),'splits':list(SPLITS),'worlds_per_cell':COUNT,'samples':96,
  'max_splits':6,'beam_width':3,'penalty':.001,'variants':list(VARIANTS)},'protocol_sha256':protocol_hash,'source_sha256':hashes,
  'extraction_teacher_class_counts':{m:np.bincount(y,minlength=4).tolist() for m,y in teacher_labels.items()},'program_traces':traces,
  'evaluation':evaluations,'primary':primary,'boundary':'behavioral decompilation of frozen teachers; no internal-site or teacher-accuracy claim'}
 (OUT/'worlds.json').write_text(json.dumps(world_records,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
 (OUT/'status.json').write_text(json.dumps({'state':'completed'}),encoding='utf-8');snapshot=OUT/'source';snapshot.mkdir()
 for p in (ROOT/'ncd').glob('*.py'):shutil.copy2(p,snapshot/p.name)
 artifacts={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}
 (OUT/'manifest.json').write_text(json.dumps({'artifacts':artifacts},indent=2),encoding='utf-8');print('completed',flush=True)
except BaseException as e:
 (OUT/'status.json').write_text(json.dumps({'state':'failed','type':type(e).__name__,'error':str(e)}),encoding='utf-8');raise
