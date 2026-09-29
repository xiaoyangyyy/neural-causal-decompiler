"""Failure decomposition for frozen factorized graph confirmations."""
from pathlib import Path
import json,numpy as np
from ncd.graph_model import graph_labels
runs=[Path('runs/factorized_graph_seed2393'),Path('runs/factorized_graph_seed2394')]
methods=('local','factorized'); out={}
for root in runs:
 rows={m:{'worlds':0,'skeleton_exact':0,'teacher_present':0,'orientation_correct':0,'false_absent':0,'false_present':0,'wrong_orientation':0} for m in methods}
 for d in sorted((root/'datasets').iterdir()):
  for mode in ('without_relations','with_relations'):
   with np.load(d/f'{mode}.npz') as z:
    teacher=z['teacher']; preds={m:z[m] for m in methods}
   n=teacher.shape[1];mask=np.triu(np.ones((n,n),bool),1);tl=np.stack([graph_labels(g)[mask] for g in teacher])
   for m,pred in preds.items():
    pl=np.stack([graph_labels(g)[mask] for g in pred]);r=rows[m];r['worlds']+=len(tl);r['skeleton_exact']+=int(np.sum(np.all((tl!=0)==(pl!=0),axis=1)));present=tl!=0;r['teacher_present']+=int(present.sum());r['orientation_correct']+=int(np.sum((tl==pl)&present));r['false_absent']+=int(np.sum(present&(pl==0)));r['false_present']+=int(np.sum((tl==0)&(pl!=0)));r['wrong_orientation']+=int(np.sum(present&(pl!=0)&(tl!=pl)))
 for m,r in rows.items():r['skeleton_exact_rate']=r['skeleton_exact']/r['worlds'];r['orientation_given_teacher_present']=r['orientation_correct']/r['teacher_present']
 out[str(json.loads((root/'summary.json').read_text())['config']['seed'])]=rows
Path('validation/factorized_graph_failure_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
