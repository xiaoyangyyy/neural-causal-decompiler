"""Post-hoc diagnostics of frozen graph predictions; never a model selection score."""
from pathlib import Path
import json,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=[];sources={}
for seed in (493,494):
    run=root/f'runs/relational_seed{seed}'
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))['artifacts']
    def checked(path):
        data=path.read_bytes();key=path.relative_to(run).as_posix()
        digest=hashlib.sha256(data).hexdigest()
        assert manifest[key]==digest,key
        sources[path.relative_to(root).as_posix()]=digest
        return data
    for n in (3,5,8):
        mask=np.triu(np.ones((n,n),bool),1)
        for split in ('test_id','test_function','test_noise','test_scale','test_intervention'):
            worlds=json.loads(checked(run/'datasets'/f'n{n}_{split}'/'worlds.json'))
            target=np.array([w['target_graph'] for w in worlds],bool)
            def labels(a):
                return (a.astype(int)+2*a.transpose(0,2,1))[:,mask]
            truth=labels(target)
            for mode in ('without_relations','with_relations'):
                path=run/'evaluations'/mode/f'n{n}_{split}'/'predictions.npz';checked(path)
                with np.load(path) as z:
                    teacher=labels(z['neural']);program=labels(z['symbolic'])
                for family in sorted({w['family'] for w in worlds}):
                    take=np.array([w['family']==family for w in worlds]);t=truth[take];a=teacher[take];p=program[take]
                    def diagnose(pred):
                        confusion=np.zeros((4,4),int);np.add.at(confusion,(t.ravel(),pred.ravel()),1)
                        missed=int(np.sum((t!=0)&(pred==0)));extra=int(np.sum((t==0)&(pred!=0)))
                        reverse=int(np.sum(((t==1)&(pred==2))|((t==2)&(pred==1))))
                        ambiguity=int(np.sum((t!=0)&(pred!=0)&(t!=pred)))-reverse
                        assert missed+extra+reverse+ambiguity==int(np.sum(t!=pred))
                        return {'confusion_true_rows_pred_columns':confusion.tolist(),'missed_edges':missed,'extra_edges':extra,'reversed_edges':reverse,'directed_undirected_mismatch':ambiguity,'exact_graph_accuracy':float(np.mean(np.all(t==pred,axis=1))),'edge_presence_recall':float(np.mean(pred[t!=0]!=0)) if np.any(t!=0) else None}
                    active=(a!=0)|(p!=0)
                    rows.append({'seed':seed,'nodes':n,'split':split,'family':family,'mode':mode,'worlds':int(take.sum()),'pair_count':int(t.size),'true_edge_pairs':int(np.sum(t!=0)),'neural':diagnose(a),'symbolic':diagnose(p),'all_pair_fidelity':float(np.mean(a==p)),'active_pair_fidelity':float(np.mean(a[active]==p[active])) if active.any() else None,'both_absent_fraction':float(np.mean((a==0)&(p==0))),'exact_graph_fidelity':float(np.mean(np.all(a==p,axis=1)))})
record={'scope':'post-hoc frozen test diagnostics; no selection or confirmatory significance claim','label_order':['absent','i_to_j','j_to_i','undirected'],'sources':sources,'rows':rows}
(root/'validation/relational_failure_audit.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
lines=['# Frozen graph failure audit','','Exploratory diagnostics of previously verified predictions. No teacher or program is changed.','Confusion matrices and all five environments are in `validation/relational_failure_audit.json`.','Active pairs are those where either teacher or program predicts an edge; this is a diagnostic conditional subset, not a replacement primary metric.','','| Seed | Nodes | Family | Mode | Worlds | Neural missed / extra / reversed / ambiguity | All-pair fidelity | Active-pair fidelity | Exact-graph fidelity |','|---|---|---|---|---:|---|---:|---:|---:|']
for r in rows:
    if r['split']!='test_id':continue
    d=r['neural'];counts=' / '.join(str(d[k]) for k in ('missed_edges','extra_edges','reversed_edges','directed_undirected_mismatch'))
    active='n/a' if r['active_pair_fidelity'] is None else f"{r['active_pair_fidelity']:.1%}"
    lines.append(f"| {r['seed']} | {r['nodes']} | {r['family']} | {r['mode']} | {r['worlds']} | {counts} | {r['all_pair_fidelity']:.1%} | {active} | {r['exact_graph_fidelity']:.1%} |")
(root/'RESULTS_GRAPH_FAILURES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'Validated input hashes and partitioned errors in {len(rows)} strata.')
