"""Bounded beam search over typed expressions and decision programs."""
import json
import numpy as np
from .dsl import Expr, Program
from .statistics import FEATURES

def expression_library():
    base = [Expr("feature", feature=f) for f in FEATURES]
    result = list(base)
    # Explicit domain prior. Kept identical for teacher and ground-truth baselines.
    for left, right in [("resdep_xy","resdep_yx"), ("reserr_xy","reserr_yx"),
                        ("skew_x","skew_y"), ("kurt_x","kurt_y"), ("mixed_xy","mixed_yx")]:
        a, b = Expr("feature",feature=left), Expr("feature",feature=right)
        diff = Expr("sub", (a,b))
        result.extend([diff, Expr("abs",(diff,)), Expr("add",(a,b))])
    result.extend(Expr("abs",(e,)) for e in base)
    return result

def synthesize(features, labels, *, penalty=.001, beam_width=4, max_splits=4, min_leaf=8):
    features, labels = np.asarray(features), np.asarray(labels, dtype=int)
    if len(features) != len(labels) or len(labels) == 0 or not np.isfinite(features).all():
        raise ValueError("Invalid synthesis data")
    if features.ndim != 2 or features.shape[1] != len(FEATURES) or not np.isin(labels, range(4)).all():
        raise ValueError("Invalid features or labels")
    if penalty < 0 or beam_width < 1 or max_splits < 0 or min_leaf < 1: raise ValueError("Invalid search budget")
    expressions = expression_library()
    values = np.column_stack([e.evaluate(features) for e in expressions])
    thresholds = [np.unique(np.quantile(values[:,i], np.linspace(.05,.95,13))) for i in range(len(expressions))]
    candidates = [(i,float(t),values[:,i] < t) for i, ts in enumerate(thresholds) for t in ts]
    initial = Program(label=int(np.bincount(labels, minlength=4).argmax()))
    def score(p):
        return float(np.mean(p.predict(features)==labels)) - penalty*p.complexity
    beam, best = [initial], initial
    history = [{"splits": 0, "score": score(initial), "fidelity": float(np.mean(initial.predict(features)==labels)),
                "complexity": 1}]
    for depth in range(max_splits):
        expanded = []
        for p in beam:
            pred = p.predict(features)
            correct = int(np.sum(pred == labels))
            for path, leaf in p.leaves():
                mask = p.mask(features, path)
                ids = np.flatnonzero(mask)
                if len(ids) < 2*min_leaf: continue
                old_correct = int(np.sum(labels[ids] == leaf.label))
                local = []
                for i,t,condition in candidates:
                    left_ids, right_ids = ids[condition[ids]], ids[~condition[ids]]
                    if min(len(left_ids),len(right_ids)) < min_leaf: continue
                    lc, rc = np.bincount(labels[left_ids],minlength=4), np.bincount(labels[right_ids],minlength=4)
                    ll, rl = int(lc.argmax()), int(rc.argmax())
                    if ll == rl: continue
                    complexity = p.complexity + 3 + expressions[i].complexity
                    new_score = (correct-old_correct+int(lc[ll])+int(rc[rl]))/len(labels) - penalty*complexity
                    local.append((new_score,i,t,ll,rl))
                local.sort(key=lambda v: (-v[0],v[1],v[2],v[3],v[4]))
                for s,i,t,ll,rl in local[:beam_width]:
                    q = p.replace(path,Program(expr=expressions[i],threshold=t,left=Program(label=ll),right=Program(label=rl)))
                    expanded.append((s,q))
        if not expanded: break
        seen, unique = set(), []
        for s,p in sorted(expanded,key=lambda v: (-v[0],v[1].complexity,v[1].text())):
            key = json.dumps(p.to_dict(),sort_keys=True)
            if key not in seen: unique.append(p); seen.add(key)
            if len(unique) == beam_width: break
        beam = unique
        if score(beam[0]) > score(best) + 1e-12: best = beam[0]
        history.append({"splits":depth+1,"score":score(beam[0]),
                        "fidelity":float(np.mean(beam[0].predict(features)==labels)),
                        "complexity":beam[0].complexity})
    return best, history
