from pathlib import Path
p=Path('ncd/rules.py');s=p.read_text(encoding='utf-8')
s=s.replace('feature_alignment=None,alignment_weight=.0):','feature_alignment=None,alignment_weight=.0,sample_weight=None):')
s=s.replace('    expressions=candidate_expressions(names,arithmetic)','''    # Weights alter empirical fidelity, including expression pruning and leaf votes.
    # Threshold quantiles and min_leaf retain their original sample-count semantics.
    weights=np.ones(len(x),dtype=float) if sample_weight is None else np.asarray(sample_weight,dtype=float)
    if weights.shape!=(len(x),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("sample_weight must contain one finite positive value per row")
    total=float(weights.sum())
    if not np.isfinite(total):raise ValueError("Non-finite total sample weight")
    def counts(ids):return np.bincount(labels[ids],weights=weights[ids],minlength=4)
    def fidelity(pred):return float(weights[pred==labels].sum()/total)
    expressions=candidate_expressions(names,arithmetic)''')
s=s.replace('lc=np.bincount(labels[left],minlength=4);rc=np.bincount(labels[~left],minlength=4)','lc=counts(left);rc=counts(~left)')
s=s.replace('int(lc.max()+rc.max())','float(lc.max()+rc.max())').replace('best_correct/len(x)','best_correct/total')
s=s.replace('np.bincount(labels,minlength=4).argmax()','counts(slice(None)).argmax()')
s=s.replace('float(np.mean(p.predict(x)==labels))-penalty','fidelity(p.predict(x))-penalty')
s=s.replace('            base=Rule(tuple(names),tree)\n            correct=np.sum(base.predict(x)==labels)\n','')
s=s.replace('old=np.sum(labels[ids]==leaf["label"])','old=weights[ids][labels[ids]==leaf["label"]].sum()')
s=s.replace('np.bincount(labels[l],minlength=4),np.bincount(labels[r],minlength=4)','counts(l),counts(r)')
s=s.replace('(int(lc[ll])+int(rc[rl])-old)/len(x)','(float(lc[ll])+float(rc[rl])-old)/total')
s=s.replace('    return Rule(tuple(names),best),trace','''        if sample_weight is not None:
            trace[-1]["weighted_fidelity"]=fidelity(Rule(tuple(names),beam[0]).predict(x))
    return Rule(tuple(names),best),trace''')
p.write_text(s,encoding='utf-8')
