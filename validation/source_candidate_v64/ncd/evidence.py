"""Finite-sample fidelity bounds and explicitly empirical equivalence classes."""
import hashlib
import numpy as np

def fidelity_bound(successes,delta=.05):
    """Hoeffding lower bound for independent bounded success units.

    Caller must supply one independent world (or disjoint world-pair) per entry.
    Repeated source/target interventions are not independent entries.
    """
    s=np.asarray(successes,dtype=float)
    if s.ndim!=1 or len(s)==0 or not np.isfinite(s).all() or np.any((s<0)|(s>1)) or not 0<delta<1:
        raise ValueError("Expected independent successes in [0,1] and delta in (0,1)")
    mean=float(s.mean());radius=float(np.sqrt(np.log(1/delta)/(2*len(s))))
    return {"units":len(s),"empirical_fidelity":mean,"confidence":1-delta,
            "lower_bound":max(0.,mean-radius),"epsilon_upper":min(1.,1-mean+radius),
            "assumptions":"fixed hypothesis; independent bounded evaluation units",
            "not_a_universal_equivalence_proof":True}

def empirical_equivalence(programs,features):
    """Group by behavior only on these probes; never equate this with proof."""
    groups={}
    for name,program in programs.items():
        predictions=program.predict(features).astype(np.int64)
        key=hashlib.sha256(predictions.tobytes()).hexdigest()
        groups.setdefault(key,[]).append(name)
    return {"classes":list(groups.values()),"scope":"provided probes only","proof":False}
