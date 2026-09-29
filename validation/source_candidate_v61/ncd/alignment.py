"""Held-out interchange interventions on a frozen neural representation.

This is an empirical test of one linear scalar abstraction. It neither assumes
single-neuron semantics nor guarantees that patched states lie on the data manifold.
"""
import numpy as np
import torch
from .statistics import FEATURES

def hidden(model, data):
    with torch.no_grad():
        a,b = model.hidden_pair(torch.as_tensor(data,dtype=torch.float32))
    return torch.cat([a,b],dim=1).numpy().astype(float)

def logits_from_hidden(model, values):
    width = values.shape[1]//2
    with torch.no_grad():
        return model.from_hidden(torch.as_tensor(values[:,:width],dtype=torch.float32),
                                 torch.as_tensor(values[:,width:],dtype=torch.float32)).softmax(-1).numpy()

def ridge(x, y):
    return np.linalg.solve(x.T@x + np.eye(x.shape[1])*5., x.T@y)

def r2(y, pred):
    return float(1-np.sum((y-pred)**2)/max(float(np.sum((y-y.mean())**2)),1e-12))

def interchange(z, beta, target):
    norm = float(beta@beta)
    if norm < 1e-16: raise ValueError("Degenerate alignment direction")
    return z + ((target-z@beta)/norm)[:,None]*beta[None,:]

def fit_and_test(model, program, fit_data, fit_features, test_data, test_features, seed=42):
    used = sorted(program.used_features())
    if not used:
        return {"status":"not_applicable_constant_program","interpretation":"no scalar rule variable exists"}, None
    h = hidden(model,fit_data)
    cut = max(16,int(.7*len(h)))
    if cut >= len(h): raise ValueError("Alignment fit requires enough validation worlds")
    hm, hs = h[:cut].mean(0), h[:cut].std(0).clip(.01)
    z = (h-hm)/hs
    choices = []
    for feature in used:
        y = fit_features[:,FEATURES.index(feature)]
        ym, ys = float(y[:cut].mean()), max(float(y[:cut].std()),1e-6)
        target = (y-ym)/ys
        beta = ridge(z[:cut],target[:cut])
        choices.append((r2(target[cut:],z[cut:]@beta),feature))
    selection_r2, feature = max(choices)
    idx = FEATURES.index(feature)
    # Mapping selection only uses alignment_fit; refit same scalar on all fit worlds.
    hm, hs = h.mean(0),h.std(0).clip(.01)
    z = (h-hm)/hs
    y = fit_features[:,idx]
    ym,ys = float(y.mean()),max(float(y.std()),1e-6)
    target = (y-ym)/ys
    beta = ridge(z,target)
    rng = np.random.default_rng(seed)
    permuted = ridge(z,rng.permutation(target))
    random_dir = rng.normal(size=z.shape[1])
    random_dir /= np.linalg.norm(random_dir)
    slope = float((z@random_dir)@target/max(float(np.sum((z@random_dir)**2)),1e-12))
    random_beta = random_dir * (slope if abs(slope)>1e-5 else 1e-5)
    ht = hidden(model,test_data)
    zt = (ht-hm)/hs
    # Non-self cyclic source assignment; frozen before outcomes are inspected.
    source = np.roll(np.arange(len(ht)),1)
    targets = (test_features[source,idx]-ym)/ys
    symbolic_features = test_features.copy()
    symbolic_features[:,idx] = test_features[source,idx]
    p0, p1 = program.predict(test_features),program.predict(symbolic_features)
    n0_probs = logits_from_hidden(model,ht)
    n0 = n0_probs.argmax(1)
    matched = n0==p0
    outcomes, mappings = {}, {}
    for name, vector in (("learned",beta),("random_direction",random_beta),("permuted_target",permuted)):
        patched = interchange(zt,vector,targets)
        probs = logits_from_hidden(model,patched*hs+hm)
        n1 = probs.argmax(1)
        changed = p1!=p0
        informative = matched&changed
        outcomes[name] = {
            "test_probe_r2":r2((test_features[:,idx]-ym)/ys,zt@vector),
            "interchange_accuracy":float(np.mean(n1==p1)),
            "baseline_matched_n":int(matched.sum()),
            "interchange_accuracy_on_baseline_matched":float(np.mean(n1[matched]==p1[matched])) if matched.any() else None,
            "symbolic_changed_n":int(changed.sum()),
            "informative_n":int(informative.sum()),
            "accuracy_on_matched_symbolic_changes":float(np.mean(n1[informative]==p1[informative])) if informative.any() else None,
            "neural_changed_n":int(np.sum(n1!=n0)),
            "mean_hidden_displacement":float(np.linalg.norm((patched-zt)*hs,axis=1).mean()),
            "normalized_decoder_max_error":float(np.max(np.abs(patched@vector-targets))),
            "predictions":n1.tolist()}
        mappings[name] = vector
    result = {"status":"measured","feature":feature,"selection_validation_r2":selection_r2,
              "n":len(ht),"baseline_fidelity":float(matched.mean()),
              "source_indices":source.tolist(),"symbolic_before":p0.tolist(),"symbolic_after":p1.tolist(),
              "controls":outcomes,
              "interpretation":"Empirical scalar-alignment test; not proof of unique internal algorithm.",
              "off_manifold_patch_possible":True,"alignment_used_in_synthesis":False}
    state = {"hidden_mean":hm,"hidden_std":hs,"target_mean":np.array(ym),"target_std":np.array(ys),
             "feature_index":np.array(idx),"source_indices":source,**mappings}
    return result,state
