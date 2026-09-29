"""Separate fidelity refinement from causal-error discovery."""
import numpy as np
from .synthesis import synthesize

def refine(program, x, teacher, pool_x, pool_teacher, *, rounds=2, per_round=64, **search):
    selected, history = [], []
    current = program
    for iteration in range(rounds):
        errors = np.flatnonzero(current.predict(pool_x) != pool_teacher)
        fresh = np.array([i for i in errors if int(i) not in selected], dtype=int)
        if len(fresh) == 0: break
        # Deterministic class-balanced selection, independent of causal truth.
        chosen = []
        for cls in range(4):
            chosen.extend(fresh[pool_teacher[fresh] == cls][:max(1,per_round//4)].tolist())
        chosen.extend(int(i) for i in fresh if int(i) not in chosen)
        chosen = chosen[:per_round]
        selected.extend(chosen)
        current, _ = synthesize(np.concatenate([x,pool_x[selected]]),
                                np.concatenate([teacher,pool_teacher[selected]]),**search)
        history.append({"round":iteration+1, "added":len(chosen), "total_added":len(selected),
                        "pool_fidelity":float(np.mean(current.predict(pool_x)==pool_teacher)),
                        "training_fidelity":float(np.mean(current.predict(x)==teacher))})
    return current, {"history":history, "refinement_indices":selected,
                     "supervision":"frozen_teacher_only", "pool_is_not_final_test":True}

def compare_search(program, features, truth, query, *, budget=64, seed=42):
    """Both strategies see one independent SCM proposal pool.

    Program-guided selection can inspect SCM truth and cheap symbolic outputs;
    it cannot inspect teacher output until selected. Teacher calls have equal budget.
    """
    if budget < 1: raise ValueError("Positive query budget required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    p = program.predict(features)
    priority = np.concatenate([order[p[order]!=truth[order]],order[p[order]==truth[order]]])
    count = min(budget,len(features))
    guided, random_ids = priority[:count], order[:count]
    outcomes = {}
    for name,ids in (("program_guided",guided),("random",random_ids)):
        nn = np.asarray(query(ids)).argmax(1)
        joint = (p[ids]==nn)&(nn!=truth[ids])
        outcomes[name] = {"indices":ids.tolist(), "teacher_predictions":nn.tolist(),
                         "teacher_errors":int(np.sum(nn!=truth[ids])),
                         "joint_errors":int(joint.sum()), "queries":count,
                         "joint_error_rate":float(joint.mean()),
                         "fidelity_errors":int(np.sum(p[ids]!=nn))}
    return {"strategies":outcomes,"proposal_count":len(features),"seed":seed,
            "interpretation":"shared wrong predictions are behavioral evidence, not unique mechanism proof",
            "gain_in_joint_error_rate":outcomes["program_guided"]["joint_error_rate"]-outcomes["random"]["joint_error_rate"]}
