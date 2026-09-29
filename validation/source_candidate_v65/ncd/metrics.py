"""Metrics report fidelity, truth, uncertainty and errors separately."""
import numpy as np
from .worlds import LABELS

def proportion_interval(successes, n):
    if n == 0: return None
    z = 1.959963984540054
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0,center-radius)),float(min(1,center+radius))]

def evaluate(truth, probabilities, program):
    truth, probabilities, program = np.asarray(truth),np.asarray(probabilities),np.asarray(program)
    neural = probabilities.argmax(1)
    n = len(truth)
    agree = program==neural
    correct_n, correct_p = neural==truth, program==truth
    confusion = np.zeros((4,4),dtype=int)
    np.add.at(confusion,(truth,program),1)
    brier = np.mean(np.sum((probabilities-np.eye(4)[truth])**2,axis=1))
    return {"n":n, "neural_accuracy":float(correct_n.mean()), "program_accuracy":float(correct_p.mean()),
            "fidelity":float(agree.mean()), "fidelity_wilson95":proportion_interval(int(agree.sum()),n),
            "neural_multiclass_brier":float(brier),
            "joint_correct":int(np.sum(correct_n&correct_p)),
            "joint_same_wrong":int(np.sum(agree&~correct_n)),
            "neural_only_correct":int(np.sum(correct_n&~correct_p)),
            "program_only_correct":int(np.sum(~correct_n&correct_p)),
            "both_wrong_different":int(np.sum(~agree&~correct_n&~correct_p)),
            "program_confusion":confusion.tolist(),
            "per_class":{name:{"n":int(np.sum(truth==i)),
                "program_accuracy":float(np.mean(correct_p[truth==i])) if np.any(truth==i) else None,
                "neural_accuracy":float(np.mean(correct_n[truth==i])) if np.any(truth==i) else None,
                "fidelity":float(np.mean(agree[truth==i])) if np.any(truth==i) else None}
                for i,name in enumerate(LABELS)},
            "program_directional_coverage":float(np.mean(program<2)),
            "program_directional_accuracy":float(np.mean(correct_p[program<2])) if np.any(program<2) else None}
