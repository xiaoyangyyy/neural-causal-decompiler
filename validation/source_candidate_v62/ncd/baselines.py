"""Same-feature baselines make the DSL prior visible."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .dsl import Expr, Program
from .statistics import FEATURES

def decision_tree(features,teacher,seed=42,max_leaves=5):
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaves,min_samples_leaf=8,random_state=seed).fit(features,teacher)
    def convert(i):
        t = tree.tree_
        if t.children_left[i] == t.children_right[i]:
            return Program(label=int(tree.classes_[t.value[i,0].argmax()]))
        return Program(expr=Expr("feature",feature=FEATURES[t.feature[i]]),
                       threshold=float(t.threshold[i]),
                       left=convert(t.children_left[i]),right=convert(t.children_right[i]))
    return convert(0)

def anm_predict(features, independence_threshold=.08, ambiguity_threshold=.02):
    dep = features[:,FEATURES.index("dep_xy")]
    diff = features[:,FEATURES.index("resdep_xy")]-features[:,FEATURES.index("resdep_yx")]
    return np.where(dep<independence_threshold,2,np.where(np.abs(diff)<ambiguity_threshold,3,np.where(diff<0,0,1)))

def calibrate_anm(features,truth):
    choices = []
    for dep in (.03,.05,.08,.12,.18):
        for ambiguity in (.005,.01,.02,.04,.08):
            pred = anm_predict(features,dep,ambiguity)
            choices.append((float(np.mean(pred==truth)),dep,ambiguity))
    accuracy,dep,ambiguity = max(choices)
    return {"independence_threshold":dep,"ambiguity_threshold":ambiguity,"dev_accuracy":accuracy,
            "selection":"fixed grid on dev truth","not_an_identifiability_oracle":True}
