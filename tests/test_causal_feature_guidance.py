import numpy as np
import pytest

from ncd.causal_feature_guidance import feature_support
from ncd.causal_guided_experiment import _used_features
from ncd.cdir import Node
from ncd.raw_program_trace import RawDiscoveryExecutor,feature_root_addresses
from ncd.rules import Rule,fit_rule
from ncd.statistics import FEATURES


def test_feature_roots_match_discovery_feature_matrix():
    executor=RawDiscoveryExecutor(Rule(tuple(FEATURES),{"label":0}))
    data=np.random.default_rng(1392).normal(size=(20,24,2))
    trace=executor.execute(data);addresses=feature_root_addresses(executor)
    assert len(addresses)==len(FEATURES)==14
    np.testing.assert_allclose(np.column_stack([trace.records[a]["value"] for a in addresses]),trace.features)
    assert all(trace.records[a]["kind"]=="vector" for a in addresses)


def _audit(target,baseline,collateral,natural=0.2):
    return {"per_node":[{"natural_nmse":natural,
        "targeted":{"n":10,"nmse":target,"no_intervention_nmse":baseline},
        "collateral":{"n":10,"nmse":collateral,"no_intervention_nmse":1.}}]}


def test_feature_support_requires_gain_over_baseline_and_controls():
    good=feature_support(_audit(.2,1.,.1),_audit(.8,1.,.1),_audit(.9,1.,.1))
    assert 0<good[0]<=1
    assert feature_support(_audit(1.1,1.,.1),_audit(2.,1.,.1),_audit(2.,1.,.1))[0]==0
    assert feature_support(_audit(.5,1.,.1),_audit(.4,1.,.1),_audit(.9,1.,.1))[0]==0


def test_rule_feature_usage_and_alignment_validation():
    expr=Node("add",(Node("var",index=2),Node("abs",(Node("var",index=5),))))
    rule=Rule(tuple(FEATURES),{"expr":expr.to_dict(),"threshold":0.,"left":{"label":0},"right":{"label":1}})
    assert _used_features(rule)==[2,5]
    x=np.random.default_rng(3).normal(size=(40,len(FEATURES)));y=np.arange(40)%4
    with pytest.raises(ValueError):fit_rule(x,y,FEATURES,feature_alignment=np.ones(3),alignment_weight=.1)
    with pytest.raises(ValueError):fit_rule(x,y,FEATURES,feature_alignment=np.r_[np.ones(13),2.],alignment_weight=.1)

def test_conservative_selection_never_uses_support_to_cover_base_loss():
    from ncd.causal_guided_experiment import CausalGuidedConfig,_select
    base_rule=Rule(tuple(FEATURES),{"expr":Node("var",index=0).to_dict(),"threshold":0.,
        "left":{"label":0},"right":{"label":1}})
    guided_rule=Rule(tuple(FEATURES),{"expr":Node("var",index=1).to_dict(),"threshold":0.,
        "left":{"label":0},"right":{"label":1}})
    def candidates():
        return {"a":{"id":"a","rule":base_rule,"origins":[{"alignment_weight":0.}],"validation_fidelity":.8},
            "b":{"id":"b","rule":guided_rule,"origins":[{"alignment_weight":.01}],"validation_fidelity":.79}}
    support=np.array([0.,1.,*([0.]*(len(FEATURES)-2))])
    c=CausalGuidedConfig.quick();c.support_weight=.2
    plain,guided,_=_select(candidates(),support,c)
    assert plain["id"]=="a" and guided["id"]=="b"
    c.selection_policy="conservative"
    plain,guided,records=_select(candidates(),support,c)
    assert plain["id"]==guided["id"]=="a"
    assert next(r for r in records if r["id"]=="b")["validation_noninferior"] is False

def test_ood_selection_requires_noninferiority_in_every_guard():
    from ncd.causal_guided_experiment import CausalGuidedConfig,_select,TESTS
    a=Rule(tuple(FEATURES),{"expr":Node("var",index=0).to_dict(),"threshold":0.,"left":{"label":0},"right":{"label":1}})
    b=Rule(tuple(FEATURES),{"expr":Node("var",index=1).to_dict(),"threshold":0.,"left":{"label":0},"right":{"label":1}})
    candidates={"a":{"id":"a","rule":a,"origins":[{"alignment_weight":0.}],"validation_fidelity":.8,
        "guard_fidelity":{t:.7 for t in TESTS}},
        "b":{"id":"b","rule":b,"origins":[{"alignment_weight":.01}],"validation_fidelity":.81,
        "guard_fidelity":{t:(.69 if t=="test_noise" else .72) for t in TESTS}}}
    c=CausalGuidedConfig.quick();c.selection_policy="ood_conservative";c.guard_worlds=64;c.support_weight=.2
    plain,guided,records=_select(candidates,np.r_[0.,1.,np.zeros(len(FEATURES)-2)],c)
    assert plain["id"]==guided["id"]=="a"
    rejected=next(r for r in records if r["id"]=="b")
    assert rejected["validation_noninferior"] and not rejected["guard_noninferior"]["test_noise"]
