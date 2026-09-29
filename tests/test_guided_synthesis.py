import pytest
from ncd.guided_synthesis import GuidedConfig,mechanism_support,select_records,_check_worlds
from ncd.worlds import generate_worlds

def metric(error,baseline,n=20):
    return {"n":n,"nmse":error,"no_intervention_nmse":baseline}

def numeric():
    return {"status":"measured","per_node":[{"natural_n":50,"natural_nmse":.1,
        "targeted":metric(.1,2.),"collateral":metric(.1,.1)}]}

def test_support_requires_valid_changes_and_numerical_improvement():
    behavioral={"overall":{"informative_pairs":20,"informative_accuracy":.8}}
    assert 0<mechanism_support(numeric(),behavioral)<1
    assert mechanism_support(numeric(),{"overall":{"informative_pairs":1,"informative_accuracy":1.}})==0
    bad=numeric();bad["per_node"][0]["targeted"]=metric(3.,2.)
    assert mechanism_support(bad,behavioral)==0
    assert mechanism_support({"status":"no_executed_pairs"},behavioral)==0

def test_internal_evidence_changes_program_selection_only_when_enabled():
    records=[{"id":"behavior","complexity":10,"validation_fidelity":.81,"mechanism_support":0.},
             {"id":"mechanism","complexity":10,"validation_fidelity":.80,"mechanism_support":.5}]
    selected=select_records(records,GuidedConfig())
    assert selected["without_internal"]=="behavior"
    assert selected["with_internal"]=="mechanism"
    disabled=select_records(records,GuidedConfig(mechanism_weight=0))
    assert disabled["with_internal"]==disabled["without_internal"]=="behavior"

def test_world_roles_reject_final_tests_and_identity_overlap():
    extraction=generate_worlds("extraction",8,391,32)
    fit=generate_worlds("alignment_fit",8,391,32)
    validation=generate_worlds("alignment_fit",8,10391,32)
    _check_worlds(extraction,fit,validation)
    with pytest.raises(ValueError):_check_worlds(extraction,fit,fit)
    with pytest.raises(ValueError):_check_worlds(extraction,fit,generate_worlds("test_id",8,391,32))
