from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
import numpy as np
import pytest
from proof_workbench.polynomial_semantics import (canonical_polynomial,prove_polynomial_equivalence,
    verify_polynomial_equivalence,certify_decimal_conversion_boundary,verify_decimal_conversion_boundary)
from proof_workbench.bounded_mdl import certify_polynomial_mdl,verify_polynomial_mdl


def test_binary_constant_semantics_reject_unsound_decimal_equivalence():
    certificate=certify_decimal_conversion_boundary()
    assert certificate['legacy_equivalence_result']
    assert certificate['actual_equivalence_certificate']['left_normal_form']==[{'powers':[1],'coefficient':'32'}]
    assert verify_decimal_conversion_boundary(certificate)['status']=='verified'
    bad=deepcopy(certificate);bad['strict_witness']['left_mathematical_value']='0'
    with pytest.raises(ValueError):verify_decimal_conversion_boundary(bad)


def test_exact_polynomial_equivalence_is_global_and_excludes_protected_division():
    a={'op':'square','args':[{'op':'add','args':[{'op':'var','index':0},{'op':'var','index':1}]}]}
    b={'op':'mul','args':[{'op':'add','args':[{'op':'var','index':0},{'op':'var','index':1}]}]*2}
    c=prove_polynomial_equivalence(a,b,2)
    assert verify_polynomial_equivalence(c)['conclusion']=='proved'
    assert c['left_normal_form']==[{'powers':[0,2],'coefficient':'1'},{'powers':[1,1],'coefficient':'2'},{'powers':[2,0],'coefficient':'1'}]
    with pytest.raises(ValueError):canonical_polynomial({'op':'div','args':[{'op':'var','index':0},{'op':'var','index':1}]},2)
    with pytest.raises(ValueError):canonical_polynomial({'op':'var','index':2},2)
    bad=deepcopy(c);bad['left_normal_form'][0]['coefficient']='0'
    with pytest.raises(ValueError):verify_polynomial_equivalence(bad)


def test_finite_mdl_exhaustion_and_syntax_uniqueness_are_separate():
    grammar={'variables':2,'constants':['0'],'operations':['add','sub','mul','square'],'max_cost':3}
    target={'op':'add','args':[{'op':'var','index':0},{'op':'var','index':1}]}
    c=certify_polynomial_mdl(target,grammar)
    assert verify_polynomial_mdl(c)['minimum_cost']==3
    assert c['enumerated_programs']==36 and len(c['minimum_programs'])==2 and not c['minimum_syntax_unique']
    bad=deepcopy(c);bad['enumerated_programs']-=1
    with pytest.raises(ValueError):verify_polynomial_mdl(bad)
    bad=deepcopy(c);bad['minimum_cost']=1
    with pytest.raises(ValueError):verify_polynomial_mdl(bad)


def test_incomplete_mdl_enumeration_cannot_claim_minimality():
    grammar={'variables':2,'constants':['0'],'operations':['add','sub','mul','square'],'max_cost':5}
    target={'op':'add','args':[{'op':'var','index':0},{'op':'var','index':1}]}
    c=certify_polynomial_mdl(target,grammar,max_candidates=35)
    assert c['status']=='unresolved' and c['current_upper_bound']==3 and c['minimum_cost'] is None
    assert verify_polynomial_mdl(c)['conclusion']=='unresolved'
    no=certify_polynomial_mdl(target,dict(grammar,max_cost=1))
    assert no['status']=='refuted' and verify_polynomial_mdl(no)['status']=='verified'
    with pytest.raises(ValueError):certify_polynomial_mdl(target,dict(grammar,constants=['1/10']))


def test_historical_mean_obstruction_matches_cut_symmetry_and_gaussian_bound():
    from proof_workbench.mean_information_boundary import certify_mean_information,verify_mean_information
    from ncd.model import load_model,set_seed
    from ncd.neural_sites import SiteDecoder,SITES
    from scipy.special import erf
    c=certify_mean_information();assert verify_mean_information(c)['conclusion']=='refuted'
    set_seed(8100);teacher=load_model(Path(c['run'])/'teacher.pt')
    data=np.array([[[float(Q(x)) for x in row] for row in c[key]] for key in ('data_a','data_b')])
    for site in SITES:
        hidden=SiteDecoder(teacher,site).extract(data)
        assert np.max(abs(hidden[0]-hidden[1]))<1e-5
    expected=erf(float(Q(c['epsilon']))*np.sqrt(c['samples']*float(Q(c['target']['fit_variance']))/2))
    assert abs(float(Q(c['statistical_success_probability_upper']))-expected)<1e-14
    assert float(Q(c['statistical_success_probability_upper']))<.043
    bad=deepcopy(c);bad['target']['fit_variance']='1'
    with pytest.raises(ValueError):verify_mean_information(bad)


def test_actual_root_has_distinct_shortest_approximately_faithful_functions():
    from proof_workbench.bounded_mdl import certify_approximate_root_nonuniqueness,verify_approximate_root_nonuniqueness
    checkpoint='runs/active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline/mechanism_1.pt'
    c=certify_approximate_root_nonuniqueness(checkpoint)
    assert verify_approximate_root_nonuniqueness(c)['minimum_cost']==1
    assert Q(c['distinct_functions_witness']['output_difference'])>0
    assert all(row['proof']['status']=='proved' for row in c['candidates'])
    bad=deepcopy(c);bad['candidates'][0]['expression']['value']=0
    with pytest.raises(ValueError):verify_approximate_root_nonuniqueness(bad)
    assert not c['exact_minimal_realization_refuted']
