from fractions import Fraction as Q
from copy import deepcopy
import numpy as np
import pytest
from ncd.proof_intervals import Interval
from proof_extensions.statistical_frontend import (pi_interval,erf_point,raw_feature,residual_statistics,partial_correlation,fisher_p)
from proof_extensions.frozen_graph import export_graph,graph_logits,layer_norm,stable_attention,certify_graph_box,verify_graph_box


@pytest.mark.parametrize('x',['0','1/100','3/2','799/100','8','-3/2','12'])
def test_gaussian_transcendentals_against_independent_high_precision(x):
    import mpmath as mp
    with mp.workdps(100):
        def exact(q):return mp.mpf(q.numerator)/q.denominator
        p=pi_interval();assert exact(p.lo)<=mp.pi<=exact(p.hi)
        v=erf_point(Q(x));target=mp.erf(exact(Q(x)))
        assert exact(v.lo)<=target<=exact(v.hi)


def test_actual_statistics_floor_differs_from_cdir_protection():
    from ncd.statistics import extract_one
    from ncd.discovery_fidelity_proof import raw_feature as old
    data=np.array([[(i-7.5)*1e-10,(i%3-1)*1e-10] for i in range(16)])
    domain=[[Interval.point(float(v)) for v in row] for row in data]
    actual=raw_feature(domain,0);incorrect=old(domain,0)
    assert abs(float(actual.lo)-extract_one(data)[0])<1e-16
    assert abs(float(incorrect.lo)-float(actual.lo))>1e-6


def test_actual_residual_uses_rbf_quantile_basis_and_response_standardization():
    from ncd.statistics import residual
    from ncd.cdir import crossfit_prediction
    x=np.linspace(-2,2,16);y=np.sin(3*x)+np.cos(7*x)/5
    computed=residual_statistics([Interval.point(float(v)) for v in y],[Interval.point(float(v)) for v in x])
    actual=residual(y,x)
    assert np.max(np.abs(np.array([float((v.lo+v.hi)/2) for v in computed])-actual))<1e-8
    assert np.max(np.abs(actual-(y-crossfit_prediction(y,x))))>.01


def test_sort_and_regression_rank_boundaries_do_not_become_success():
    from proof_extensions.statistical_frontend import certified_order
    with pytest.raises(ValueError):certified_order([[Interval('0','1'),Interval('0','1')]])
    data=[[Interval.point(0),Interval.point(i),Interval.point(0)] for i in range(16)]
    with pytest.raises(ValueError):partial_correlation(data,0,1,(2,))


def test_gaussian_ci_uses_correct_conditional_regression_and_tail():
    from ncd.graphs import partial_correlation as actual_corr,fisher_p as actual_p
    rng=np.random.default_rng(8100);data=rng.normal(size=(16,3));data[:,1]+=.4*data[:,2]
    point=[[Interval.point(float(v)) for v in row] for row in data]
    r=partial_correlation(point,0,1,(2,));p=fisher_p(point,0,1,(2,))
    assert abs(float((r.lo+r.hi)/2)-actual_corr(data,0,1,(2,)))<1e-12
    assert abs(float((p.lo+p.hi)/2)-actual_p(data,0,1,(2,)))<1e-12


def test_attention_and_normalization_keep_exact_structural_bounds():
    q=[Interval('-1000','1000')];keys=[[Interval.point(10)],[Interval.point(-10)]]
    values=[[Interval.point(7)],[Interval.point(7)]]
    assert stable_attention(q,keys,values)==[Interval.point(7)]
    ln=layer_norm([Interval('-100','100')]*4,['1']*4,['0']*4,'1/100000')
    assert all(v.abs().hi<=Interval.point(3).sqrt().hi for v in ln)


def graph_fixture(kind,path):
    import torch
    from ncd.graph_model import GraphDiscoverer
    from ncd.node_context_graph import NodeContextGraphDiscoverer
    from ncd.active_intervention_graph import ActiveFactorizedGraphDiscoverer
    torch.manual_seed(8100);torch.set_num_threads(2)
    cls={'plain':GraphDiscoverer,'context':NodeContextGraphDiscoverer,'active':ActiveFactorizedGraphDiscoverer}[kind]
    model=cls(4);model.eval()
    saved={'width':4,'state_dict':model.state_dict()}
    if kind=='context':saved['architecture']='node_context_v1'
    if kind=='active':saved['architecture']='active_input_intervention_v1'
    torch.save(saved,path)
    return model


@pytest.mark.parametrize('kind',['plain','context','active'])
def test_full_graph_backend_matches_actual_independent_neural_forward(kind,tmp_path):
    import torch
    checkpoint=tmp_path/'model.pt';model=graph_fixture(kind,checkpoint);network=export_graph(checkpoint)
    count=len(network['mean']);data=np.random.default_rng(8100).normal(scale=.2,size=(3,3,count))
    points=[[[Interval.point(float(v)) for v in token] for token in row] for row in data]
    computed=graph_logits(network,points)
    expected=model.double()(torch.tensor(data[None],dtype=torch.float64)).detach().numpy()[0]
    assert np.max(np.abs(np.array([[[float((v.lo+v.hi)/2) for v in token] for token in row] for row in computed])-expected))<1e-11


def test_graph_full_box_certificate_and_tampered_label(tmp_path):
    import torch
    checkpoint=tmp_path/'model.pt';model=graph_fixture('active',checkpoint)
    with torch.no_grad():model.skeleton_head.weight.zero_();model.skeleton_head.bias.fill_(-10)
    torch.save({'width':4,'state_dict':model.state_dict(),'architecture':'active_input_intervention_v1'},checkpoint)
    network=export_graph(checkpoint);domain=[[[['-1','1'] for _ in network['mean']] for _ in range(3)] for _ in range(3)]
    c=certify_graph_box(network,domain,[[0]*3 for _ in range(3)])
    assert verify_graph_box(network,c)['conclusion']=='proved'
    changed=deepcopy(c);changed['program_labels'][0][1]=1
    with pytest.raises(ValueError):verify_graph_box(network,changed)
    wrong=[[0]*3 for _ in range(3)];wrong[0][1]=1
    assert certify_graph_box(network,domain,wrong)['status']=='refuted'



def test_population_distinctness_does_not_remove_finite_sample_separation_requirement():
    from proof_extensions.finite_sample_boundary import certify_finite_sample_boundary,verify_finite_sample_boundary
    c=certify_finite_sample_boundary()
    assert c['population_laws_distinct'] and c['status']=='refuted'
    assert Q(c['minimax_graph_error_lower'])>Q(47,100)
    assert verify_finite_sample_boundary(c)['conclusion']=='refuted'
    changed=deepcopy(c);changed['single_row_kl_b_to_a']='0'
    with pytest.raises(ValueError):verify_finite_sample_boundary(changed)
    assert certify_finite_sample_boundary(samples=100000,coefficient='1')['status']=='unresolved'


@pytest.mark.parametrize('case',['label_tie','confidence_cycle'])
def test_legacy_graph_decoder_equivariance_is_strictly_refuted(case):
    from proof_extensions.graph_candidates import certify_decoder_boundary,verify_decoder_boundary
    c=certify_decoder_boundary(case)
    assert verify_decoder_boundary(c)['conclusion']=='refuted'
    assert c['decoded_permuted']!=c['required_equivariant_output']
    changed=deepcopy(c);changed['decoded_permuted']=changed['required_equivariant_output']
    with pytest.raises(ValueError):verify_decoder_boundary(changed)


def test_graph_candidate_family_preserves_ties_and_detects_forced_cycle():
    from proof_extensions.graph_candidates import graph_family,family_contains,verify_graph_family,certify_decoder_boundary
    c=certify_decoder_boundary('label_tie');scores=[[[Q(x) for x in token] for token in row] for row in c['probabilities']]
    family=graph_family(scores);a=[[0,1,0],[0,0,0],[0,0,0]];b=[[0,0,0],[1,0,0],[0,0,0]]
    assert family_contains(family,a) and family_contains(family,b)
    assert verify_graph_family(scores,family)['true_causal_graph_coverage_proved'] is False
    c=certify_decoder_boundary('confidence_cycle');scores=[[[Q(x) for x in token] for token in row] for row in c['probabilities']]
    family=graph_family(scores)
    assert family['forced_cycle_conflict'] and family['nonempty_witness'] is None
    assert not family_contains(family,[[0]*3 for _ in range(3)])
    forged=deepcopy(family);forged['pairs'][0]['allowed_dag_states']=[0,1,2]
    with pytest.raises(ValueError):verify_graph_family(scores,forged)


def test_candidate_family_membership_commutes_with_node_relabeling():
    from itertools import permutations
    from proof_extensions.graph_candidates import graph_family,family_contains,certify_decoder_boundary
    c=certify_decoder_boundary('label_tie');scores=[[[Q(x) for x in token] for token in row] for row in c['probabilities']]
    a=np.array([[0,1,0],[0,0,0],[0,0,0]],int);b=a.T;empty=np.zeros((3,3),int)
    for p in permutations(range(3)):
        moved=[ [scores[i][j] for j in p] for i in p];family=graph_family(moved)
        assert family_contains(family,a[np.ix_(p,p)]) and family_contains(family,b[np.ix_(p,p)])
        assert not family_contains(family,empty)


def test_feature_collision_preserves_all_actual_statistics_but_not_raw_data():
    from proof_extensions.feature_collision import template,equality_witness
    from ncd.statistics import extract_one
    _,rows=template();a=np.array(rows,float);b=a*[-1,1]
    witness=equality_witness()
    assert witness['feature_count']==14 and not witness['datasets_are_row_permutations']
    assert np.max(abs(extract_one(a)-extract_one(b)))<1e-12
    assert np.mean(a[:,0]*a[:,1]**4)!=np.mean(b[:,0]*b[:,1]**4)
    for repeat in (1,3):
        with pytest.raises(ValueError):equality_witness(repeat_count=repeat)


def test_coupling_bound_covers_every_combined_intervention_and_rejects_wrong_parents():
    from proof_extensions.scm_propagation import certify_linear_gaussian_coupling,verify_linear_gaussian_coupling
    source={'coefficients':[['0','0','0'],['1/2','0','0'],['1/4','1/2','0']],
        'intercepts':['0']*3,'noise_means':['0']*3,'noise_stds':['1']*3,'noise_joint':'independent_gaussian'}
    target=deepcopy(source);target['intercepts']=['1/1000']*3;target['noise_stds']=['1001/1000']*3
    c=certify_linear_gaussian_coupling(source,target);assert verify_linear_gaussian_coupling(c)['status']=='verified'
    assert len(c['interventions'])==8
    assert c['interventions'][0]['coordinate_w1_upper']==['1/500','3/1000','1/250']
    assert c['interventions'][-1]['joint_w1_l1_upper']=='0'
    assert all(all(Q(row['coordinate_w1_upper'][i])==0 for i in row['targets']) for row in c['interventions'])
    changed=deepcopy(target);changed['coefficients'][2][0]='0'
    with pytest.raises(ValueError):certify_linear_gaussian_coupling(source,changed)
    changed=deepcopy(target);changed['noise_joint']='correlated_gaussian'
    with pytest.raises(ValueError):certify_linear_gaussian_coupling(source,changed)
    forged=deepcopy(c);forged['interventions'][0]['joint_w1_l1_upper']='0'
    with pytest.raises(ValueError):verify_linear_gaussian_coupling(forged)
