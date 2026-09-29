import numpy as np
import pytest
from ncd.cdir import composed_features
from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups,dependence_scalar_groups,regression_scalar_groups
from ncd.rules import Rule
from ncd.statistics import FEATURES


def rule():
    return Rule(FEATURES,{"expr":{"op":"var","args":[],"value":0.,"index":2},"threshold":0.,
                          "left":{"label":0},"right":{"label":1}})

def worlds():
    rng=np.random.default_rng(81);a=rng.normal(size=(32,2));b=rng.normal(size=(32,2));
    a[:,0]*=.8;a[:,1]*=2.;b[:,0]*=3.;b[:,1]*=.4
    return np.stack([a,b])

def test_raw_execution_exactly_composes_declared_features_and_rule():
    data=worlds();e=RawDiscoveryExecutor(rule());trace=e.execute(data)
    np.testing.assert_allclose(trace.features,composed_features(data),rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(trace.output,rule().predict(trace.features))
    assert all(r['visited'].all() for r in trace.records.values())
    assert raw_scalar_frontier(e) and raw_scalar_groups(e)
    assert all(e.catalog[a.split(':',1)[1]]['semantic_kind']=='scalar' for a in raw_scalar_frontier(e)+raw_scalar_groups(e))

def test_inner_variance_interchange_creates_hybrid_not_predicate_alias():
    data=worlds();e=RawDiscoveryExecutor(rule());base=e.execute(data)
    # feature 2 = log(var(x)/var(y)); patch only var(x), retaining base-world var(y).
    address=e.address('feature/2/0/0');source=base.records[address]['value'][::-1]
    patched=e.execute(data,{address:source})
    expected=np.log(source/base.records[e.address('feature/2/0/1')]['value'])
    np.testing.assert_allclose(patched.features[:,2],expected)
    assert not np.array_equal(patched.output,base.output[::-1])

def test_occurrence_identity_and_intervention_validation():
    data=worlds();e=RawDiscoveryExecutor(rule());same=RawDiscoveryExecutor(Rule.from_dict(rule().to_dict()))
    assert e.program_id==same.program_id
    address=e.address('feature/2/0/0')
    with pytest.raises(ValueError,match='shape'):e.execute(data,{address:np.ones((2,1))})
    with pytest.raises(ValueError,match='Foreign'):e.execute(data,{address.replace(e.program_id,'0'*64):np.ones(2)})
    with pytest.raises(ValueError,match='Overlapping'):e.execute(data,{address:np.ones(2),e.address('feature/2/0'):np.ones(2)})

def test_node_permutation_is_not_silently_accepted_as_feature_schema():
    bad=Rule(tuple(reversed(FEATURES)),rule().tree)
    with pytest.raises(ValueError,match='schema'):RawDiscoveryExecutor(bad)



def test_structural_equivalence_group_patches_all_occurrences():
    data=worlds();e=RawDiscoveryExecutor(rule());groups=raw_scalar_groups(e)
    address=next(a for a in groups if len(e.catalog[a.split(':',1)[1]]['members'])>1)
    base=e.execute(data);source=base.records[address]['value'][::-1]
    patched=e.execute(data,{address:source});record=patched.records[address]
    assert record['intervened'] and len(record['members'])>1
    for path in record['members']:
        np.testing.assert_allclose(patched.records[e.address(path)]['value'],source)
        assert patched.records[e.address(path)]['intervened']


def test_grouped_raw_interventions_use_world_level_pair_engine():
    from ncd.mechanism_pairs import compatible_combinations,execution_conditioned_pairs
    data=np.concatenate([worlds() for _ in range(6)],axis=0);e=RawDiscoveryExecutor(rule());addresses=raw_scalar_groups(e)
    masks=compatible_combinations(e,addresses,max_order=2)
    pairs,record=execution_conditioned_pairs(e,data,addresses,masks,count=4,seed=17)
    assert len(pairs.base)==4 and record['neural_or_truth_filtering'] is False
    natural=e.execute(data)
    for i in range(4):
        patches={a:natural.records[a]['value'][pairs.sources[i,j]:pairs.sources[i,j]+1]
                 for j,a in enumerate(addresses) if pairs.masks[i,j]}
        replay=e.execute(data[pairs.base[i:i+1]],patches)
        assert replay.output[0]==pairs.target[i]


def test_group_compatibility_expands_causal_ancestry():
    from ncd.mechanism_pairs import compatible,compatible_combinations
    e=RawDiscoveryExecutor(rule());groups=raw_scalar_groups(e)
    def op(a):return e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op
    candidates=[]
    for variance in [a for a in groups if op(a)=='var_stat']:
        for square_root in [a for a in groups if op(a)=='sqrt']:
            vm=e.catalog[variance.split(':',1)[1]]['members'];sm=e.catalog[square_root.split(':',1)[1]]['members']
            if any(v.startswith(p+'/') for p in sm for v in vm):candidates.append((variance,square_root))
    variance,square_root=candidates[0]
    assert not compatible(e,[variance,square_root])
    masks=compatible_combinations(e,groups,max_order=2)
    vi,si=groups.index(variance),groups.index(square_root)
    assert not any(m[vi] and m[si] for m in masks)


def test_dependence_internal_steps_are_exact_and_causally_typed():
    from ncd.mechanism_pairs import compatible,compatible_combinations
    data=worlds();e=RawDiscoveryExecutor(rule(),trace_dependence=True);base=e.execute(data);groups=dependence_scalar_groups(e)
    assert len(groups)==18
    def find(parent,role):
        return next(a for a in groups if any(e.catalog[p].get('dep_parent')==parent and e.catalog[p].get('dep_role')==role
            for p in e.catalog[a.split(':',1)[1]]['members']))
    bandwidth=find('feature/7','bandwidth_0');energy=find('feature/7','energy_0')
    numerator=find('feature/7','numerator');denominator=find('feature/7','denominator')
    assert not compatible(e,[bandwidth,energy]) and not compatible(e,[bandwidth,numerator])
    assert compatible(e,[numerator,denominator])
    masks=compatible_combinations(e,[bandwidth,energy,numerator,denominator],max_order=2)
    assert not any(m[0] and (m[1] or m[2] or m[3]) for m in masks)
    source=base.records[bandwidth]['value'][::-1]
    patched=e.execute(data,{bandwidth:source})
    assert not np.allclose(patched.features[:,7],base.features[:,7])
    assert not np.allclose(patched.features[:,7],base.features[::-1,7])
    n=base.records[numerator]['value'][::-1];d=base.records[denominator]['value'][::-1]
    both=e.execute(data,{numerator:n,denominator:d})
    np.testing.assert_allclose(both.features[:,7],n/np.maximum(np.abs(d),1e-12))


def test_regression_internal_steps_are_exact_grouped_and_causally_typed():
    from ncd.mechanism_pairs import compatible,compatible_combinations
    data=worlds();e=RawDiscoveryExecutor(rule(),trace_regression=True);base=e.execute(data);groups=regression_scalar_groups(e)
    assert len(groups)==28 and all(len(e.catalog[a.split(':',1)[1]]['members'])==2 for a in groups)
    np.testing.assert_allclose(base.features,composed_features(data),rtol=1e-10,atol=1e-10)
    def find(parent,fold,role):
        return next(a for a in groups if any(e.catalog[p].get('reg_parent')==parent and e.catalog[p].get('reg_fold')==fold and e.catalog[p].get('reg_role')==role
            for p in e.catalog[a.split(':',1)[1]]['members']))
    mean=find('feature/8/1/1',0,'mean_0');beta0=find('feature/8/1/1',0,'beta_0');beta1=find('feature/8/1/1',0,'beta_1')
    assert not compatible(e,[mean,beta0]) and compatible(e,[beta0,beta1])
    masks=compatible_combinations(e,[mean,beta0,beta1],max_order=2)
    assert not any(m[0] and (m[1] or m[2]) for m in masks)
    source=base.records[beta0]['value'][::-1];patched=e.execute(data,{beta0:source})
    assert not np.allclose(patched.features[:,8],base.features[:,8])
    assert not np.allclose(patched.features[:,10],base.features[:,10])
