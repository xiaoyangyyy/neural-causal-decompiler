from pathlib import Path
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');s=s.replace('RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups,dependence_scalar_groups','RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups,dependence_scalar_groups,regression_scalar_groups')
s+='''

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
''';p.write_text(s,encoding='utf-8')
