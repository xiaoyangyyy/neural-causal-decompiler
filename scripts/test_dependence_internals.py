from pathlib import Path
p=Path('ncd/mechanism_pairs.py');s=p.read_text(encoding='utf-8');s=s.replace('''                            if {left,right}=={"left","right"}:return False
                            break
    return True
''','''                            if {left,right}=={"left","right"}:return False
                            break
    if hasattr(executor,"interventions_compatible") and not executor.interventions_compatible(addresses):return False
    return True
''');p.write_text(s,encoding='utf-8')
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');s=s.replace('from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups','from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups,dependence_scalar_groups')
s+='''

def test_dependence_internal_steps_are_exact_and_causally_typed():
    from ncd.mechanism_pairs import compatible,compatible_combinations
    data=worlds();e=RawDiscoveryExecutor(rule());base=e.execute(data);groups=dependence_scalar_groups(e)
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
''';p.write_text(s,encoding='utf-8')
