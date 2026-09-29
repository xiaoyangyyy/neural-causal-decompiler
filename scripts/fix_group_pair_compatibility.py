from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8');needle='''    def address(self,path):
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self.program_id+":"+path
''';replacement=needle+'''    def intervention_paths(self,address_or_path):
        path=address_or_path.split(":",1)[1] if ":" in address_or_path else address_or_path
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self.groups.get(path,(path,))
''';s=s.replace(needle,replacement);p.write_text(s,encoding='utf-8')
p=Path('ncd/mechanism_pairs.py');s=p.read_text(encoding='utf-8');old='''    paths=[a.split(":",1)[1] for a in addresses]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a.startswith(b+"/") or b.startswith(a+"/"):return False
            for p,q in ((a,b),(b,a)):
                if p.endswith("/predicate"):
                    expr=p[:-len("predicate")]+"expr"
                    if q==expr or q.startswith(expr+"/"):return False
            # Rule branches: two incompatible branches cannot both execute.
            aa=a.split("/");bb=b.split("/")
            for left,right in zip(aa,bb):
                if left!=right:
                    if {left,right}=={"left","right"}:return False
                    break
''';new='''    paths=[tuple(executor.intervention_paths(a)) if hasattr(executor,"intervention_paths") else (a.split(":",1)[1],) for a in addresses]
    for i,left_paths in enumerate(paths):
        for right_paths in paths[i+1:]:
            for a in left_paths:
                for b in right_paths:
                    if a.startswith(b+"/") or b.startswith(a+"/"):return False
                    for p,q in ((a,b),(b,a)):
                        if p.endswith("/predicate"):
                            expr=p[:-len("predicate")]+"expr"
                            if q==expr or q.startswith(expr+"/"):return False
                    # Rule branches: two incompatible branches cannot both execute.
                    aa=a.split("/");bb=b.split("/")
                    for left,right in zip(aa,bb):
                        if left!=right:
                            if {left,right}=={"left","right"}:return False
                            break
''';assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');s+='''

def test_group_compatibility_expands_causal_ancestry():
    from ncd.mechanism_pairs import compatible,compatible_combinations
    e=RawDiscoveryExecutor(rule());groups=raw_scalar_groups(e)
    variance=next(a for a in groups if e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op=='var_stat')
    square_root=next(a for a in groups if e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op=='sqrt' and
        any(v.startswith(p+'/') for p in e.catalog[square_root.split(':',1)[1]]['members'] for v in e.catalog[variance.split(':',1)[1]]['members']))
    assert not compatible(e,[variance,square_root])
    masks=compatible_combinations(e,groups,max_order=2)
    vi,si=groups.index(variance),groups.index(square_root)
    assert not any(m[vi] and m[si] for m in masks)
''';p.write_text(s,encoding='utf-8')
