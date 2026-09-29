from pathlib import Path
p=Path('ncd/raw_numeric_experiment.py');s=p.read_text(encoding='utf-8')
s=s.replace('RawDiscoveryExecutor,raw_scalar_groups,dependence_scalar_groups','RawDiscoveryExecutor,raw_scalar_groups,dependence_scalar_groups,regression_scalar_groups')
s=s.replace('self.target_scope not in ("base","dependence")','self.target_scope not in ("base","dependence","regression")')
old="meta=executor.catalog[member];parent=executor._nodes[meta['dep_parent']]\n            ast={'op':meta['op'],'role':meta['dep_role'],'parent':parent.to_dict()};operation=meta['op']"
new="meta=executor.catalog[member];parent_path=meta.get('dep_parent',meta.get('reg_parent'));parent=executor._nodes[parent_path]\n            ast={'op':meta['op'],'role':meta.get('dep_role',meta.get('reg_role')),'parent':parent.to_dict(),'fold':meta.get('reg_fold')};operation=meta['op']"
assert old in s;s=s.replace(old,new)
old="executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')),trace_dependence=c.target_scope=='dependence');addresses=(dependence_scalar_groups(executor) if c.target_scope=='dependence' else raw_scalar_groups(executor))"
new="executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')),trace_dependence=c.target_scope=='dependence',trace_regression=c.target_scope=='regression');addresses=(dependence_scalar_groups(executor) if c.target_scope=='dependence' else regression_scalar_groups(executor) if c.target_scope=='regression' else raw_scalar_groups(executor))"
assert s.count(old)==2;s=s.replace(old,new)
old="'scope':('dependence kernel bandwidth/energy/numerator/denominator groups; regression internals remain opaque' if c.target_scope=='dependence' else 'raw-sample mean/variance/std/correlation/variance-ratio operation groups; dependence and regression internals remain opaque')"
new="'scope':('dependence kernel bandwidth/energy/numerator/denominator groups; regression internals remain opaque' if c.target_scope=='dependence' else 'cross-fit regression fold normalization and basis-coefficient groups' if c.target_scope=='regression' else 'raw-sample mean/variance/std/correlation/variance-ratio operation groups; dependence and regression internals remain opaque')"
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
