from pathlib import Path
p=Path('ncd/raw_numeric_experiment.py');s=p.read_text(encoding='utf-8')
s=s.replace('from .raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups','from .raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups,dependence_scalar_groups')
s=s.replace('    train_pairs:int=384;test_pairs:int=1024;steps:int=120;rank:int=1;numeric_weight:float=.5','    train_pairs:int=384;test_pairs:int=1024;steps:int=120;rank:int=1;numeric_weight:float=.5;target_scope:str="base"')
s=s.replace('''        if self.samples<16 or not np.isfinite(self.numeric_weight) or self.numeric_weight<0:raise ValueError("Invalid raw numeric configuration")''','''        if self.samples<16 or not np.isfinite(self.numeric_weight) or self.numeric_weight<0 or self.target_scope not in ("base","dependence"):raise ValueError("Invalid raw numeric configuration")''')
old='''        path=address.split(':',1)[1];members=executor.catalog[path]['members'];node=executor._nodes[members[0]]
        rows.append({'address':address,'path':path,'ast':node.to_dict(),'operation':node.op,'occurrences':list(members)})'''
new='''        path=address.split(':',1)[1];members=executor.catalog[path]['members'];member=members[0]
        if member in executor._nodes:
            node=executor._nodes[member];ast=node.to_dict();operation=node.op
        else:
            meta=executor.catalog[member];parent=executor._nodes[meta['dep_parent']]
            ast={'op':meta['op'],'role':meta['dep_role'],'parent':parent.to_dict()};operation=meta['op']
        rows.append({'address':address,'path':path,'ast':ast,'operation':operation,'occurrences':list(members)})'''
assert old in s;s=s.replace(old,new)
s=s.replace("model=load_model(root/'teacher.pt');executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')));addresses=raw_scalar_groups(executor)","model=load_model(root/'teacher.pt');executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')),trace_dependence=c.target_scope=='dependence');addresses=(dependence_scalar_groups(executor) if c.target_scope=='dependence' else raw_scalar_groups(executor))")
s=s.replace("'scope':'raw-sample mean/variance/std/correlation/variance-ratio operation groups; dependence and regression internals remain opaque'","'scope':('dependence kernel bandwidth/energy/numerator/denominator groups; regression internals remain opaque' if c.target_scope=='dependence' else 'raw-sample mean/variance/std/correlation/variance-ratio operation groups; dependence and regression internals remain opaque')")
# Occurs twice run and verify; replace caught both exact string.
p.write_text(s,encoding='utf-8')
# Ensure both constructor instances changed
print(s.count("trace_dependence=c.target_scope=='dependence'"))
