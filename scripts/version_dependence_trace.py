from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8')
s=s.replace('def __init__(self,rule):\n        if not isinstance', 'def __init__(self,rule,*,trace_dependence=False):\n        if not isinstance')
s=s.replace('''        self.program=rule;self.nodes=discovery_feature_nodes();self.catalog={};self._nodes={};physical=[];virtual=[];dep_keys={}
''','''        self.program=rule;self.trace_dependence=bool(trace_dependence);self.nodes=discovery_feature_nodes();self.catalog={};self._nodes={};physical=[];virtual=[];dep_keys={}
''')
s=s.replace('            if node.op=="dep":','            if self.trace_dependence and node.op=="dep":')
s=s.replace('''        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict(),"groups":self._all_groups}
        self.program_id=hashlib.sha256''','''        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict(),"groups":self.groups}
        if self.trace_dependence:payload["dependence_groups"]=self.dep_groups
        self.program_id=hashlib.sha256''')
s=s.replace('value=dep(values,path,index) if node.op=="dep" else _apply(node,values,samples)','value=dep(values,path,index) if self.trace_dependence and node.op=="dep" else _apply(node,values,samples)')
p.write_text(s,encoding='utf-8')
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8')
old="data=worlds();e=RawDiscoveryExecutor(rule());base=e.execute(data);groups=dependence_scalar_groups(e)"
s=s.replace(old,"data=worlds();e=RawDiscoveryExecutor(rule(),trace_dependence=True);base=e.execute(data);groups=dependence_scalar_groups(e)")
p.write_text(s,encoding='utf-8')
