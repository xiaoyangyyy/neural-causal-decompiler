from ncd.io import read_json
from ncd.rules import Rule
from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups
from collections import Counter
for seed in (191,192):
 e=RawDiscoveryExecutor(Rule.from_dict(read_json(f'runs/joint_seed{seed}/program.json')))
 groups=raw_scalar_groups(e)
 print(seed,len(groups),Counter(e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op for a in groups),[len(e.catalog[a.split(':',1)[1]]['members']) for a in groups])
