from ncd.io import read_json
from ncd.rules import Rule
from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups
import json
x=RawDiscoveryExecutor(Rule.from_dict(read_json('runs/joint_seed191/program.json')))
for a in raw_scalar_groups(x):
 p=a.split(':',1)[1];m=x.catalog[p]['members'];node=x._nodes[m[0]]
 print(p,node.op,len(m),json.dumps(node.to_dict(),sort_keys=True))
