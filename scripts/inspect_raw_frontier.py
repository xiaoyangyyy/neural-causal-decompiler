from ncd.io import read_json
from ncd.rules import Rule
from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier
from collections import Counter
for seed in (191,192):
 e=RawDiscoveryExecutor(Rule.from_dict(read_json(f'runs/joint_seed{seed}/program.json')))
 frontier=raw_scalar_frontier(e)
 print(seed,len(frontier),Counter(e.catalog[a.split(':',1)[1]]['op'] for a in frontier))
 for a in frontier:print(a.split(':',1)[1],e.catalog[a.split(':',1)[1]]['op'])
