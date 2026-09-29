from ncd.raw_program_trace import RawDiscoveryExecutor
from ncd.rules import Rule
from ncd.io import read_json
for seed in (893,894):
 saved=read_json(f'runs/dependence_numeric_seed{seed}/summary.json')['groups'][0]['address'].split(':')[0]
 current=RawDiscoveryExecutor(Rule.from_dict(read_json(f'runs/dependence_numeric_seed{seed}/program.json')),trace_dependence=True).program_id
 print(seed,saved==current)
