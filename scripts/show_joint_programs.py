from ncd.io import read_json
from ncd.rules import Rule
for seed in (191,192):print(seed,'\n'+Rule.from_dict(read_json(f'runs/joint_seed{seed}/program.json')).text())
