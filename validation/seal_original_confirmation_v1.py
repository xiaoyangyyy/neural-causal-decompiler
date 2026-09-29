"""Seal completed independent replay without repeating training or world inference."""
from pathlib import Path
from ncd.io import read_json,save_json,digest
from ncd.original_confirmation import check_protocol,declared_units,_unit_name,_summary
ROOT=Path(__file__).resolve().parents[1]
protocol=ROOT/'validation/original_confirmation_protocol.json';p=read_json(protocol);out=ROOT/p['output']
assert len(check_protocol(p,ROOT))==300
launch=read_json(ROOT/'validation/original_confirmation_launch/launch.json')
assert launch['status']=='stage-ended' and launch['installed_replay_exit_code']==0 and launch['replayed_worlds']==300
s=read_json(out/'verified_summary.json');replay=read_json(ROOT/'validation/original_confirmation_launch/replay_result.json')
assert s==replay and s['science_passed'] and len(s['criteria'])==14 and all(s['criteria'].values()) and not s['unresolved_units']
rows=[];seal={};ids=set()
for unit in declared_units(p):
 cell=out/'units'/_unit_name(unit);pointer=cell/'complete.json';attempt=cell/read_json(pointer)['attempt'];record=read_json(attempt/'record.json')
 assert record['world_id'] not in ids and record['protocol_sha256']==digest(protocol);ids.add(record['world_id']);rows.append((unit,record))
 manifest=read_json(attempt/'manifest.json');assert manifest['files']['record.json']==digest(attempt/'record.json')
 for file in (pointer,attempt/'manifest.json',attempt/'record.json'):seal[file.relative_to(ROOT).as_posix()]=digest(file)
assert _summary(p,rows,True)==s
for seed in p['seeds']:
 t=out/f'training_seed_{seed}';meta=read_json(t/'complete.json');a=t/meta['attempt'];assert digest(a/'manifest.json')==meta['manifest_sha256']
 for mode,folder in (('observational_graph','observational_padded'),('active_graph','active_intervention')):assert digest(a/'models'/folder/'graph_teacher.pt')==meta['teacher_sha256'][mode]
 for f in (t/'complete.json',a/'manifest.json'):seal[f.relative_to(ROOT).as_posix()]=digest(f)
for f in (protocol,out/'verified_summary.json',ROOT/'validation/original_confirmation_launch/replay_result.json',ROOT/'validation/original_confirmation_launch/launch.json'):seal[f.relative_to(ROOT).as_posix()]=digest(f)
bounds=[{'seed':b['seed'],'mode':b['mode'],**{k:v for k,v in b['bound'].items() if k!='units'},'unit_count':len(b['bound']['units'])} for b in s['world_level_confidence_bounds']]
acceptance={'schema':'ncd.original-confirmation-acceptance.v1','status':'300-world-confirmation-and-installed-replay-accepted-original-open','declared_worlds':300,'replayed_worlds':300,'unresolved_experimental_units':[],
 'original_thresholds_unchanged':True,'criteria':s['criteria'],'runs':s['runs'],'world_level_conditional_confidence_bounds':bounds,
 'scope':'frozen finite benchmark; not uniform mechanism error, causal identification, noise-law identification, or all intervention families',
 'original_claim_counts':{'proved':0,'refuted':2,'unresolved':36},'whole_project_complete':False,'original_objective_achieved':False,'full_plan_implemented':False,
 'resource_monitor_limitation':'Historical supervisor monitored launcher handles; no retrospective claim of an enforced descendant-wide 8 GiB job limit.',
 'failed_initial_replay_launch_retained':True,'evidence_sha256':seal,'sealer_sha256':digest(Path(__file__))}
save_json(ROOT/'validation/original_confirmation_acceptance_v1.json',acceptance)
print('accepted 300 worlds, 14 criteria; original 36 claims still unresolved')
