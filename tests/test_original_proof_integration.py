from copy import deepcopy
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
import pytest
from ncd.discovery_fidelity_proof import certify_discovery_box,verify_discovery_box
from ncd.interchange_proof import certify_interchange,verify_interchange
from ncd.mechanism_cegis_proof import run_cegis,verify_cegis
from ncd.original_proof_workflow import prove,audit_requirements
from ncd.original_confirmation import _summary,_resource_guard,declared_units
from ncd.io import save_json,read_json


def tiny_discoverer():
    def layer(rows,cols,bias,activation):return {'weights':[['0']*cols for _ in range(rows)],'bias':bias,'activation':activation}
    return {'schema':'ncd.frozen-discoverer.v1','width':1,'checkpoint_sha256':'test',
        'encoder':[layer(1,2,['0'],'tanh'),layer(1,1,['0'],'tanh')],
        'head':[layer(1,4,['0'],'tanh'),layer(4,1,['0','0','1','0'],'identity')]}


def test_discovery_full_dataset_box_and_forged_winner():
    network=tiny_discoverer();program={'names':['corr','abs_corr','var_log_ratio','skew_x','skew_y','kurt_x','kurt_y','dep_xy','resdep_xy','resdep_yx','reserr_xy','reserr_yx','mixed_xy','mixed_yx'],'tree':{'label':2}}
    domain=[[[str(Q(i,17)),str(Q(i,17))],[str(Q(i*i,289)),str(Q(i*i,289))]] for i in range(16)]
    c=certify_discovery_box(network,program,domain)
    assert c['status']=='proved' and verify_discovery_box(network,program,c)['conclusion']=='proved'
    changed=deepcopy(c);changed['neural_winners']=[1]
    with pytest.raises(ValueError):verify_discovery_box(network,program,changed)
    wrong=deepcopy(program);wrong['tree']['label']=0
    assert certify_discovery_box(network,wrong,domain)['status']=='refuted'


def test_uncertain_raw_regression_boundary_is_not_a_success():
    network=tiny_discoverer();names=['corr','abs_corr','var_log_ratio','skew_x','skew_y','kurt_x','kurt_y','dep_xy','resdep_xy','resdep_yx','reserr_xy','reserr_yx','mixed_xy','mixed_yx']
    program={'names':names,'tree':{'expr':{'op':'var','index':8},'threshold':.1,'left':{'label':2},'right':{'label':1}}}
    domain=[[['0','1'],['0','1']]]*16
    c=certify_discovery_box(network,program,domain)
    assert c['status']=='unresolved' and ('boundary' in c['reason'] or 'pivot' in c['reason'])


def test_joint_interchange_and_unpatched_collateral():
    domain=[['-1','1']]*2;read=[['1','0'],['0','1']];write=[['1','0'],['0','1']]
    c=certify_interchange(read,write,domain,domain,[[True,False],[False,True],[True,True]],'0')
    assert verify_interchange(c)['conclusion']=='proved'
    changed=certify_interchange(read,[['1','0'],['1/10','1']],domain,domain,[[True,False]],'1/100')
    assert changed['status']=='refuted' and changed['rows'][0]['coordinate_error_intervals'][1]==['-1/5','1/5']
    assert changed['mapping_family_impossibility_proved'] is False


def test_cegis_uses_frozen_predictions_and_retains_first_failure(tmp_path):
    import torch
    from ncd.mechanisms import NeuralMechanism
    model=NeuralMechanism((0,),width=4)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.network[4].bias.fill_(.15)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'parents':(0,),'width':4,'state_dict':model.state_dict()},checkpoint)
    save_json(tmp_path/'source_scm.json',{'equations':[{'op':'constant','value':0}]})
    c=run_cegis(tmp_path,checkpoint,tmp_path/'source_scm.json',0,[['-1','1']],rounds=1,max_boxes=8,seconds=30)
    assert c['history'][0]['conclusion']=='refuted' and c['status']=='proved'
    assert verify_cegis(tmp_path,c)['rounds']==2
    changed=deepcopy(c);changed['global_minimum_proved']=True
    with pytest.raises(ValueError):verify_cegis(tmp_path,changed)


def test_cli_require_closed_returns_nonzero(tmp_path,capsys):
    from ncd.cli import main
    (tmp_path/'requirements.md').write_text('R0-R13')
    save_json(tmp_path/'config.json',{'root':'.','output':'bundle','requirements':'requirements.md',
        'jobs':[{'id':'gaussian','kind':'gaussian_nonidentifiability'}]})
    prove(tmp_path/'config.json')
    assert main(['audit-requirements',str(tmp_path/'bundle'),'--require-closed'])==1
    assert audit_requirements(tmp_path/'bundle')['overall_objective_achieved'] is False


def protocol():
    return {'seeds':[8101,8102],'nodes':[3,5,8],'environments':['test_id','test_function','test_noise','test_scale','test_intervention'],
        'worlds_per_cell':10,'declared_worlds':300,'delta':'1/100'}


def test_confirmation_retains_every_uncomputed_world():
    p=protocol();s=_summary(p,[],False)
    assert len(declared_units(p))==300 and len(s['unresolved_units'])==300
    assert s['state']=='partial' and not s['science_passed'] and not s['overall_objective_achieved']


def test_artifact_resource_budget_preserves_existing_files(tmp_path):
    existing=tmp_path/'artifact';existing.write_bytes(b'preserve')
    with pytest.raises(RuntimeError):_resource_guard(tmp_path,{'artifact_bytes':1,'memory_bytes':8*1024**3})
    assert existing.read_bytes()==b'preserve'


def test_new_confirmation_preserves_historical_frozen_metric_gates():
    root=Path(__file__).resolve().parents[1]
    saved=root/'validation/active_end_to_end_acceptance.json'
    if not saved.exists():return
    protocol={'seeds':[4993,4994],'nodes':[3,5,8],'environments':['test_id','test_function','test_noise','test_scale','test_intervention'],
        'worlds_per_cell':1,'declared_worlds':30,'delta':'1/100'}
    rows=[]
    for seed in protocol['seeds']:
        summary=read_json(root/f'runs/active_end_to_end_seed{seed}/summary.json')
        for r in summary['records']:
            unit=(seed,r['nodes'],r['environment'],r['world_index'])
            rows.append((unit,{'world_id':r['world_id'],'metrics':r}))
    c=_summary(protocol,rows,True)['criteria'];old=read_json(saved)['criteria']
    for key,value in old.items():
        if key not in ('complete_replay','full_15_cell_coverage'):assert c[key]==value,key


def test_completed_world_metadata_roundtrip_preserves_tuple_graph(tmp_path):
    from types import SimpleNamespace
    from ncd.original_confirmation import _completed_unit,_artifact_manifest
    world=SimpleNamespace(identity='same-world',metadata=lambda:{'graph':((0,1),(0,0)),'parents':(0,)})
    save_json(tmp_path/'record.json',{'world_id':world.identity,'world_metadata':world.metadata(),
        'protocol_sha256':'fixed','teacher_sha256':{'a':'b'}})
    save_json(tmp_path/'manifest.json',{'files':_artifact_manifest(tmp_path)})
    assert _completed_unit(tmp_path,world,'fixed',{'a':'b'})['world_id']=='same-world'


def test_interval_linear_solver_and_crossfit_tied_predictors():
    from ncd.proof_intervals import Interval
    from ncd.discovery_fidelity_proof import interval_solve,crossfit_intervals
    matrix=[[Interval.point(2),Interval.point(1)],[Interval.point(1),Interval.point(3)]]
    solution=interval_solve(matrix,[Interval.point(3),Interval.point(4)])
    assert all(v.lo<=1<=v.hi for v in solution)
    y=[Interval.point(i) for i in range(16)]
    prediction=crossfit_intervals(y,[[Interval.point(0)]]*16)
    for i,v in enumerate(prediction):
        numerator=64 if i%2==0 else 56
        exact=Q(numerator)/(8+Q(1e-8))
        assert v.lo<=exact<=v.hi
    with pytest.raises(ValueError):interval_solve([[Interval('-1','1')]],[Interval.point(1)])


def test_independent_intervention_sources_cannot_use_shared_source_cancellation():
    read=[['1'],['1']];write=[['1/2','1/2']];base=[['-1','1']];masks=[[True,True]]
    common=certify_interchange(read,write,base,base,masks,'0')
    independent=certify_interchange(read,write,base,[base,base],masks,'0',True)
    assert verify_interchange(common)['conclusion']=='proved'
    assert verify_interchange(independent)['conclusion']=='refuted'
    assert independent['rows'][0]['coordinate_error_intervals']==[['-1','1'],['-1','1']]
