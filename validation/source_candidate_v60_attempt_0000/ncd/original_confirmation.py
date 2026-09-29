"""Frozen, per-world resumable confirmation of the original SCM pipeline.

Each declared world is retained. Historical teachers/runs are never changed.
An empirical pass is not a universal causal theorem or a noise-law certificate.
"""
from dataclasses import asdict
from pathlib import Path
from fractions import Fraction as Q
import json
import tempfile
import time
import os
import numpy as np
from .io import read_json,save_json,digest
from .model import set_seed
from .active_intervention_experiment import ActiveInterventionConfig,run_active_intervention,verify_active_intervention
from .active_end_to_end_experiment import ActiveEndToEndConfig,_world_record,_aggregate,_assert_nested_close
from .active_intervention_graph import load_active_factorized_graph,intervention_response_features,padded_observational_features
from .graph_model import pair_features,graph_probabilities,decode_graph,dag_completion
from .multiverse import generate_graph_worlds
from .original_proof_workflow import statistical_bound

MODES=('observational_graph','active_graph','oracle_graph_diagnostic')
ENVS=('test_id','test_function','test_noise','test_scale','test_intervention')


def declared_units(protocol):
    return [(seed,n,env,i) for seed in protocol['seeds'] for n in protocol['nodes']
            for env in protocol['environments'] for i in range(protocol['worlds_per_cell'])]


def check_protocol(protocol,root):
    if protocol['schema']!='ncd.original-confirmation-protocol.v1' or protocol['status']!='frozen-before-confirmation':raise ValueError('Unfrozen confirmation protocol')
    units=declared_units(protocol)
    if protocol['worlds_per_cell']<1 or not units or len(units)!=len(set(units)) or protocol['declared_worlds']!=len(units):raise ValueError('Missing or duplicated declared profiles')
    if any(type(s)is not int or s<=0 for s in protocol['seeds']):raise ValueError('Invalid confirmation seeds')
    if any(n not in (3,5,8) for n in protocol['nodes']) or any(e not in ENVS for e in protocol['environments']):raise ValueError('Unsupported graph family')
    for name,value in protocol['scientific_source_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or digest(p)!=value:raise ValueError('Frozen scientific module changed: '+name)
    if protocol['historical_acceptance_sha256']!=digest(root/'scripts/active_end_to_end_acceptance.py'):raise ValueError('Historical confirmation thresholds changed')
    return units


def _artifact_manifest(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}


def _check_manifest(root):
    for name,expected in read_json(root/'manifest.json')['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or digest(p)!=expected:raise ValueError('Confirmation artifact changed')


def _graphs(world,teachers):
    data=world.sample();base=pair_features(data)
    features={'observational_graph':padded_observational_features(data,base),
              'active_graph':intervention_response_features(world,data,base)}
    graphs={};decodes={}
    for mode,teacher in teachers.items():
        probabilities=graph_probabilities(teacher,features[mode][None])[0]
        partial,decode=decode_graph(probabilities);inferred,choices=dag_completion(partial)
        graphs[mode]=inferred;decodes[mode]={**decode,'completion_choices':choices,'probabilities':probabilities.tolist()}
    return graphs,decodes


def _teachers(source):
    return {mode:load_active_factorized_graph(source/'models'/folder/'graph_teacher.pt')
            for mode,folder in (('observational_graph','observational_padded'),('active_graph','active_intervention'))}


def _training_config(protocol,seed):
    raw=protocol['training'];return ActiveInterventionConfig(seed=seed,nodes=tuple(protocol['nodes']),**raw)


def _mechanism_config(protocol,seed):
    return ActiveEndToEndConfig(seed=seed,nodes=tuple(protocol['nodes']),environments=tuple(protocol['environments']),
                               worlds_per_cell=protocol['worlds_per_cell'],**protocol['end_to_end'])


def _world(protocol,unit):
    seed,n,env,index=unit
    return generate_graph_worlds(env,protocol['worlds_per_cell'],n,seed,protocol['end_to_end']['samples'])[index]


def _unit_name(unit):
    seed,n,env,i=unit;return f'seed_{seed}_n{n}_{env}_{i}'


def _completed_unit(path,world,protocol_hash,teacher_hashes):
    _check_manifest(path);saved=read_json(path/'record.json')
    if saved['world_id']!=world.identity or saved['protocol_sha256']!=protocol_hash or saved['teacher_sha256']!=teacher_hashes:
        raise ValueError('World/teacher/protocol checkpoint mismatch')
    if saved['world_metadata']!=json.loads(json.dumps(world.metadata())):raise ValueError('World metadata changed')
    return saved


def _world_compute(path,world,unit,teachers,c,protocol_hash,teacher_hashes):
    _,n,env,index=unit
    graphs,decodes=_graphs(world,teachers)
    record=_world_record(path,world,n,env,index,graphs,decodes,c,True)
    saved={'world_id':world.identity,'world_metadata':world.metadata(),'protocol_sha256':protocol_hash,
           'teacher_sha256':teacher_hashes,'metrics':record}
    save_json(path/'record.json',saved);save_json(path/'manifest.json',{'files':_artifact_manifest(path)})
    return saved


def verify_world(path,world,unit,teachers,c,protocol_hash,teacher_hashes):
    saved=_completed_unit(path,world,protocol_hash,teacher_hashes)
    graphs,decodes=_graphs(world,teachers);_,n,env,index=unit
    with tempfile.TemporaryDirectory(prefix='ncd_verify_world_') as temp:
        record=_world_record(path,world,n,env,index,graphs,decodes,c,False,temp)
    _assert_nested_close(record,saved['metrics'],'world_confirmation')
    return saved


def _resource_guard(output,limits):
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>limits['artifact_bytes']:
        raise RuntimeError('Artifact budget exhausted; preserved checkpoints')
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        data=Counters();data.cb=ctypes.sizeof(data)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(data),data.cb):
            raise RuntimeError('Cannot check process memory budget')
        if data.WorkingSetSize>limits['memory_bytes']:raise RuntimeError('Memory budget exhausted; preserved checkpoints')


def _summary(protocol,rows,fully_replayed):
    complete=len(rows)==protocol['declared_worlds']
    runs=[]
    for seed in protocol['seeds']:
        records=[r['metrics'] for unit,r in rows if unit[0]==seed]
        if not records:continue
        aggregate={mode:{method:_aggregate(records,mode,method) for method in ('baseline','structured')} for mode in MODES}
        graph={mode:{'exact_accuracy':float(np.mean([r['graph'][mode]['exact'] for r in records])),
                     'mean_shd':float(np.mean([r['graph'][mode]['shd'] for r in records]))} for mode in MODES[:2]}
        runs.append({'seed':seed,'world_count':len(records),'aggregate':aggregate,'graph':graph})
    criteria={};bounds=[]
    if complete:
        def mean(mode,key,method='structured'):return float(np.mean([r['aggregate'][mode][method][key] for r in runs]))
        def reduction(before,after):return (before-after)/before if before>0 else 0.0
        obs,active,oracle=MODES
        criteria={
            'all_declared_worlds_retained':complete,'all_worlds_independently_replayed':fully_replayed,
            'active_graph_exact_higher_each_seed':all(r['graph'][active]['exact_accuracy']>r['graph'][obs]['exact_accuracy'] for r in runs),
            'pooled_graph_exact_gain_at_least_20pp':float(np.mean([r['graph'][active]['exact_accuracy']-r['graph'][obs]['exact_accuracy'] for r in runs]))>=.20,
            'active_graph_shd_lower_each_seed':all(r['graph'][active]['mean_shd']<r['graph'][obs]['mean_shd'] for r in runs),
            'active_truth_nmse_lower_each_seed':all(r['aggregate'][active]['structured']['mean_symbolic_truth_nmse']<r['aggregate'][obs]['structured']['mean_symbolic_truth_nmse'] for r in runs),
            'pooled_truth_nmse_reduction_at_least_20pct':reduction(mean(obs,'mean_symbolic_truth_nmse'),mean(active,'mean_symbolic_truth_nmse'))>=.20,
            'active_intervention_mae_lower_each_seed':all(r['aggregate'][active]['structured']['mean_intervention_effect_mae']<r['aggregate'][obs]['structured']['mean_intervention_effect_mae'] for r in runs),
            'pooled_intervention_mae_reduction_at_least_20pct':reduction(mean(obs,'mean_intervention_effect_mae'),mean(active,'mean_intervention_effect_mae'))>=.20,
            'active_structured_neural_nmse_lower_each_seed':all(r['aggregate'][active]['structured']['mean_symbolic_neural_nmse']<r['aggregate'][active]['baseline']['mean_symbolic_neural_nmse'] for r in runs),
            'pooled_neural_nmse_reduction_at_least_15pct':reduction(mean(active,'mean_symbolic_neural_nmse','baseline'),mean(active,'mean_symbolic_neural_nmse'))>=.15,
            'active_structured_atoms_no_more_than_baseline':mean(active,'mean_nonconstant_atoms')<=mean(active,'mean_nonconstant_atoms','baseline'),
            'active_truth_within_1_75x_oracle':mean(active,'mean_symbolic_truth_nmse')<=1.75*mean(oracle,'mean_symbolic_truth_nmse'),
            'active_intervention_within_2x_oracle':mean(active,'mean_intervention_effect_mae')<=2*mean(oracle,'mean_intervention_effect_mae')}
        # Independent worlds are the only statistical units, never node edges or source-target pairs.
        for seed in list(protocol['seeds'])+[None]:
            for mode in MODES[:2]:
                selected=[r for unit,r in rows if seed is None or unit[0]==seed]
                units=[{'world_id':r['world_id'],'success':'1' if r['metrics']['graph'][mode]['exact'] else '0'} for r in selected]
                bounds.append({'seed':seed,'mode':mode,'bound':statistical_bound(units,protocol['delta'],2*(len(protocol['seeds'])+1))})
    done={tuple(unit) for unit,_ in rows}
    return {'schema':'ncd.original-confirmation-summary.v1','state':'verified' if complete and fully_replayed else 'computed' if complete else 'partial',
        'declared_worlds':protocol['declared_worlds'],'computed_worlds':len(rows),'unresolved_units':[_unit_name(u) for u in declared_units(protocol) if tuple(u) not in done],'runs':runs,'criteria':criteria,
        'science_passed':complete and fully_replayed and bool(criteria) and all(criteria.values()),'world_level_confidence_bounds':bounds,
        'overall_objective_achieved':False,'noise_independence_and_uniform_mechanism_guarantees':'unresolved',
        'boundary':'Finite synthetic benchmark confirmation; neither global causal identification nor full-domain circuit recovery follows.'}



def _run_worker(protocol_path,task,output,deadline,seed,unit=None):
    import subprocess
    import sys
    args=[sys.executable,'-m','ncd.original_confirmation_worker','--protocol',str(protocol_path),
          '--task',task,'--output',str(output),'--seed',str(seed)]
    if unit is not None:args+=['--unit',json.dumps(list(unit))]
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Stage budget exhausted before starting unit')
    # No interactive window. Only this worker is terminated on deadline/resource excess.
    process=subprocess.Popen(args,creationflags=0x08000000 if os.name=='nt' else 0)
    try:
        while process.poll() is None:
            if time.monotonic()>=deadline:raise TimeoutError('Stage deadline reached; incomplete attempt retained')
            if os.name=='nt':
                import ctypes
                from ctypes import wintypes
                class Counters(ctypes.Structure):
                    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
                        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
                counters=Counters();counters.cb=ctypes.sizeof(counters)
                psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
                if psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(process._handle)),ctypes.byref(counters),counters.cb):
                    limit=read_json(protocol_path)['limits']['memory_bytes']
                    if counters.WorkingSetSize>limit:raise RuntimeError('Worker memory budget exceeded')
            time.sleep(min(.25,max(.001,deadline-time.monotonic())))
        if process.returncode:raise RuntimeError('Confirmation worker failed with exit code '+str(process.returncode))
    except BaseException:
        if process.poll() is None:process.terminate();process.wait()
        raise

def run_confirmation(protocol_path,resume=False,seconds=None,verify_only=False):
    protocol_path=Path(protocol_path).resolve();protocol=read_json(protocol_path)
    if seconds is not None and (not np.isfinite(seconds) or seconds<=0):raise ValueError('Positive finite stage budget required')
    root=(protocol_path.parent/protocol.get('root','..')).resolve();units=check_protocol(protocol,root)
    output=(root/protocol['output']).resolve()
    if not output.is_relative_to(root):raise ValueError('Confirmation output escapes project')
    if output.exists() and any(output.iterdir()) and not (resume or verify_only):raise FileExistsError('Use --resume for existing confirmation')
    output.mkdir(parents=True,exist_ok=True)
    frozen=output/'protocol.json'
    if frozen.exists() and read_json(frozen)!=protocol:raise ValueError('Changed frozen confirmation protocol')
    if not verify_only:save_json(frozen,protocol)
    ph=digest(protocol_path);deadline=time.monotonic()+(protocol['limits']['stage_seconds'] if seconds is None else seconds)
    set_seed(protocol['development_seed']);rows=[];replayed=True
    for seed in protocol['seeds']:
        training=output/f'training_seed_{seed}'
        current=training/'complete.json'
        if not current.exists():
            if verify_only or time.monotonic()>=deadline:break
            attempts=sorted(training.glob('attempt_*'));attempt=training/f'attempt_{len(attempts):04d}'
            attempt.mkdir(parents=True)
            print('confirmation training seed',seed,flush=True)
            try:
                _run_worker(protocol_path,'train',attempt,deadline,seed)
            except (TimeoutError,RuntimeError) as exc:
                save_json(output/'last_incomplete_attempt.json',{'phase':'training','seed':seed,'error':str(exc),'attempt':str(attempt)})
                break
            save_json(current,{'attempt':attempt.name,'protocol_sha256':ph,'manifest_sha256':digest(attempt/'manifest.json'),
                'teacher_sha256':{mode:digest(attempt/'models'/folder/'graph_teacher.pt') for mode,folder in
                    (('observational_graph','observational_padded'),('active_graph','active_intervention'))}})
        meta=read_json(current);source=training/meta['attempt']
        expected=asdict(_training_config(protocol,seed));expected=json.loads(json.dumps(expected))
        if read_json(source/'config.json')!=expected:raise ValueError('Training config differs from protocol')
        if meta['protocol_sha256']!=ph or digest(source/'manifest.json')!=meta['manifest_sha256']:raise ValueError('Training checkpoint contract changed')
        if verify_only:
            try:_run_worker(protocol_path,'verify_training',source,deadline,seed)
            except TimeoutError:replayed=False;break
        teachers=_teachers(source)
        for mode,folder in (('observational_graph','observational_padded'),('active_graph','active_intervention')):
            if digest(source/'models'/folder/'graph_teacher.pt')!=meta['teacher_sha256'][mode]:raise ValueError('Teacher changed after freeze')
        c=_mechanism_config(protocol,seed)
        for unit in [u for u in units if u[0]==seed]:
            cell=output/'units'/_unit_name(unit);pointer=cell/'complete.json';world=_world(protocol,unit)
            if pointer.exists():
                path=cell/read_json(pointer)['attempt']
                if verify_only:
                    try:_run_worker(protocol_path,'verify_world',path,deadline,seed,unit)
                    except TimeoutError:replayed=False;break
                saved=_completed_unit(path,world,ph,meta['teacher_sha256'])
            else:
                if verify_only or time.monotonic()>=deadline:replayed=False;break
                _resource_guard(output,protocol['limits'])
                attempts=sorted(cell.glob('attempt_*'));path=cell/f'attempt_{len(attempts):04d}';path.mkdir(parents=True)
                print('confirmation world',_unit_name(unit),flush=True)
                try:
                    _run_worker(protocol_path,'world',path,deadline,seed,unit)
                except (TimeoutError,RuntimeError) as exc:
                    save_json(cell/'last_incomplete_attempt.json',{'error':str(exc),'attempt':path.name})
                    replayed=False
                    if isinstance(exc,TimeoutError):break
                    continue
                saved=_completed_unit(path,world,ph,meta['teacher_sha256'])
                save_json(pointer,{'attempt':path.name})
                replayed=False
            rows.append((unit,saved))
            if not verify_only:save_json(output/'summary.json',_summary(protocol,rows,False))
        if time.monotonic()>=deadline:break
    summary=_summary(protocol,rows,verify_only and replayed)
    if verify_only:
        # Verification is allowed to report only a partial verified prefix, never all 300.
        if len(rows)==len(units):save_json(output/'verified_summary.json',summary)
    else:save_json(output/'summary.json',summary)
    return summary
