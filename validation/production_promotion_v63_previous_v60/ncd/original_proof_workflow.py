"""Original-claim ledger and independent, scope-preserving proof workflows."""
from pathlib import Path
from fractions import Fraction as Q
import shutil
from .io import read_json,save_json,digest
from .gaussian_identifiability import certify_gaussian_nonidentifiability,verify_gaussian_nonidentifiability
from .frozen_mechanism_proof import export_mechanism,certify_mechanism,verify_mechanism
from .scm_error_certificate import certify_scm_error,verify_scm_error
from .discovery_fidelity_proof import export_discoverer,certify_discovery_box,verify_discovery_box
from .proof_intervals import Interval,logarithm
from .mechanism_cegis_proof import run_cegis,verify_cegis
from .interchange_proof import certify_interchange,verify_interchange

CLAIMS = {
 'R0': [('objects','Separate discovery network, discovery program, true SCM and neural mechanisms'),
        ('observational_unique_direction','Every allowed observational linear Gaussian SCM has a uniquely recoverable direction'),
        ('uniform_finite_sample_direction','Some observational estimator recovers every allowed linear Gaussian direction with error strictly below one half'),
        ('equivalence_contract','Specify fidelity domains and distinguish mathematical from device semantics')],
 'R1': [('world_generator','SCM sampling, noises and interventions implement the declared benchmark'),('noise_law','Explicitly justify each declared noise independence and law')],
 'R2': [('sample_invariance','Discovery is invariant to sample order on its declared domain'),('variable_equivariance','Variable relabeling commutes with discovery'),('discovery_guarantee','State a justified discovery accuracy or identification guarantee')],
 'R3': [('typed_semantics','CDIR operators, protected operations and branches have explicit executable semantics'),('compositionality','Composed program execution follows the declared semantics')],
 'R4': [('program_fidelity','Automatically extract a program with verified fidelity on the declared discovery domain'),('cegis','Use checked counterexamples to refine candidates'),('mdl','Prove minimality only under the explicitly fixed language and search bounds')],
 'R5': [('interchange','Verify the fixed neural/program mapping under all declared compatible interventions'),('intermediate_numeric','Verify numeric intermediates and collateral preservation'),('composition','Compose local relations across multiple computation steps')],
 'R6': [('separate_targets','Report network fidelity, truth accuracy, graph and mechanism errors separately'),('evaluation_guarantee','Replay comparisons and justify world-level statistical units')],
 'R7': [('counterexamples','Return independently checked concrete counterexamples'),('worst_case','Certify any claimed global worst-case bound on its entire declared domain')],
 'R8': [('function_ood','Justify claims on the declared function shift family'),('noise_ood','Justify claims on the declared noise shift family'),('scale_ood','Justify claims on the declared scale shift family'),('intervention_ood','Justify claims on the declared intervention family')],
 'R9': [('graph_recovery','Recover an identified DAG or justified graph equivalence class on 3/5/8 nodes'),('sepsets','Tie separations and orientation decisions to evidence rather than inserted teacher truth'),('neural_program','Extract and verify the network graph-discovery computation')],
 'R10': [('end_to_end','Recover an explicit SCM using inferred parents without oracle repairs'),('mechanism','Certify the frozen neural-to-symbolic mechanism approximation'),('noise','Justify exogenous laws and independence rather than silently assuming empirical residual validity'),('intervention_distribution','Certify intervention effects and distribution bounds with verified premises')],
 'R11': [('ablations','Preserve and independently replay all declared ablations without selective reporting')],
 'R12': [('semantic_equivalence','Certify equivalence in the declared operator subset'),('canonicalization','Separate deterministic serialization from semantic uniqueness'),('shortest_unique','Justify any uniqueness or globally shortest-program claim')],
 'R13': [('reproduction','Provide portable artifacts, source bindings and independent installed-package replay'),('extensions','Distinguish defined external tasks from architecture visions')]
}


def _claim_contract(requirement,key):
    # These are obligations, not assertions that the listed premises were verified.
    target='frozen discovery network and extracted discovery program'
    domain='entire admissible discovery input domain; a local proof box is only scoped evidence'
    interventions='observational input changes; no identifying intervention assumed'
    error='classification label agreement for fidelity; graph accuracy reported separately'
    assumptions=['a frozen target, declared input semantics and domain must be bound before closure']
    category='network_fidelity'
    if requirement in ('R0','R1'):
        target='allowed true SCM family, frozen networks and executable programs as separate objects'
        domain='all allowed SCMs and observation laws, including linear Gaussian models'
        error='identifiability or exact object/semantics specification, not benchmark mean accuracy'
        category='true_causal_correctness'
        assumptions=['acyclic SCM with explicitly stated external noise law; additional identification assumptions require separate evidence']
    elif requirement=='R5':
        target='fixed neural read/write map, executed intermediate variables and program steps'
        domain='all reachable base states, source states and branch paths in the declared abstraction domain'
        interventions='all declared compatible combinations, including independent source worlds per variable'
        error='numeric intermediate and final-output bounds, collateral preservation and guard agreement'
        category='internal_mechanism_alignment'
        assumptions=['fixed mapping before validation','branch accessibility and local relations must be certified','a failed map does not exclude its mapping family']
    elif requirement=='R6':
        target='frozen neural/program pair, independent true SCM evaluator and world-level statistics'
        domain='all registered experiments, including failed runs and confirmation worlds'
        error='separate network fidelity, graph, true mechanism, noise and intervention errors; familywise confidence 99 percent'
        category='evaluation_contract'
        assumptions=['independent world units','no reuse of evaluator truth in candidate search']
    elif requirement=='R8':
        target='frozen discovery and mechanism networks, programs and true benchmark SCMs'
        domain={'function_ood':'declared function-shift family','noise_ood':'declared noise-shift family','scale_ood':'declared scale-shift family','intervention_ood':'declared intervention-shift family'}[key]+'; unbounded universal OOD claims are not substituted by a finite benchmark'
        interventions='declared node interventions and compatible combinations within that family'
        error='truth accuracy and neural/program fidelity separately; numeric and distributional intervention errors'
        category='true_causal_correctness'
        assumptions=['function, noise, scale, intervention support and sampling law frozen before confirmation']
    elif requirement=='R9':
        target='3/5/8-node discovery networks, extracted graph computation and true SCM evaluator'
        domain='full declared 3/5/8-node observation or intervention task, separated by information access'
        interventions='none for observational task; recorded node-do coverage for interventional task'
        error='identified DAG or valid equivalence class; network/program graph labels checked separately'
        category='true_causal_correctness' if key!='neural_program' else 'network_fidelity'
        assumptions=['observational directions require identification premises','DAG completion and oracle diagnostics do not prove identification']
    elif requirement=='R10':
        target='frozen conditional neural mechanisms and inferred explicit SCM including graph, equations and exogenous laws'
        domain='entire declared mechanism/intervention domain for the inferred parent sets'
        interventions='all declared node-do values and compatible simultaneous interventions'
        error='neural-symbolic mechanism error <=1/100 of frozen training scale; joint intervention distribution distance requires certified noise coupling'
        category='network_fidelity' if key=='mechanism' else 'true_causal_correctness'
        assumptions=['correct parent sets or explicit conditional theorem','local Lipschitz and domain coverage','noise law, independence and coupling distance separately justified']
    elif requirement in ('R3','R12'):
        target='CDIR operators and frozen finite operator/constant/search grammar'
        domain='all branches and protected-operation boundaries in the declared semantics; all shorter candidates for a minimality claim'
        error='semantic equivalence; deterministic serialization does not imply semantic uniqueness'
        assumptions=['frozen operator and constant sets','explicit finite search bounds for class-relative MDL','unsupported operations remain unresolved']
        category='program_semantics'
    elif requirement in ('R11','R13'):
        target='all declared source, wheel, checkpoint, ablation and executable-program artifacts'
        domain='all registered release/ablation cases, including failures and isolated installed execution'
        error='identity and independent replay; passing reproduction does not establish causal truth'
        assumptions=['frozen hashes and requirements','external extension tasks require their own specifications']
        category='engineering_reproduction'
    if requirement=='R7':
        domain='entire declared closed input domain including branch boundaries and degenerate inputs'
        error='rigorous point counterexample or full-domain worst-case enclosure; no omitted hard regions'
        assumptions+=['strict rational numerical enclosures','search exhaustion is unresolved, not a universal refutation']
    if requirement=='R4':assumptions+=['candidate search reads only allowed data and frozen network','shorter candidates must be exhausted to claim bounded-language MDL minimality']
    return {'quantifiers':'all instances in the stated original scope; instance binding alone does not reduce this quantifier',
        'assumptions':assumptions,'target_network':target,'input_domain':domain,'interventions':interventions,
        'error_contract':error,'evidence_class':category,'proof_dependencies':['frozen target/data identities','declared domain and intervention contract','independent certificate or scope-matched counterexample']}


def initial_ledger(requirements_path,config=None):
    rows=[]
    for requirement,claims in CLAIMS.items():
        for key,statement in claims:
            rows.append({'id':requirement+'.'+key,'requirement':requirement,'statement':statement,
                'status':'unresolved',**_claim_contract(requirement,key),'evidence':[],
                'uncovered':'Historical module/test existence or a scoped certificate does not close this original scientific claim.'})
    return {'schema':'ncd.original-claims.v1','requirements_sha256':digest(requirements_path),
        'claims':rows,'target_instance_bindings':[{k:v for k,v in job.items() if k in ('id','kind','checkpoint','checkpoint_sha256','program','program_sha256','scm','scm_sha256','domain','epsilon','node')} for job in (config or {}).get('jobs',[])],'overall_objective_achieved':False}


def statistical_bound(units,delta='1/100',family_size=1):
    if type(family_size)is not int or family_size<1 or not 0<Q(delta)<1:raise ValueError('Invalid confidence contract')
    if not units or len({u['world_id'] for u in units})!=len(units):raise ValueError('Independent world identifiers required')
    values=[Q(u['success']) for u in units]
    if any(not 0<=v<=1 for v in values):raise ValueError('Success outside [0,1]')
    mean=sum(values)/len(values)
    local_delta=Q(delta)/family_size
    radius=(logarithm(Interval.point(1/local_delta))/(2*len(values))).sqrt()
    lower=max(Q(0),mean-radius.hi)
    return {'schema':'ncd.world-hoeffding.v1','status':'proved-conditionally',
        'units':units,'delta':str(Q(delta)),'family_size':family_size,'local_delta':str(local_delta),
        'mean':str(mean),'radius_enclosure':radius.to_dict(),'fidelity_lower':str(lower),
        'assumptions':['fixed hypothesis before final evaluation','independent bounded world units',
            'declared sampling law or fixed stratum mixture weights','at most family_size simultaneous claims'],
        'not_a_universal_equivalence_proof':True,'independence_proved_by_this_certificate':False}


def verify_statistical(c):
    if c!=statistical_bound(c['units'],c['delta'],c['family_size']):raise ValueError('Statistical bound mismatch')
    return {'status':'verified','conclusion':'proved-conditionally','premises_verified':False}


def _relative(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Artifact escapes proof directory')
    return p


def _job_result(root,job):
    kind=job['kind'];case=root/job['id'];case.mkdir(exist_ok=True)
    for field in ('checkpoint','scm','program'):
        if field in job and digest(job[field])!=job[field+'_sha256']:
            raise ValueError('Frozen source identity changed: '+field)
    if kind=='gaussian_nonidentifiability':
        certificate=certify_gaussian_nonidentifiability(job.get('coefficient','1/2'),job.get('do_value','1'))
    elif kind=='frozen_mechanism':
        network=export_mechanism(job['checkpoint'])
        equations=read_json(job['scm'])['equations']
        program=equations[job['node']]
        shutil.copy2(job['checkpoint'],case/'checkpoint.pt')
        save_json(case/'network.json',network);save_json(case/'program.json',program)
        shutil.copy2(job['scm'],case/'source_scm.json')
        certificate=certify_mechanism(network,program,job['domain'],job.get('epsilon','1/100'),
            max_boxes=job.get('max_boxes',256),seconds=job.get('seconds',43200))
        certificate.update(network_sha256=digest(case/'network.json'),program_sha256=digest(case/'program.json'),
                           source_scm_sha256=digest(case/'source_scm.json'),source_node=job['node'])
    elif kind=='mechanism_cegis':
        shutil.copy2(job['checkpoint'],case/'checkpoint.pt');shutil.copy2(job['scm'],case/'source_scm.json')
        certificate=run_cegis(case,case/'checkpoint.pt',case/'source_scm.json',job['node'],job['domain'],job.get('epsilon','1/100'),
            job.get('rounds',3),job.get('max_boxes',32),job.get('seconds',300),job.get('seed',8100))
    elif kind=='frozen_discovery':
        network=export_discoverer(job['checkpoint'])
        source=read_json(job['program'])
        if source.get('type')!='Rule':raise ValueError('Expected historical Rule')
        program=source['program']
        shutil.copy2(job['checkpoint'],case/'checkpoint.pt')
        shutil.copy2(job['program'],case/'source_program.json')
        save_json(case/'network.json',network);save_json(case/'program.json',program)
        certificate=certify_discovery_box(network,program,job['domain'])
        certificate.update(network_sha256=digest(case/'network.json'),program_sha256=digest(case/'program.json'),source_program_sha256=digest(case/'source_program.json'))
    elif kind=='conditional_scm_error':
        certificate=certify_scm_error(job['graph'],job['local_errors'],job['lipschitz'],job['noise_distances'],
            job.get('interventions',()),job.get('premises',[]))
    elif kind=='linear_interchange':certificate=certify_interchange(job['read'],job['write'],job['base_domain'],job['source_domain'],job['masks'],job.get('epsilon','1/100'),job.get('independent_sources',False))
    elif kind=='statistical_bound':certificate=statistical_bound(job['units'],job.get('delta','1/100'),job.get('family_size',1))
    else:raise ValueError('Unsupported proof job: '+kind)
    save_json(case/'certificate.json',certificate)
    return {'id':job['id'],'kind':kind,'certificate':job['id']+'/certificate.json',
        'certificate_sha256':digest(case/'certificate.json'),'conclusion':certificate['status']}


def verify_record(root,record):
    case=_relative(root,record['id']);path=_relative(root,record['certificate'])
    if digest(path)!=record['certificate_sha256']:raise ValueError('Certificate digest mismatch')
    certificate=read_json(path);kind=record['kind']
    if kind=='gaussian_nonidentifiability':result=verify_gaussian_nonidentifiability(certificate)
    elif kind=='frozen_mechanism':
        for name,key in (('network.json','network_sha256'),('program.json','program_sha256'),('source_scm.json','source_scm_sha256')):
            if digest(case/name)!=certificate[key]:raise ValueError('Fidelity dependency changed')
        network=read_json(case/'network.json');program=read_json(case/'program.json')
        if network!=export_mechanism(case/'checkpoint.pt'):raise ValueError('Frozen export differs from actual checkpoint')
        if program!=read_json(case/'source_scm.json')['equations'][certificate['source_node']]:raise ValueError('Expression was not extracted from bound historical SCM')
        result=verify_mechanism(network,program,certificate)
    elif kind=='mechanism_cegis':result=verify_cegis(case,certificate)
    elif kind=='frozen_discovery':
        for name,key in (('network.json','network_sha256'),('program.json','program_sha256'),('source_program.json','source_program_sha256')):
            if digest(case/name)!=certificate[key]:raise ValueError('Discovery dependency changed')
        network=read_json(case/'network.json');program=read_json(case/'program.json')
        if network!=export_discoverer(case/'checkpoint.pt'):raise ValueError('Discovery export differs from frozen checkpoint')
        if program!=read_json(case/'source_program.json')['program']:raise ValueError('Changed historical program')
        result=verify_discovery_box(network,program,certificate)
    elif kind=='conditional_scm_error':result=verify_scm_error(certificate)
    elif kind=='linear_interchange':result=verify_interchange(certificate)
    elif kind=='statistical_bound':result=verify_statistical(certificate)
    else:raise ValueError('Unknown proof backend')
    if result['conclusion']!=record['conclusion']:raise ValueError('False recorded conclusion')
    return result


def prove(config_path,resume=False):
    config_path=Path(config_path).resolve();config=read_json(config_path)
    base=(config_path.parent/config.get('root','.')).resolve()
    output=(base/config['output']).resolve()
    jobs=config['jobs']
    ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)) or any(not x or '/' in x or '\\' in x or x in ('.','..') for x in ids):raise ValueError('Invalid or duplicate job identifiers')
    if output.exists() and any(output.iterdir()) and not resume:raise FileExistsError('Use an empty proof output or --resume')
    output.mkdir(parents=True,exist_ok=True)
    frozen=output/'config.json'
    if frozen.exists() and read_json(frozen)!=config:raise ValueError('Resume config differs from frozen config')
    save_json(frozen,config)
    requirements=base/config.get('requirements','docs/FULL_REQUIREMENTS.md')
    if (output/'original_requirements.md').exists() and digest(requirements)!=digest(output/'original_requirements.md'):
        raise ValueError('Original requirements changed during resume')
    shutil.copy2(requirements,output/'original_requirements.md')
    records=[]
    for job in jobs:
        record_file=output/job['id']/'record.json'
        if resume and record_file.exists():
            row=read_json(record_file);verify_record(output,row)
        else:
            resolved=dict(job)
            for field in ('checkpoint','scm','program'):
                if field in resolved:resolved[field]=str((base/resolved[field]).resolve())
            row=_job_result(output,resolved);verify_record(output,row)
            save_json(record_file,row)
        records.append(row)
        save_json(output/'progress.json',{'completed_jobs':len(records),'declared_jobs':len(jobs),'records':records})
        print('proof',row['id'],row['conclusion'],flush=True)
    ledger=initial_ledger(output/'original_requirements.md',config)
    # Only the checked universal Gaussian impossibility discharges original claims.
    gaussian=next((r for r in records if r['kind']=='gaussian_nonidentifiability'),None)
    if gaussian:
        for claim in ledger['claims']:
            if claim['id'] in ('R0.observational_unique_direction','R0.uniform_finite_sample_direction'):
                claim.update(status='refuted',quantifiers='any observational estimator on the allowed linear Gaussian family',
                    assumptions=['independent centered Gaussian noises','no confounding','acyclic'],
                    input_domain='all sample sizes including population observations',interventions='none observed',
                    evidence=[gaussian['certificate']],uncovered='Additional identifying assumptions or informative interventions are separate claims.')
    save_json(output/'ledger.json',ledger)
    summary={'schema':'ncd.original-proof-bundle.v1','records':records,
        'artifact_replay_status':'verified','overall_objective_achieved':False,
        'declared_jobs_retained':len(records)==len(jobs),'ledger_sha256':digest(output/'ledger.json'),
        'config_sha256':digest(frozen)}
    save_json(output/'summary.json',summary)
    save_json(output/'manifest.json',{'files':{p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    return verify_proof(output)


def verify_proof(directory):
    root=Path(directory).resolve();summary=read_json(root/'summary.json');config=read_json(root/'config.json')
    for name,value in read_json(root/'manifest.json')['files'].items():
        if digest(_relative(root,name))!=value:raise ValueError('Proof artifact binding mismatch')
    records=summary['records']
    if len(records)!=len(config['jobs']) or [(r['id'],r['kind']) for r in records]!=[(j['id'],j['kind']) for j in config['jobs']]:raise ValueError('Declared proof jobs missing')
    results=[]
    for record,job in zip(records,config['jobs']):
        certificate=read_json(_relative(root,record['certificate']))
        kind=job['kind'];case=_relative(root,record['id'])
        if kind in ('frozen_mechanism','frozen_discovery','mechanism_cegis'):
            if certificate['domain']!=job['domain'] or digest(case/'checkpoint.pt')!=job['checkpoint_sha256']:
                raise ValueError('Frozen model or declared domain contract changed')
        if kind=='frozen_mechanism':
            if certificate['epsilon']!=str(Q(job.get('epsilon','1/100'))) or certificate['source_node']!=job['node'] or digest(case/'source_scm.json')!=job['scm_sha256']:
                raise ValueError('Historical mechanism contract changed')
            if certificate['normalizer']!=read_json(case/'network.json')['output_scale']:
                raise ValueError('Frozen training scale was replaced')
        elif kind=='mechanism_cegis':
            if (certificate['epsilon']!=str(Q(job.get('epsilon','1/100'))) or certificate['node']!=job['node']
                or digest(case/'source_scm.json')!=job['scm_sha256'] or certificate['round_budget']!=job.get('rounds',3)
                or certificate['seed']!=job.get('seed',8100) or certificate['max_boxes']!=job.get('max_boxes',32) or certificate['seconds']!=job.get('seconds',300)):
                raise ValueError('CEGIS frozen contract changed')
        elif kind=='frozen_discovery':
            if digest(case/'source_program.json')!=job['program_sha256']:raise ValueError('Historical discovery program changed')
        elif kind=='gaussian_nonidentifiability':
            if certificate!=certify_gaussian_nonidentifiability(job.get('coefficient','1/2'),job.get('do_value','1')):
                raise ValueError('Gaussian contract changed')
        elif kind=='conditional_scm_error':
            if certificate!=certify_scm_error(job['graph'],job['local_errors'],job['lipschitz'],job['noise_distances'],job.get('interventions',()),job.get('premises',[])):
                raise ValueError('Conditional SCM premises changed')
        elif kind=='linear_interchange':
            if certificate!=certify_interchange(job['read'],job['write'],job['base_domain'],job['source_domain'],job['masks'],job.get('epsilon','1/100'),job.get('independent_sources',False)):
                raise ValueError('Interchange contract changed')
        elif kind=='statistical_bound':
            if certificate!=statistical_bound(job['units'],job.get('delta','1/100'),job.get('family_size',1)):
                raise ValueError('Statistical sampling contract changed')
        results.append(verify_record(root,record))
    ledger=initial_ledger(root/'original_requirements.md',config)
    gaussian=next((r for r in records if r['kind']=='gaussian_nonidentifiability'),None)
    saved=read_json(root/'ledger.json')
    if gaussian:
        for claim in ledger['claims']:
            if claim['id'] in ('R0.observational_unique_direction','R0.uniform_finite_sample_direction'):
                claim.update(status='refuted',quantifiers='any observational estimator on the allowed linear Gaussian family',
                    assumptions=['independent centered Gaussian noises','no confounding','acyclic'],
                    input_domain='all sample sizes including population observations',interventions='none observed',
                    evidence=[gaussian['certificate']],uncovered='Additional identifying assumptions or informative interventions are separate claims.')
    if saved!=ledger:raise ValueError('Original claim ledger differs from independently derived closure')
    if saved['requirements_sha256']!=ledger['requirements_sha256'] or saved['overall_objective_achieved'] or summary['overall_objective_achieved']:raise ValueError('False whole-project completion')
    if digest(root/'ledger.json')!=summary['ledger_sha256'] or digest(root/'config.json')!=summary['config_sha256']:raise ValueError('Summary binding mismatch')
    return {'status':'verified','jobs':len(results),'results':results,'overall_objective_achieved':False}


def audit_requirements(directory):
    root=Path(directory).resolve();verify_proof(root)
    ledger=read_json(root/'ledger.json');claims=ledger['claims']
    groups={r:{'proved':0,'refuted':0,'unresolved':0} for r in CLAIMS}
    for claim in claims:groups[claim['requirement']][claim['status']]+=1
    return {'schema':'ncd.original-requirements-audit.v1','requirements':groups,
        'claim_count':len(claims),'unresolved_claims':[c['id'] for c in claims if c['status']=='unresolved'],
        'overall_objective_achieved':all(c['status'] in ('proved','refuted') for c in claims)}
