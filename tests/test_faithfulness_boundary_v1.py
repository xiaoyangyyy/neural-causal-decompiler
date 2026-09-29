from pathlib import Path
from fractions import Fraction as Q
import copy,hashlib,importlib,json,os,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
if os.environ.get('NCD_FAITHFULNESS_TEST_SOURCE')=='1':
    sys.path.insert(0,str(ROOT/'validation/faithfulness_boundary_package_v1'))
from faithfulness_boundary_proof.boundary import (construct,verify,structural,invert_det,do_means,
    conditional_numerators,model,equivalence,validate,SCOPE,REQUIREMENTS_SHA,GENERATOR_SHA)
from faithfulness_boundary_proof.__main__ import generate,replay


@pytest.mark.parametrize('n',[3,5,8])
@pytest.mark.parametrize('mode',['population','faithful-finite-sample'])
def test_declared_certificates_recompute(n,mode):
    c=construct(n,96,'1/100',mode);result=verify(c)
    assert result['conclusion']=='refuted' and result['original_objective_achieved'] is False
    assert c['triangle_equivalence']['dag_count']==6 and c['collider_equivalence']['dag_count']==1
    assert c['triangle_equivalence']['cpdag'][0][2]==1 and c['collider_equivalence']['cpdag'][0][2]==0
    assert c['intervention']['triangle_mean'][1]=='0' and c['intervention']['collider_mean'][1]=='12/25'
    assert (c['bounds']['triangle_faithful'])==(mode!='population')


@pytest.mark.parametrize('N,delta',[(1,'1/100'),(96,'1/100'),(512,'1/100'),(768,'1/100'),
    (10**9,'1/100'),(1,'499/1000'),(10000,'1/1000000'),(96,'1/4')])
def test_faithful_nonuniform_bound_for_any_estimator(N,delta):
    c=construct(3,N,delta,'faithful-finite-sample');verify(c)
    t=Q(c['perturbation']);lower=Q(c['bounds']['finite_sample_minimax_error_lower'])
    assert 0<t<Q(1,16) and lower>Q(delta)
    assert Q(c['bounds']['iid_KL'])==N*t*t/2
    assert Q(c['bounds']['iid_KL'])<=Q(c['bounds']['total_variation_upper'])**2
    assert all(Q(row['unconditional']) and Q(row['conditional_numerator']) for row in c['gaussian_CI_triangle'])
    # Large structural coefficients do not imply a separation margin.
    assert all(abs(Q(x))>Q(1,2) for row in c['triangle']['weights'] for x in row if Q(x))


@pytest.mark.parametrize('n',[3,5,8])
def test_known_population_covariance_independent_structural_formula(n):
    c=construct(n);A,_,_=structural(c['triangle']);B,_,_=structural(c['collider'])
    sig2=Q(49,400);known=[[Q(1),Q(3,4),Q(0)],[Q(3,4),Q(25,16),Q(3,4)],[Q(0),Q(3,4),Q(25,16)]]
    assert A==B
    for i in range(n):
        for j in range(n): assert A[i][j]==sig2*(known[i][j] if i<3 and j<3 else Q(i==j))
    inv,det=invert_det(A)
    assert det==sig2**n
    assert [[sum(A[i][k]*inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]==[[Q(i==j) for j in range(n)] for i in range(n)]


@pytest.mark.parametrize('target,variable,expected_a,expected_b',[(0,1,'3/4','3/4'),(1,2,'3/4','0'),(2,1,'0','12/25')])
def test_distinguishing_and_nondistinguishing_interventions(target,variable,expected_a,expected_b):
    c=construct();assert do_means(c['triangle'],target,Q(1))[variable]==expected_a
    assert do_means(c['collider'],target,Q(1))[variable]==expected_b


MUTATIONS=[('status','proved'),('mode','other'),('nodes',4),('samples',0),('requested_error_upper','0'),
    ('perturbation','1/8'),('requirements_sha256','0'*64),('generator_sha256','0'*64),
    ('original_objective_achieved',True),('one_row_KL','1'),('determinant','1'),('scope',{}),
    ('covariance_triangle',[]),('covariance_collider',[]),('triangle_equivalence',{}),
    ('collider_equivalence',{}),('gaussian_CI_triangle',[]),('gaussian_CI_collider',[]),
    ('intervention',{}),('bounds',{}),('proof_dependencies',[]),('uncovered',[]),
    ('evidence_class','network_fidelity')]
@pytest.mark.parametrize('field,value',MUTATIONS)
def test_tampered_scientific_claims_rejected(field,value):
    c=construct();c[field]=value
    with pytest.raises((ValueError,KeyError,TypeError,ZeroDivisionError)):verify(c)


@pytest.mark.parametrize('field',['weights','graph','noise_std','scales','noise_law','latent_confounding'])
def test_wrong_structural_model_or_noise_contract_rejected(field):
    c=construct();c['collider'][field]=[] if field not in ('noise_law','latent_confounding') else 'wrong'
    with pytest.raises(ValueError):verify(c)


@pytest.mark.parametrize('n,N,delta',[(True,96,'1/100'),(2,96,'1/100'),(100000000,96,'1/100'),
    (3,True,'1/100'),(3,0,'1/100'),(3,10**9+1,'1/100'),(3,96,0.01),
    (3,96,True),(3,96,'-1/10'),(3,96,'1/2')])
def test_invalid_budget_and_probability_rejected(n,N,delta):
    with pytest.raises(ValueError):construct(n,N,delta)


def test_adding_or_deleting_certificate_field_rejected():
    c=construct();c['entails_original_claim']=True
    with pytest.raises(ValueError):verify(c)
    c=construct();del c['uncovered']
    with pytest.raises(ValueError):verify(c)


def protocol(tmp_path):
    root=tmp_path/'project';(root/'docs').mkdir(parents=True);(root/'ncd').mkdir()
    (root/'docs/FULL_REQUIREMENTS.md').write_bytes((ROOT/'docs/FULL_REQUIREMENTS.md').read_bytes())
    (root/'ncd/multiverse.py').write_bytes((ROOT/'ncd/multiverse.py').read_bytes())
    package=Path(importlib.import_module('faithfulness_boundary_proof').__file__).resolve().parent
    c={'schema':'ncd.faithfulness-boundary-plan.v1','project_root':str(root),'output':'proof',
        'requirements_sha256':REQUIREMENTS_SHA,'generator_sha256':GENERATOR_SHA,
        'loaded_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')},
        'original_objective_achieved':False,
        'jobs':[{'id':mode.replace('-','_')+'_n'+str(n),'nodes':n,'samples':96,'delta':'1/100','mode':mode}
            for n in (3,5,8) for mode in ('population','faithful-finite-sample')]}
    path=root/'config.json';path.write_text(json.dumps(c),encoding='utf-8');return root,path


def test_portable_bundle_independent_replay_and_retention(tmp_path,monkeypatch):
    root,path=protocol(tmp_path);result=generate(path);assert result['jobs']==6
    foreign=tmp_path/'foreign';foreign.mkdir();monkeypatch.chdir(foreign)
    assert replay(root/'proof')==result
    with pytest.raises(FileExistsError):generate(path)


@pytest.mark.parametrize('name',['population_n3.json','faithful_finite_sample_n8.json','scoped_ledger.json','generator.py','requirements.md','protocol.json'])
def test_bundle_tamper_detected(tmp_path,name):
    root,path=protocol(tmp_path);generate(path);p=root/'proof'/name;p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):replay(root/'proof')


def test_omitted_job_cannot_pass_even_after_rehash(tmp_path):
    root,path=protocol(tmp_path);generate(path);folder=root/'proof';p=folder/'protocol.json'
    c=json.loads(p.read_text(encoding='utf-8'));c['jobs'].pop();p.write_text(json.dumps(c),encoding='utf-8')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['protocol.json']=hashlib.sha256(p.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):replay(folder)
