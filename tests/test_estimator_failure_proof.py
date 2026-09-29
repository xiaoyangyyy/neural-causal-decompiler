from pathlib import Path
from copy import deepcopy
from fractions import Fraction as Q
import json,pytest
from estimator_failure_proof.boundary import certify,verify
from estimator_failure_proof.__main__ import prove,verify_bundle,read,save,check_config
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('n,k',[(1,1),(96,100)])
def test_strict_failure_bound_for_exact_same_gaussian_estimator(n,k):
 c=certify(n,k);r=verify(c)
 assert r['requested_recovery_success_probability_gate']=='refuted'
 assert Q(r['family_failure_strict_lower'])>Q(1,100)
 assert c['base_certificate']['estimator']['statistic']=='T=mean(X0*(X1-X2)); return edge01 iff sign(a)*T>=0, else edge02'
 assert not r['all_estimators_refuted'] and not r['original_claim_closed']
 if n==1:assert r['family_failure_strict_lower']=='3/256'
 else:
  assert Q(1,5000)<Q(r['per_event_error_strict_lower'])<Q(1,4000)
  assert Q(1,100)<Q(r['family_failure_strict_lower'])<Q(3,100)

@pytest.mark.parametrize('change',['density','conditional_variance','e_tail','sample_scope','family_scope','failure_bound','independence','all_estimators','neural_target','six_events'])
def test_false_lower_bound_or_stronger_refutation_is_rejected(change):
 c=deepcopy(certify(96,100))
 if change=='density':c['primitive_bounds']['rectangle_density_strict_lower']='1'
 elif change=='conditional_variance':c['primitive_bounds']['conditional_c_squared']='1/4'
 elif change=='e_tail':c['primitive_bounds']['e_series_tail_upper']='0'
 elif change=='sample_scope':c['samples']=95
 elif change=='family_scope':c['family_size']=6
 elif change=='failure_bound':c['family_failure_strict_lower']='1'
 elif change=='independence':c['counterexample_joint_law']['between_events']='shared dataset'
 elif change=='all_estimators':c['all_estimators_refuted']=True
 elif change=='neural_target':c['actual_neural_graph_recovery_refuted']=True
 else:c['six_event_96_row_theorem_refuted']=True
 with pytest.raises(ValueError):verify(c)


def test_dependency_covariance_and_identity_cannot_be_changed():
 c=deepcopy(certify());c['base_certificate']['models'][0]['covariance'][0][0]='2'
 with pytest.raises(ValueError):verify(c)


def test_successful_six_event_scope_cannot_be_refuted_by_this_kernel():
 with pytest.raises(ValueError):certify(96,6)
 with pytest.raises(ValueError):certify(96,100,'1/10')


def test_portable_bundle_replay_overwrite_and_record_tampering(tmp_path):
 config=read(ROOT/'validation/estimator_failure_protocol_v1.json')
 folder=tmp_path/'validation';folder.mkdir();config['output']='bundle'
 path=folder/'config.json';save(path,config)
 first=prove(path);assert first==verify_bundle(tmp_path/'bundle/manifest.json')
 assert first==prove(path,resume=True)
 with pytest.raises(FileExistsError):prove(path)
 cert=read(tmp_path/'bundle/certificates.json');cert.pop('single_row_99_percent');save(tmp_path/'bundle/certificates.json',cert)
 with pytest.raises(ValueError):verify_bundle(tmp_path/'bundle/manifest.json')


def test_incomplete_loaded_source_binding_is_rejected(tmp_path):
 config=read(ROOT/'validation/estimator_failure_protocol_v1.json');config['source_sha256'].pop(next(iter(config['source_sha256'])))
 path=tmp_path/'bad.json';save(path,config)
 with pytest.raises(ValueError):check_config(path)
