from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from ncd.io import read_json
from neural_boundary_proof.boundary import certify,verify,sample_invariance,branch_identity
from neural_ssa_proof.compiler import target
from neural_ssa_proof.runtime import NumericProgram

@pytest.fixture(scope='module')
def proof():
 base=Path('runs/full_neural_ssa_realization_v1/discoverer91');r=read_json(base/'certificate.json');p=read_json(base/'program.json');return r,p,certify(r,p)

def test_reachable_strict_swap_counterexample(proof):
 r,p,c=proof;v=verify(c,r,p)
 assert v['target_reachability_proved'] and v['conclusion']=='refuted'
 assert c['label_on_input']==c['label_on_swapped_input']==0 and c['required_swapped_label']==1
 assert not v['score_equivariance_refuted'] and not c['original_claim_closed_by_this_bundle']

def test_certificate_tampering_rejected(proof):
 r,p,c=proof;q=deepcopy(c);q['label_on_swapped_input']=1
 with pytest.raises(ValueError,match='mismatch'):verify(q,r,p)

def test_wrong_dependency_weights_rejected(proof):
 r,p,c=proof;q=deepcopy(r);q['target_sha256']='0'*64
 with pytest.raises(ValueError,match='mismatch'):verify(c,q,p)

def test_branch_structure_requires_shared_computation(proof):
 r,p,c=proof;t=target(r['checkpoint']);q=deepcopy(t)
 next(n for n in q['fx']['nodes'] if n['name']=='head_3')['args']=[{'node':'cat'}]
 with pytest.raises(ValueError,match='branch'):branch_identity(q,p)

def test_row_invariance_relation_and_numeric_diagnostic(proof):
 r,p,c=proof;t=sample_invariance(p);assert t['operators']['argmax']==1
 runtime=NumericProgram(p);data=np.random.default_rng(8100).normal(size=(24,2));perm=np.random.default_rng(8100).permutation(24)
 a,b=runtime.run(data),runtime.run(data[perm])
 assert np.allclose(a['logits'],b['logits'],rtol=0,atol=1e-12)
 for sid,spec in p['sites'].items():assert np.allclose(np.asarray(a['states'][sid])[perm] if spec['kind']=='vector' else a['states'][sid],b['states'][sid],rtol=1e-10,atol=1e-12)
