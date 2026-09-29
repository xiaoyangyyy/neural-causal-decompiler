"""Isolated exact identification-boundary replay; no caller scope promotion."""
from pathlib import Path
import hashlib,json,sys
APPROVED_SCOPE={'arithmetic': 'exact rational model parameters; ideal real SCM and Gaussian laws', 'canonical_coefficient_magnitude': ['2/5', '11/10'], 'estimator_family': 'all measurable estimators, including randomized estimators', 'information': 'only IID observational rows, no true graph/scales/noises/do data', 'known_math_is_not_project_innovation': True, 'nodes': [3, 5, 8], 'noise': 'independent centered Gaussian, common standard deviation 7/20', 'not_claimed': ['hardware floating execution', 'exact rounded generator realization', 'almost-sure or 99-percent random-world performance', 'all estimators given interventions', 'failure under a justified positive strong-faithfulness floor', 'original R0-R13 closure'], 'observed_scales': 'inside exp([-3/2,3/2]); unknown to observational estimator', 'target': 'one correct graph Markov equivalence class (CPDAG); abstention counts as nonrecovery'}
APPROVED_CONTRACTS={'population_n3': '8a5764123bb63052cc1555f0fc9b462426f9898218f94c49436fa0df0a021582', 'faithful_finite_sample_n3': '0bcecda42c631b9ea78592980a1208acbc0809ab71d2afb5744d5ec8a693461e', 'population_n5': '57fb9b6096240d86029fdc55ca4a9643d9ead34ef57ef27a1af916a9ff3314f4', 'faithful_finite_sample_n5': 'e9b8193a8bf3604961eb819a526e18328c31252c58469dc21d71e3af2a211b12', 'population_n8': '8cea542aef672ea00bf9b5c78411fcd0afb65b67a5f5bb933a116187fd52f639', 'faithful_finite_sample_n8': '0ce8bd8d9d0c0b5901d9ae879398307c8c7f40c53845463eb32db227199f4244'}
APPROVED_SNAPSHOT='f4bcf2016b85795025f72e77257ed2f2783cf77b15a55eb2a516977e78f485d7'


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def unresolved_contracts(reason):
    return [{'id':key,'status':'unresolved','evidence_class':'true_causal_correctness',
        'scope':APPROVED_SCOPE,'original_claim_entailment':False,'reason':reason,
        'independent_counterexample_verified':False,'uncovered':APPROVED_SCOPE['not_claimed']}
        for key in sorted(APPROVED_CONTRACTS)]


def main():
    request=read(sys.argv[1]);snapshot=Path(request['snapshot']).resolve();env=Path(request['environment']).resolve()
    if request.get('schema')!='ncd.identification-proof-request.v1' or Path.cwd().resolve()!=snapshot or not Path(sys.executable).resolve().is_relative_to(env):
        raise ValueError('Independent identification context mismatch')
    import faithfulness_boundary_proof as kernel
    from importlib.metadata import version
    installed=Path(kernel.__file__).resolve().parent
    if not installed.is_relative_to(env) or version('ncd-faithfulness-boundary-proof')!='0.1.0':raise ValueError('Wrong installed identification kernel')
    actual={'faithfulness_boundary_proof/'+p.name:sha(p) for p in installed.glob('*.py')}
    if len(actual)!=3 or actual!=request['source_sha256']:raise ValueError('Installed identification sources changed')
    if any(sha(snapshot/name)!=expected for name,expected in actual.items()):raise ValueError('Identification snapshot changed')
    job=request['job']
    if job['kind']!='observational_cpdag_boundary' or job['identification_scope']!=APPROVED_SCOPE or job['original_claim_closed'] is not False:
        raise ValueError('Identification job scope changed')
    from faithfulness_boundary_proof.__main__ import replay
    manifest=Path(job['artifact']);verification=replay(manifest.parent)
    rows=read(manifest.parent/'scoped_ledger.json')
    if len(rows)!=6 or {r['id'] for r in rows}!=set(APPROVED_CONTRACTS):raise ValueError('Identification cases missing')
    for row in rows:
        if fingerprint(row)!=APPROVED_CONTRACTS[row['id']]:raise ValueError('Accepted identification contract changed')
    result={'status':'verified','conclusion':'verified','kernel':'faithfulness_boundary_proof','case_count':6,
        'identification_contracts':rows,'independent_bundle_verification':verification,
        'original_claim_closed':False,'original_objective_achieved':False}
    Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
