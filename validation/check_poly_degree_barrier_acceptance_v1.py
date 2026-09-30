"""Strict acceptance of one degree-six polynomial exclusion."""
from fractions import Fraction as Q
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import xml.etree.ElementTree as ET

R = Path(__file__).resolve().parents[1]
V = R / 'validation'
B = R / 'runs/frozen_mechanism_polynomial_barrier_v1'
P = V / 'poly_degree_barrier_protocol_v1.json'

def h(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def j(path):
    return json.loads(Path(path).read_bytes())

def main():
    p = j(P)
    if not (p['schema'] == 'ncd.polynomial-degree-barrier-plan.v1'):
        raise ValueError('Acceptance condition failed')
    if not ((p['degree_bound'], p['epsilon_normalized'], p['input_domain']) == (6, '1/100', [['-1', '1']]*3)):
        raise ValueError('Acceptance condition failed')
    if not (p['original_atomic_claim_counts'] == {'proved': 0, 'refuted': 2, 'unresolved': 36}):
        raise ValueError('Acceptance condition failed')
    if not (p['original_objective_achieved'] is False):
        raise ValueError('Acceptance condition failed')
    points = [str(Q(-37,40)+i*Q(11,40)) for i in range(8)]
    weights = [(-1)**(7-i)*comb(7,i) for i in range(8)]
    if not (p['points'] == points):
        raise ValueError('Acceptance condition failed')
    for name, digest in p['loaded_package_sha256'].items():
        if not (h(V/'poly_degree_barrier_env_v1/Lib/site-packages/poly_degree_barrier'/name) == digest):
            raise ValueError('Acceptance condition failed')
        if not (h(V/'poly_degree_barrier_package_v1/poly_degree_barrier'/name) == digest):
            raise ValueError('Acceptance condition failed')
    m = j(B/'manifest.json')
    if not (m['schema'] == 'ncd.polynomial-degree-barrier-bundle.v1'):
        raise ValueError('Acceptance condition failed')
    if not (set(m['files']) == {'protocol.json', 'certificate.json', 'source/mechanism_0.pt'}):
        raise ValueError('Acceptance condition failed')
    if not (m['original_objective_achieved'] is False):
        raise ValueError('Acceptance condition failed')
    for name, digest in m['files'].items():
        if not (h(B/name) == digest):
            raise ValueError('Acceptance condition failed')
    if not (h(B/'protocol.json') == h(P)):
        raise ValueError('Acceptance condition failed')
    if not (h(B/'source/mechanism_0.pt') == p['checkpoint_sha256']):
        raise ValueError('Acceptance condition failed')
    c = j(B/'certificate.json')
    if not (c['schema'] == 'ncd.frozen-polynomial-degree-barrier.v1'):
        raise ValueError('Acceptance condition failed')
    if not ((c['status'],c['checkpoint_sha256'],c['degree_bound']) == ('refuted-scoped',p['checkpoint_sha256'],6)):
        raise ValueError('Acceptance condition failed')
    if not ((c['points'],c['difference_weights']) == (points,weights)):
        raise ValueError('Acceptance condition failed')
    if not (c['original_claim_closed'] is False and c['original_objective_achieved'] is False):
        raise ValueError('Acceptance condition failed')
    intervals = c['network_point_enclosures']
    if not (len(intervals) == 8 and all(Q(z[0]) <= Q(z[1]) for z in intervals)):
        raise ValueError('Acceptance condition failed')
    lo = sum(w*Q(z[0] if w>=0 else z[1]) for w,z in zip(weights,intervals))
    hi = sum(w*Q(z[1] if w>=0 else z[0]) for w,z in zip(weights,intervals))
    sep = lo if lo>0 else -hi if hi<0 else Q(0)
    budget = Q(c['output_training_scale'])*Q(128,100)
    margin = sep-budget
    if not (margin>0 and c['seventh_difference_enclosure']==[str(lo),str(hi)]):
        raise ValueError('Acceptance condition failed')
    if not (Q(c['seventh_difference_absolute_lower'])==sep):
        raise ValueError('Acceptance condition failed')
    if not (Q(c['all_point_error_difference_upper'])==budget):
        raise ValueError('Acceptance condition failed')
    if not (Q(c['strict_incompatibility_margin_lower'])==margin):
        raise ValueError('Acceptance condition failed')
    expected = {'status':'verified-scoped','conclusion':'refuted-scoped','degree_bound':6,'points':8,'original_objective_achieved':False}
    stages = {}
    for mode in ('prove','verify'):
        path = V/('poly_degree_barrier_stage_v1_'+mode+'_attempt0000.json')
        row = j(path)
        guard = row['resource_guard']
        if not ((row['mode'],row['status'],row['exit_code'],row['result']) == (mode,'verified-scoped',0,expected)):
            raise ValueError('Acceptance condition failed')
        if not (row['protocol_sha256']==h(P) and row['bundle_manifest_sha256']==h(B/'manifest.json')):
            raise ValueError('Acceptance condition failed')
        if not (row['runner_sha256']==h(V/'run_poly_degree_barrier_stage_v1.py')):
            raise ValueError('Acceptance condition failed')
        if not (row['artifact_bytes']<=p['artifact_budget_bytes']):
            raise ValueError('Acceptance condition failed')
        if not (not guard['timeout'] and guard['descendants_included']):
            raise ValueError('Acceptance condition failed')
        if not (guard['active_processes_on_return']==0 and guard['peak_job_memory_bytes']<=p['memory_bytes']):
            raise ValueError('Acceptance condition failed')
        stages[mode]=h(path)
    junit = V/'pytest_poly_degree_barrier_source_v1.xml'
    suite = ET.parse(junit).getroot()[0]
    if not (tuple(suite.attrib[k] for k in ('tests','failures','errors','skipped'))==('3','0','0','0')):
        raise ValueError('Acceptance condition failed')
    wheel = V/'poly_degree_barrier_wheels_v1/ncd_poly_degree_barrier-0.1.0-py3-none-any.whl'
    result = {'schema':'ncd.polynomial-degree-barrier-acceptance.v1','status':'accepted-scoped',
              'conclusion':'refuted-scoped','degree_bound':6,'checked_points':8,
              'strict_margin_lower':str(margin),'protocol_sha256':h(P),
              'bundle_manifest_sha256':h(B/'manifest.json'),'installed_wheel_sha256':h(wheel),
              'source_tests_junit_sha256':h(junit),'stage_record_sha256':stages,
              'original_claim_counts':p['original_atomic_claim_counts'],'original_objective_achieved':False}
    target = V/'poly_degree_barrier_acceptance_v1.json'
    if target.exists():
        if j(target) != result:
            raise ValueError('Immutable acceptance receipt differs')
    else:
        target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'accepted-scoped','degree_bound':6,'original_objective_achieved':False},sort_keys=True))

if __name__ == '__main__':
    main()
