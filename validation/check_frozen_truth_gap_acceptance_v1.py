"""Accept a separately verified fixed-instance neural/true-root gap."""
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'validation'
BUNDLE = ROOT / 'runs/frozen_three_node_truth_gap_v1'
COMPOSITION = ROOT / 'runs/frozen_three_node_composition_v1'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    plan_path = V / 'frozen_truth_gap_protocol_v1.json'
    accepted_path = V / 'frozen_three_node_acceptance_v1.json'
    plan = read(plan_path)
    accepted = read(accepted_path)
    if plan['schema'] != 'ncd.frozen-truth-gap-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong truth-only scope')
    if plan['original_atomic_claim_counts'] != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Original atomic claims were improperly closed')
    if accepted['status'] != 'accepted-scoped' or accepted['original_objective_achieved'] is not False:
        raise ValueError('Neural candidate was not accepted')
    if plan['composition_acceptance_sha256'] != digest(accepted_path):
        raise ValueError('Candidate acceptance changed')
    if plan['composition_manifest_sha256'] != accepted['bundle_manifest_sha256'] or plan['composition_manifest_sha256'] != digest(COMPOSITION / 'manifest.json'):
        raise ValueError('Candidate bundle changed')
    installed = V / 'frozen_three_node_env_v1/Lib/site-packages/frozen_truth_gap'
    for name, expected in plan['loaded_package_sha256'].items():
        if digest(installed / name) != expected:
            raise ValueError('Independent installed truth evaluator changed')
    stages = {}
    for mode in ('prove', 'verify'):
        path = V / ('frozen_truth_gap_stage_v1_%s_attempt0000.json' % mode)
        row = read(path)
        if row['mode'] != mode or row['status'] != 'verified-scoped' or row['exit_code'] != 0:
            raise ValueError('Independent %s stage has not passed' % mode)
        resources = row['resource_guard']
        if resources['timeout'] or not resources['descendants_included'] or resources['active_processes_on_return'] != 0 or resources['peak_job_memory_bytes'] > plan['memory_bytes']:
            raise ValueError('Independent %s resource guard failed' % mode)
        if row['runner_sha256'] != digest(V / 'run_frozen_truth_gap_v1.py'):
            raise ValueError('Truth stage runner code changed')
        if row['protocol_sha256'] != digest(plan_path) or row['acceptance_sha256'] != digest(accepted_path):
            raise ValueError('Independent stage used a different frozen protocol')
        if row['bundle_manifest_sha256'] != digest(BUNDLE / 'manifest.json') or row['artifact_bytes'] > plan['artifact_budget_bytes']:
            raise ValueError('Independent stage used a different truth bundle')
        if row['result'] != {'status': 'verified-scoped', 'conclusion': 'refuted-scoped',
                             'strict_root_gaps': 2, 'original_objective_achieved': False}:
            raise ValueError('Independent verifier conclusion changed')
        stages[mode] = digest(path)
    manifest = read(BUNDLE / 'manifest.json')
    expected = {'protocol.json', 'certificate.json', 'source/historical_manifest.json',
                'source/world.json', 'source/composition_acceptance.json'}
    if manifest['schema'] != 'ncd.frozen-truth-gap-bundle.v1' or set(manifest['files']) != expected or manifest['original_objective_achieved'] is not False:
        raise ValueError('Truth diagnostic bundle is incomplete')
    for relative, expected_sha in manifest['files'].items():
        if digest(BUNDLE / relative) != expected_sha:
            raise ValueError('Truth diagnostic artifact changed')
    if digest(BUNDLE / 'protocol.json') != digest(plan_path):
        raise ValueError('Truth bundle used a different plan')
    if digest(BUNDLE / 'source/composition_acceptance.json') != digest(accepted_path):
        raise ValueError('Truth bundle used a different candidate acceptance')
    if digest(BUNDLE / 'source/world.json') != plan['world_sha256'] or digest(BUNDLE / 'source/historical_manifest.json') != plan['historical_manifest_sha256']:
        raise ValueError('Truth source changed')
    certificate = read(BUNDLE / 'certificate.json')
    if certificate['status'] != 'refuted-scoped' or certificate['original_claim_closed'] is not False or certificate['original_objective_achieved'] is not False:
        raise ValueError('Truth diagnostic scope changed')
    if certificate['learned_graph_equals_true_graph_for_this_instance'] is not True or set(certificate['roots']) != {'1', '2'}:
        raise ValueError('Wrong fixed-instance root claim')
    for root in certificate['roots'].values():
        if Fraction(root['two_target_incompatibility_margin_lower']) <= 0:
            raise ValueError('No strict root incompatibility')
    suite = ET.fromstring((V / 'pytest_frozen_truth_gap_source_v1.xml').read_bytes())[0]
    if (suite.attrib['tests'], suite.attrib['failures'], suite.attrib['errors'], suite.attrib['skipped']) != ('2', '0', '0', '0'):
        raise ValueError('Truth diagnostic source tests failed')
    wheel = V / 'frozen_truth_gap_wheels_v1_r2/ncd_frozen_truth_gap-0.1.0-py3-none-any.whl'
    result = {
        'schema': 'ncd.frozen-truth-gap-acceptance.v1', 'status': 'accepted-scoped-refutation',
        'protocol_sha256': digest(plan_path), 'composition_acceptance_sha256': digest(accepted_path),
        'bundle_manifest_sha256': digest(BUNDLE / 'manifest.json'),
        'stage_record_sha256': stages, 'source_tests_junit_sha256': digest(V / 'pytest_frozen_truth_gap_source_v1.xml'),
        'installed_wheel_sha256': digest(wheel), 'strict_root_gaps': 2,
        'original_claim_counts': plan['original_atomic_claim_counts'], 'original_objective_achieved': False,
    }
    target = V / 'frozen_truth_gap_acceptance_v1.json'
    if target.exists():
        raise FileExistsError('Truth diagnostic acceptance is immutable')
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'strict_root_gaps': 2,
                      'original_objective_achieved': False}, sort_keys=True))


if __name__ == '__main__':
    main()
