"""Strict acceptance gate for the scoped frozen three-node composition."""
from hashlib import sha256
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'runs/frozen_three_node_composition_v1'
VALIDATION = ROOT / 'validation'


def raw(path):
    return Path(path).read_bytes()


def digest(path):
    return sha256(raw(path)).hexdigest()


def read(path):
    return json.loads(raw(path))


def main():
    plan_path = VALIDATION / 'frozen_three_node_protocol_v1.json'
    preflight_path = VALIDATION / 'frozen_three_node_preflight_v1.json'
    plan = read(plan_path)
    preflight = read(preflight_path)
    if plan['schema'] != 'ncd.frozen-three-node-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong frozen scope')
    if plan['original_atomic_claim_counts'] != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Original claims improperly promoted')
    if preflight['status'] != 'verified-preflight' or preflight['loaded_package_sha256'] != plan['loaded_package_sha256']:
        raise ValueError('Preflight does not bind installed proof code')
    if preflight['source_sha256'] != plan['source_sha256'] or preflight['compatible_intervention_masks'] != 8:
        raise ValueError('Preflight source or intervention coverage changed')
    for name, expected_sha in plan['loaded_package_sha256'].items():
        if digest(VALIDATION / 'frozen_three_node_env_v1/Lib/site-packages/frozen_three_node' / name) != expected_sha:
            raise ValueError('Installed composition checker changed')
    stages = {}
    for mode in ('prove', 'verify'):
        path = VALIDATION / ('frozen_three_node_stage_v1_%s_attempt0000.json' % mode)
        row = read(path)
        if row['mode'] != mode or row['status'] != 'verified-scoped' or row['exit_code'] != 0:
            raise ValueError('Isolated %s stage did not pass' % mode)
        resources = row['resource_guard']
        if resources['timeout'] or not resources['descendants_included'] or resources['active_processes_on_return'] != 0 or resources['peak_job_memory_bytes'] > plan['memory_bytes']:
            raise ValueError('Isolated %s resource guard failed' % mode)
        if row['runner_sha256'] != digest(VALIDATION / 'run_frozen_three_node_stage_v1.py'):
            raise ValueError('Stage runner code changed')
        if row['protocol_sha256'] != digest(plan_path) or row['preflight_sha256'] != digest(preflight_path):
            raise ValueError('Stage used a different frozen proof protocol')
        if row['bundle_manifest_sha256'] != digest(BUNDLE / 'manifest.json') or row['artifact_bytes'] > plan['artifact_budget_bytes']:
            raise ValueError('Stage bundle or artifact budget changed')
        if row['result'] != {'status': 'verified-scoped', 'conclusion': 'proved-scoped', 'network_nodes': 3, 'compatible_intervention_masks': 8, 'worst_joint_l1_error_upper': preflight['worst_joint_l1_error_upper'], 'original_objective_achieved': False}:
            raise ValueError('Stage scope changed')
        stages[mode] = digest(path)
    manifest = read(BUNDLE / 'manifest.json')
    if manifest['schema'] != 'ncd.frozen-three-node-bundle.v1' or manifest['original_objective_achieved'] is not False:
        raise ValueError('Wrong proof bundle')
    expected = {'protocol.json', 'root_certificates.json', 'composition_certificate.json', 'program.json'} | {
        'source/' + name for name in ('child_manifest.json', 'child_protocol.json',
            'child_certificate.json', 'child_program.json', 'mechanism_0.pt',
            'mechanism_1.pt', 'mechanism_2.pt')}
    if set(manifest['files']) != expected:
        raise ValueError('Missing or extra proof artifact')
    for relative, expected_sha in manifest['files'].items():
        if digest(BUNDLE / relative) != expected_sha:
            raise ValueError('Changed proof artifact: ' + relative)
    if digest(BUNDLE / 'protocol.json') != digest(plan_path):
        raise ValueError('Proof bundle used another plan')
    composition = read(BUNDLE / 'composition_certificate.json')
    if composition['status'] != 'proved-scoped' or composition['original_claim_closed'] is not False:
        raise ValueError('Scoped theorem status changed')
    masks = [tuple(row['intervened_nodes']) for row in composition['joint_intervention_bounds']]
    if masks != [tuple(j for j in range(3) if i & (1 << j)) for i in range(8)]:
        raise ValueError('A compatible intervention mask is missing')
    program = read(BUNDLE / 'program.json')
    if program['schema'] != 'ncd.frozen-three-node-program.v1' or program['noise_model'] is not None:
        raise ValueError('Changed extracted program or unsupported noise claim')
    child = read(BUNDLE / 'source/child_manifest.json')
    if digest(BUNDLE / 'source/child_manifest.json') != plan['child_manifest_sha256']:
        raise ValueError('Accepted child source mismatch')
    if child['schema'] != 'ncd.piecewise-resume-bundle.v2':
        raise ValueError('Wrong historical child proof')
    suite = ET.fromstring(raw(VALIDATION / 'pytest_frozen_three_node_source_v1_r2.xml'))[0]
    if (suite.attrib['tests'], suite.attrib['failures'], suite.attrib['errors'], suite.attrib['skipped']) != ('3', '0', '0', '0'):
        raise ValueError('Adversarial source tests failed')
    result = {
        'schema': 'ncd.frozen-three-node-acceptance.v1', 'status': 'accepted-scoped',
        'protocol_sha256': digest(plan_path), 'preflight_sha256': digest(preflight_path),
        'stage_record_sha256': stages, 'bundle_manifest_sha256': digest(BUNDLE / 'manifest.json'),
        'source_tests_junit_sha256': digest(VALIDATION / 'pytest_frozen_three_node_source_v1_r2.xml'),
        'installed_wheel_sha256': digest(VALIDATION / 'frozen_three_node_wheels_v1_r2/ncd_frozen_three_node_proof-0.1.0-py3-none-any.whl'),
        'root_count': 2, 'network_nodes': 3, 'compatible_intervention_masks': 8,
        'worst_joint_l1_error_upper': composition['worst_joint_l1_error_upper'],
        'original_claim_counts': plan['original_atomic_claim_counts'],
        'original_objective_achieved': False,
    }
    target = VALIDATION / 'frozen_three_node_acceptance_v1.json'
    if target.exists():
        raise FileExistsError('Accepted proof receipt is immutable')
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'network_nodes': 3,
                      'compatible_intervention_masks': 8, 'original_objective_achieved': False}, sort_keys=True))


if __name__ == '__main__':
    main()