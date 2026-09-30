"""Strict acceptance gate for the historical Student5 second-moment extension."""
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'validation'
BUNDLE = ROOT / 'runs/scm_second_moment_completion_v1'
PLAN = V / 'scm_second_moment_protocol_v1.json'
UNITS = (
    'seed_8101_n5_test_noise_7',
    'seed_8101_n8_test_noise_9',
    'seed_8102_n3_test_noise_2',
    'seed_8102_n3_test_noise_4',
    'seed_8102_n5_test_noise_3',
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    plan = read(PLAN)
    if plan['schema'] != 'ncd.scm-second-moment-plan.v1' or plan['unresolved_units'] != list(UNITS):
        raise ValueError('Wrong frozen five-world scope')
    if plan['original_atomic_claim_counts'] != {'proved': 0, 'refuted': 2, 'unresolved': 36} or plan['original_objective_achieved'] is not False:
        raise ValueError('Original requirements were improperly closed')
    old_acceptance = read(BUNDLE / 'source/prior_acceptance.json')
    if old_acceptance['status'] != 'accepted-scoped' or old_acceptance['results']['second_moment_proved'] != 295 or old_acceptance['results']['second_moment_unresolved'] != 5:
        raise ValueError('Previous finite-moment classification changed')
    if old_acceptance['second_moment_unresolved_worlds'] != list(UNITS):
        raise ValueError('Wrong five historical unresolved worlds')
    for name, expected in plan['loaded_package_sha256'].items():
        installed = V / 'scm_second_moment_env_v1/Lib/site-packages/scm_second_moment' / name
        if digest(installed) != expected:
            raise ValueError('Independent installed proof code changed')
    stage_hashes = {}
    expected_result = {'status': 'verified-scoped', 'prior_finite_second_moments': 295,
                       'new_infinite_second_moments': 5, 'classified_observational_worlds': 300,
                       'original_objective_achieved': False}
    for mode in ('prove', 'verify'):
        path = V / ('scm_second_moment_stage_v1_' + mode + '_attempt0000.json')
        row = read(path)
        if row['mode'] != mode or row['status'] != 'verified-scoped' or row['exit_code'] != 0 or row['result'] != expected_result:
            raise ValueError('Installed %s stage did not verify' % mode)
        resource = row['resource_guard']
        if resource['timeout'] or not resource['descendants_included'] or resource['active_processes_on_return'] != 0 or resource['peak_job_memory_bytes'] > plan['memory_bytes']:
            raise ValueError('Installed %s resource guard failed' % mode)
        if row['protocol_sha256'] != digest(PLAN) or row['bundle_manifest_sha256'] != digest(BUNDLE / 'manifest.json') or row['artifact_bytes'] > plan['artifact_budget_bytes']:
            raise ValueError('Installed %s proof identity or artifact limit changed' % mode)
        stage_hashes[mode] = digest(path)
    manifest = read(BUNDLE / 'manifest.json')
    expected_files = {'protocol.json', 'certificates.json', 'source/prior_acceptance.json',
                      'source/prior_manifest.json', 'source/prior_ledger.json'} | {
                          'source/units/' + unit + '.zip' for unit in UNITS}
    if manifest['schema'] != 'ncd.scm-second-moment-bundle.v1' or set(manifest['files']) != expected_files or manifest['original_objective_achieved'] is not False:
        raise ValueError('Incomplete portable second-moment proof bundle')
    for relative, expected in manifest['files'].items():
        if digest(BUNDLE / relative) != expected:
            raise ValueError('Changed portable proof artifact: ' + relative)
    if digest(BUNDLE / 'protocol.json') != digest(PLAN):
        raise ValueError('Proof bundle used another protocol')
    for source, key in (('prior_acceptance.json', 'prior_acceptance_sha256'),
                        ('prior_manifest.json', 'prior_manifest_sha256'),
                        ('prior_ledger.json', 'prior_ledger_sha256')):
        if digest(BUNDLE / 'source' / source) != plan[key]:
            raise ValueError('Changed historical acceptance source')
    certificates = read(BUNDLE / 'certificates.json')
    if set(certificates) != set(UNITS):
        raise ValueError('Missing tail witnesses')
    for unit in UNITS:
        item = certificates[unit]
        if item['unit'] != unit or item['degree'] < 3 or Q(item['leading_absolute_lower']) <= 0 or Q(item['truncated_second_moment_coefficient']) <= 0 or item['original_claim_closed'] is not False:
            raise ValueError('Invalid scoped strict tail witness')
    junit = V / 'pytest_scm_second_moment_source_v1.xml'
    suite = ET.parse(junit).getroot()[0]
    if (suite.attrib['tests'], suite.attrib['failures'], suite.attrib['errors'], suite.attrib['skipped']) != ('3', '0', '0', '0'):
        raise ValueError('Source adversarial tests failed')
    wheel = V / 'scm_second_moment_wheels_v1/ncd_scm_second_moment-0.1.0-py3-none-any.whl'
    result = {
        'schema': 'ncd.scm-second-moment-acceptance.v1', 'status': 'accepted-scoped',
        'protocol_sha256': digest(PLAN), 'bundle_manifest_sha256': digest(BUNDLE / 'manifest.json'),
        'installed_wheel_sha256': digest(wheel), 'source_tests_junit_sha256': digest(junit),
        'stage_record_sha256': stage_hashes, 'prior_finite_second_moments': 295,
        'new_infinite_second_moments': 5, 'classified_observational_worlds': 300,
        'worlds_with_any_intervention_claim': 0,
        'original_claim_counts': plan['original_atomic_claim_counts'],
        'original_objective_achieved': False,
    }
    target = V / 'scm_second_moment_acceptance_v1.json'
    if target.exists():
        raise FileExistsError('Accepted scoped proof receipt is immutable')
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'accepted-scoped', 'finite': 295, 'infinite': 5,
                      'original_objective_achieved': False}, sort_keys=True))


if __name__ == '__main__':
    main()