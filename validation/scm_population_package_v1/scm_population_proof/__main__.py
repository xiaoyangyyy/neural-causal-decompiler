"""Portable population-SCM evidence generation and independent replay."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

from . import __version__
from .compact import derive as support_derive, fingerprint
from .counterexamples import derive as counter_derive, verify as counter_verify
from .distribution import derive as tv_derive
from .moments import derive as moment_derive

REQUIREMENTS_SHA = '04d608c94e83c8b77adaca81fcf4942826d3aa7d5d61152d6e895b010fcc724c'
GENERATOR_SHA = '9746dd2c23015e5138c65398065dfc20d1bb34cc8aaca2f5109e326eb1f19883'
CONFIRMATION_SHA = 'ac8e3bd7d7e29d4b6e818165994c3a096e3b3a0063de280163493de5d8dd1c44'
ACCEPTANCE_SHA = '63f9bd9ba187aedcdff58f99932011a9b92ae81587dc5dacf47acfd160c0f271'
ANCHORS = {
    'requirements.md': ('docs/FULL_REQUIREMENTS.md', REQUIREMENTS_SHA),
    'generator.py': ('ncd/multiverse.py', GENERATOR_SHA),
    'confirmation_protocol.json': ('validation/original_confirmation_protocol.json', CONFIRMATION_SHA),
    'confirmation_acceptance.json': ('validation/original_confirmation_acceptance_v1.json', ACCEPTANCE_SHA),
}
MODES = ('active_graph', 'observational_graph', 'oracle_graph_diagnostic')
METHODS = ('baseline', 'structured')
ENVIRONMENTS = ('test_id', 'test_function', 'test_noise', 'test_scale', 'test_intervention')
EXPECTED = tuple('seed_%d_n%d_%s_%d' % (seed, n, env, i)
                 for seed in (8101, 8102) for n in (3, 5, 8)
                 for env in ENVIRONMENTS for i in range(10))
EXPECTED_SET = set(EXPECTED)
UNIT_RE = re.compile(r'^seed_(8101|8102)_n(3|5|8)_(test_id|test_function|test_noise|test_scale|test_intervention)_[0-9]$')
PACKAGE_FILES = ('__init__.py', 'compact.py', 'moments.py', 'distribution.py',
                 'counterexamples.py', '__main__.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(data):
    def object_without_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=object_without_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON constant')))


def read(path):
    return parse(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('An existing proof file differs: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def loaded_sha():
    base = Path(__file__).resolve().parent
    return {name: sha((base / name).read_bytes()) for name in PACKAGE_FILES}


def validate_plan(plan):
    if plan.get('schema') != 'ncd.scm-population-plan.v1' or plan.get('version') != __version__:
        raise ValueError('Unsupported population proof protocol')
    if plan.get('expected_units') != list(EXPECTED) or plan.get('original_objective_achieved') is not False:
        raise ValueError('Cases omitted or original objective promoted')
    if plan.get('output') != 'runs/scm_population_proof_v1':
        raise ValueError('Unknown v1 proof output path')
    if plan.get('source_sha256') != {key: digest for key, (_, digest) in ANCHORS.items()}:
        raise ValueError('Historical source bindings changed')
    if plan.get('loaded_package_sha256') != loaded_sha():
        raise ValueError('Loaded checker source differs from frozen protocol')
    if plan.get('stage_seconds') not in range(1, 43201) or plan.get('artifact_budget_bytes') != 8589934592:
        raise ValueError('Invalid stage resources')
    return plan


def historical_roots(files):
    confirmation = parse(files['confirmation_protocol.json'])
    accepted = parse(files['confirmation_acceptance.json'])
    if confirmation.get('schema') != 'ncd.original-confirmation-protocol.v1':
        raise ValueError('Different confirmation protocol')
    if confirmation.get('seeds') != [8101, 8102] or confirmation.get('nodes') != [3, 5, 8]:
        raise ValueError('Confirmation seeds or graph sizes changed')
    if confirmation.get('environments') != list(ENVIRONMENTS) or confirmation.get('worlds_per_cell') != 10:
        raise ValueError('Confirmation cells changed')
    if confirmation['scientific_source_sha256'].get('ncd/multiverse.py') != GENERATOR_SHA:
        raise ValueError('Generator source is not historical confirmation source')
    if accepted.get('declared_worlds') != 300 or accepted.get('replayed_worlds') != 300:
        raise ValueError('Confirmation world coverage changed')
    if accepted.get('original_claim_counts') != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Original claim ledger changed')
    if accepted['evidence_sha256'].get('validation/original_confirmation_protocol.json') != CONFIRMATION_SHA:
        raise ValueError('Accepted confirmation protocol hash changed')
    return accepted


def verified_raw_unit(unit, contents, receipt):
    if unit not in EXPECTED_SET or not UNIT_RE.fullmatch(unit):
        raise ValueError('Undeclared confirmation world')
    if set(contents) != {'complete.json', 'manifest.json', 'record.json', 'world.json'} | {
            'models/' + mode + '/' + method + '.json' for mode in MODES for method in METHODS}:
        raise ValueError('Unit bundle is missing a declared model or contains extras')
    complete = parse(contents['complete.json'])
    attempt = complete.get('attempt')
    if not isinstance(attempt, str) or not re.fullmatch(r'attempt_[0-9]{4}', attempt):
        raise ValueError('Invalid immutable attempt name')
    base = 'runs/original_confirmation_v1/units/' + unit + '/'
    evidence = receipt['evidence_sha256']
    for key, original in [('complete.json', base + 'complete.json'),
                          ('manifest.json', base + attempt + '/manifest.json'),
                          ('record.json', base + attempt + '/record.json')]:
        if sha(contents[key]) != evidence.get(original):
            raise ValueError('Unit provenance does not match accepted receipt: ' + key)
    manifest = parse(contents['manifest.json'])['files']
    record = parse(contents['record.json'])
    if record.get('protocol_sha256') != CONFIRMATION_SHA:
        raise ValueError('Unit belongs to another confirmation protocol')
    world_paths = [key for key in manifest if key.endswith('/world.json')]
    if len(world_paths) != 1 or not world_paths[0].startswith('worlds/'):
        raise ValueError('Missing or ambiguous historical world')
    world_path = world_paths[0]
    world_dir = world_path[:-len('/world.json')]
    if sha(contents['world.json']) != manifest[world_path]:
        raise ValueError('World data differs from frozen unit manifest')
    world = parse(contents['world.json'])
    if record['world_id'] != world.get('world_id') or record['world_metadata'] != world:
        raise ValueError('Frozen record/world metadata mismatch')
    models = {}
    for mode in MODES:
        for method in METHODS:
            key = 'models/' + mode + '/' + method + '.json'
            historical = world_dir + '/' + mode + '/' + method + '/explicit_scm.json'
            if sha(contents[key]) != manifest.get(historical):
                raise ValueError('SCM differs from frozen unit manifest: ' + key)
            models[mode + '/' + method] = parse(contents[key])
    return world, models, record, world_path


def source_unit(root, unit, receipt):
    location = root / 'runs' / 'original_confirmation_v1' / 'units' / unit
    complete = (location / 'complete.json').read_bytes()
    attempt = parse(complete)['attempt']
    if not isinstance(attempt, str) or not re.fullmatch(r'attempt_[0-9]{4}', attempt):
        raise ValueError('Unsafe or malformed attempt path')
    attempt_root = location / attempt
    manifest = (attempt_root / 'manifest.json').read_bytes()
    record = (attempt_root / 'record.json').read_bytes()
    listed = parse(manifest)['files']
    world_paths = [key for key in listed if key.endswith('/world.json')]
    if len(world_paths) != 1:
        raise ValueError('Ambiguous world source')
    world_path = world_paths[0]
    if not re.fullmatch(r'worlds/n[358]_test_(?:id|function|noise|scale|intervention)_[0-9]/world.json', world_path):
        raise ValueError('Unexpected world source path')
    base = world_path[:-len('/world.json')]
    contents = {'complete.json': complete, 'manifest.json': manifest, 'record.json': record,
                'world.json': (attempt_root / world_path).read_bytes()}
    for mode in MODES:
        for method in METHODS:
            contents['models/' + mode + '/' + method + '.json'] = (
                attempt_root / base / mode / method / 'explicit_scm.json').read_bytes()
    verified_raw_unit(unit, contents, receipt)
    return contents


def unit_zip(contents):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as bundle:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, contents[name], compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=6)
    return stream.getvalue()


def unzip_unit(data):
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        names = bundle.namelist()
        if len(names) != 10 or len(set(names)) != 10 or any(name.startswith('/') or '..' in name.split('/') for name in names):
            raise ValueError('Missing or duplicate unit archive entries')
        if sum(x.file_size for x in bundle.infolist()) > 2000000:
            raise ValueError('Unit source archive exceeds resource bound')
        return {name: bundle.read(name) for name in names}


def unit_certificate(unit, contents, receipt):
    world, models, record, world_path = verified_raw_unit(unit, contents, receipt)
    first = moment_derive(world, '1')
    second = moment_derive(world, '2')
    if first['status'] != 'proved-scoped':
        raise ValueError('True first-moment condition unresolved')
    support = {}
    tv = {}
    for key, model in models.items():
        support[key] = support_derive(model)
        tv[key] = tv_derive(world, model, support[key])
    if set(support) != {mode + '/' + method for mode in MODES for method in METHODS}:
        raise ValueError('A declared estimated SCM was skipped')
    return {
        'schema': 'ncd.scm-population-unit-certificate.v1', 'unit': unit,
        'world_path': world_path, 'world_id': record['world_id'],
        'world_sha256': sha(contents['world.json']),
        'source_model_sha256': {key: sha(contents['models/' + key + '.json']) for key in sorted(models)},
        'true_first_moment': first, 'true_second_moment': second,
        'estimated_support': support, 'population_tv_boundaries': tv,
        'graph_oracle_is_diagnostic_only': True,
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def verify_unit(directory, unit, receipt):
    archive = directory / 'units' / (unit + '.zip')
    certificate = directory / 'certificates' / (unit + '.json')
    contents = unzip_unit(archive.read_bytes())
    expected = unit_certificate(unit, contents, receipt)
    if read(certificate) != expected:
        raise ValueError('Unit proof recomputation mismatch: ' + unit)
    return expected


def summary(cases):
    status = Counter(case['true_second_moment']['status'] for case in cases)
    results = {'worlds': len(cases), 'empirical_scms': sum(len(x['estimated_support']) for x in cases),
               'first_moment_proved': sum(x['true_first_moment']['status'] == 'proved-scoped' for x in cases),
               'second_moment_proved': status['proved-scoped'],
               'second_moment_unresolved': status['unresolved-by-sufficient-bound'],
               'tv_boundaries_refuted_scoped': sum(len(x['population_tv_boundaries']) for x in cases)}
    if results != {'worlds': 300, 'empirical_scms': 1800, 'first_moment_proved': 300,
                   'second_moment_proved': 295, 'second_moment_unresolved': 5,
                   'tv_boundaries_refuted_scoped': 1800}:
        raise ValueError('Declared proof coverage/statuses changed: ' + repr(results))
    return results


def replay(folder):
    folder = Path(folder).resolve()
    manifest = read(folder / 'manifest.json')
    plan = validate_plan(read(folder / 'protocol.json'))
    expected = {'protocol.json', 'scoped_ledger.json', 'counter_n5.json', 'counter_n8.json'} | {
        'source/' + name for name in ANCHORS} | {
        'units/' + unit + '.zip' for unit in EXPECTED} | {
        'certificates/' + unit + '.json' for unit in EXPECTED}
    if manifest.get('schema') != 'ncd.scm-population-bundle.v1' or set(manifest.get('files', {})) != expected:
        raise ValueError('Incomplete proof bundle manifest')
    if manifest.get('original_objective_achieved') is not False:
        raise ValueError('Original claim was improperly closed')
    for relative, digest in manifest['files'].items():
        if sha((folder / relative).read_bytes()) != digest:
            raise ValueError('Proof artifact hash differs: ' + relative)
    source = {name: (folder / 'source' / name).read_bytes() for name in ANCHORS}
    for name, (_, digest) in ANCHORS.items():
        if sha(source[name]) != digest:
            raise ValueError('Historical source root changed: ' + name)
    accepted = historical_roots(source)
    if sha((folder / 'protocol.json').read_bytes()) != manifest['files']['protocol.json']:
        raise ValueError('Proof protocol changed')
    cases = [verify_unit(folder, unit, accepted) for unit in EXPECTED]
    results = summary(cases)
    max_bound = max(Q(bound) for case in cases for item in case['estimated_support'].values()
                    for bound in [item['joint_first_moment_upper']])
    for n in (5, 8):
        certificate = read(folder / ('counter_n%d.json' % n))
        expected_counter = counter_derive(n, str(max_bound), '1')
        if certificate != expected_counter:
            raise ValueError('Family counterexample certificate mismatch')
        counter_verify(certificate)
    ledger = read(folder / 'scoped_ledger.json')
    if ledger != {'schema': 'ncd.scm-population-scoped-ledger.v1',
                  'results': results,
                  'original_claim_counts': {'proved': 0, 'refuted': 2, 'unresolved': 36},
                  'second_moment_unresolved_worlds': sorted(case['unit'] for case in cases
                        if case['true_second_moment']['status'] != 'proved-scoped'),
                  'counterexample_first_moment_upper_all_empirical_scms': str(max_bound),
                  'original_objective_achieved': False}:
        raise ValueError('Scientific ledger differs from independent replay')
    return {'status': 'verified-scoped', 'results': results,
            'original_claim_counts': ledger['original_claim_counts'],
            'original_objective_achieved': False}


def generate(config):
    path = Path(config).resolve()
    plan = validate_plan(read(path))
    root = Path(plan['project_root']).resolve()
    if not root.is_dir():
        raise ValueError('Missing original research workspace')
    out = (root / plan['output']).resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('Unsafe output location')
    if (out / 'manifest.json').exists():
        raise FileExistsError('Completed proof bundle is immutable')
    source = {}
    for name, (relative, digest) in ANCHORS.items():
        data = (root / relative).read_bytes()
        if sha(data) != digest:
            raise ValueError('Historical source changed: ' + name)
        source[name] = data
    accepted = historical_roots(source)
    deadline = time.monotonic() + plan['stage_seconds']
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / 'protocol.json', path.read_bytes())
    for name, data in source.items():
        write_once(out / 'source' / name, data)
    cases = []
    for unit in EXPECTED:
        if time.monotonic() >= deadline:
            return {'status': 'unresolved-budget', 'completed_worlds': len(cases),
                    'original_objective_achieved': False}
        archive = out / 'units' / (unit + '.zip')
        certificate = out / 'certificates' / (unit + '.json')
        if archive.exists() and certificate.exists():
            result = verify_unit(out, unit, accepted)
        else:
            contents = source_unit(root, unit, accepted)
            write_once(archive, unit_zip(contents))
            result = unit_certificate(unit, contents, accepted)
            write_once(certificate, encoded(result))
        cases.append(result)
        if len(cases) % 10 == 0:
            checkpoint = {'completed_worlds': len(cases), 'last_unit': unit,
                          'original_objective_achieved': False}
            (out / 'checkpoint.json').write_bytes(encoded(checkpoint))
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > plan['artifact_budget_bytes']:
            return {'status': 'unresolved-artifact-budget', 'completed_worlds': len(cases),
                    'original_objective_achieved': False}
    results = summary(cases)
    max_bound = max(Q(item['joint_first_moment_upper']) for case in cases
                    for item in case['estimated_support'].values())
    for n in (5, 8):
        write_once(out / ('counter_n%d.json' % n), encoded(counter_derive(n, str(max_bound), '1')))
    ledger = {'schema': 'ncd.scm-population-scoped-ledger.v1', 'results': results,
              'original_claim_counts': {'proved': 0, 'refuted': 2, 'unresolved': 36},
              'second_moment_unresolved_worlds': sorted(case['unit'] for case in cases
                    if case['true_second_moment']['status'] != 'proved-scoped'),
              'counterexample_first_moment_upper_all_empirical_scms': str(max_bound),
              'original_objective_achieved': False}
    write_once(out / 'scoped_ledger.json', encoded(ledger))
    names = ['protocol.json', 'scoped_ledger.json', 'counter_n5.json', 'counter_n8.json']
    names += ['source/' + name for name in ANCHORS]
    names += ['units/' + unit + '.zip' for unit in EXPECTED]
    names += ['certificates/' + unit + '.json' for unit in EXPECTED]
    write_once(out / 'manifest.json', encoded({'schema': 'ncd.scm-population-bundle.v1',
        'files': {name: sha((out / name).read_bytes()) for name in names},
        'original_objective_achieved': False}))
    return replay(out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prove').add_argument('--config', required=True)
    sub.add_parser('verify-proof').add_argument('bundle')
    args = parser.parse_args()
    result = generate(args.config) if args.command == 'prove' else replay(args.bundle)
    print(json.dumps(result, sort_keys=True))
    if result.get('status') != 'verified-scoped':
        sys.exit(2)


if __name__ == '__main__':
    main()
