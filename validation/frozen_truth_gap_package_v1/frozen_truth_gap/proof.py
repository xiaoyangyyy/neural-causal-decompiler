"""Truth-only evaluator for a sealed frozen neural-to-program mechanism pair.

Truth is read only after a separate composition acceptance receipt exists.
The result is a fixed-instance counterexample, not a universal impossibility.
"""
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import json
import os

from . import __version__

HISTORICAL_MANIFEST_SHA = '9254d42ed7a352871e6c3d9cf4ffe952556362649e79f1b3eb15112fbae436ce'
WORLD_SHA = '0466d361c03164efd0f2e011d3aaf9aceb74dfc13b0b676e2349dc167e8aae80'
WORLD_RELATIVE = 'worlds/n3_test_id_0/world.json'
ROOT_WEIGHT_SHA = {
    '1': '2a88efbbad9a0963d9dc1c55df4909d540b3565adb8d5c7e03571988916e9886',
    '2': '77bff2f77190843f2b627231d1b33e65fe176036bedbb0d3d422106d4dd74620',
}
PACKAGE_FILES = ('__init__.py', 'proof.py', '__main__.py')


def digest(data):
    return sha256(data).hexdigest()


def parse(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def loaded_sha():
    base = Path(__file__).resolve().parent
    return {name: digest((base / name).read_bytes()) for name in PACKAGE_FILES}


def read(path):
    return parse(Path(path).read_bytes())


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing truth diagnostic artifact differs')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def validate_plan(plan):
    if plan.get('schema') != 'ncd.frozen-truth-gap-plan.v1' or plan.get('version') != __version__:
        raise ValueError('Wrong truth-only protocol')
    if plan.get('output') != 'runs/frozen_three_node_truth_gap_v1' or plan.get('original_objective_achieved') is not False:
        raise ValueError('Truth diagnostic scope changed')
    if plan.get('historical_manifest_sha256') != HISTORICAL_MANIFEST_SHA or plan.get('world_sha256') != WORLD_SHA:
        raise ValueError('Historical true-world anchor changed')
    if not isinstance(plan.get('composition_acceptance_sha256'), str) or len(plan['composition_acceptance_sha256']) != 64:
        raise ValueError('Composition acceptance not frozen')
    if not isinstance(plan.get('composition_manifest_sha256'), str) or len(plan['composition_manifest_sha256']) != 64:
        raise ValueError('Composition proof not frozen')
    if plan.get('original_atomic_claim_counts') != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Original requirement claim was improperly closed')
    if plan.get('loaded_package_sha256') != loaded_sha() or 'site-packages' not in Path(__file__).resolve().parts:
        raise ValueError('Truth evaluator is not the frozen installed package')
    return plan


def checked_composition(bundle, plan):
    from ncd import __version__ as ncd_version
    import ncd
    if ncd_version != '0.59.0' or 'site-packages' not in Path(ncd.__file__).resolve().parts:
        raise ValueError('Wrong historical installed neural checker')
    bundle = Path(bundle)
    manifest_bytes = (bundle / 'manifest.json').read_bytes()
    if digest(manifest_bytes) != plan['composition_manifest_sha256']:
        raise ValueError('Sealed composition manifest changed')
    manifest = parse(manifest_bytes)
    if manifest['schema'] != 'ncd.frozen-three-node-bundle.v1' or manifest['original_objective_achieved'] is not False:
        raise ValueError('Composition proof was not scope-limited')
    for relative, expected in manifest['files'].items():
        if digest((bundle / relative).read_bytes()) != expected:
            raise ValueError('Changed composition proof artifact: ' + relative)
    old_plan = read(bundle / 'source/child_protocol.json')
    ncd_dir = Path(ncd.__file__).resolve().parent
    for relative, expected in old_plan['historical_loaded_source_sha256'].items():
        if relative.startswith('ncd/') and digest((ncd_dir / relative.split('/', 1)[1]).read_bytes()) != expected:
            raise ValueError('Historical truth-checker code changed')
    return read(bundle / 'composition_certificate.json'), read(bundle / 'program.json')


def checked_truth(source_dir):
    from ncd.multiverse import GraphWorld
    source_dir = Path(source_dir)
    manifest_bytes = (source_dir / 'historical_manifest.json').read_bytes()
    world_bytes = (source_dir / 'world.json').read_bytes()
    if digest(manifest_bytes) != HISTORICAL_MANIFEST_SHA or digest(world_bytes) != WORLD_SHA:
        raise ValueError('Changed historical true-world evidence')
    manifest = parse(manifest_bytes)
    if manifest['artifacts'].get(WORLD_RELATIVE) != WORLD_SHA:
        raise ValueError('Historical run did not bind the true world')
    world = parse(world_bytes)
    if GraphWorld.from_dict(world).identity != world['world_id']:
        raise ValueError('True-world identity does not match its SCM parameters')
    if world['graph'] != [[0, 0, 0], [1, 0, 0], [1, 0, 0]]:
        raise ValueError('Learned and true parent graphs do not agree in this fixed instance')
    if any(world['equations'][j] for j in (1, 2)) or world['scales'] != [1.0, 1.0, 1.0]:
        raise ValueError('Declared root truth is not zero in observed coordinates')
    return world


def strict_gap(enclosure, scale, candidate_value):
    lo, hi = map(Q, enclosure)
    normalized_threshold = Q(1, 100) * Q(scale)
    true_distance_lower = lo if lo > 0 else -hi if hi < 0 else Q(0)
    margin = true_distance_lower - 2 * normalized_threshold
    if margin <= 0:
        raise ValueError('No strict incompatibility with both fidelity targets')
    candidate = Q(candidate_value)
    candidate_distance = abs(candidate)
    neural_program_error_upper = max(abs(lo - candidate), abs(hi - candidate))
    if neural_program_error_upper > normalized_threshold:
        raise ValueError('Extracted root program lacks neural fidelity')
    if candidate_distance <= normalized_threshold:
        raise ValueError('Extracted root program has no strict true-mechanism gap')
    return {'neural_value_enclosure': [str(lo), str(hi)],
            'true_structural_root_value': '0',
            'neural_truth_absolute_distance_lower': str(true_distance_lower),
            'one_percent_training_scale': str(normalized_threshold),
            'two_target_incompatibility_margin_lower': str(margin),
            'neural_program_absolute_error_upper': str(neural_program_error_upper),
            'extracted_program_value': str(Q(candidate_value)),
            'extracted_program_truth_absolute_error': str(candidate_distance)}


def derive(bundle, source_dir, plan):
    from ncd.frozen_mechanism_proof import export_mechanism, neural_interval
    from ncd.proof_intervals import Interval
    checked_truth(source_dir)
    certificate, program = checked_composition(bundle, plan)
    if certificate['status'] != 'proved-scoped' or certificate['original_objective_achieved'] is not False or certificate['graph'] != [[0, 0, 0], [1, 0, 0], [1, 0, 0]]:
        raise ValueError('Candidate composition theorem was not sealed')
    if program['schema'] != 'ncd.frozen-three-node-program.v1' or program['noise_model'] is not None:
        raise ValueError('Candidate program or noise scope changed')
    domain = certificate['domain']
    if len(domain) != 3 or any([Q(a), Q(b)] != [Q(-1), Q(1)] for a, b in domain):
        raise ValueError('Unknown fixed-instance proof domain')
    box = [Interval.from_dict(pair) for pair in domain]
    roots = {}
    for j in (1, 2):
        path = Path(bundle) / 'source' / ('mechanism_%d.pt' % j)
        network = export_mechanism(path)
        if network['parents'] or network['checkpoint_sha256'] != ROOT_WEIGHT_SHA[str(j)] or network['checkpoint_sha256'] != certificate['checkpoint_sha256'][j]:
            raise ValueError('Wrong root neural target')
        root_program = program['root_expressions'][str(j)]
        if set(root_program) != {'op', 'value'} or root_program['op'] != 'constant':
            raise ValueError('Root program was not the sealed constant candidate')
        enclosure = neural_interval(network, box).to_dict()
        roots[str(j)] = {'checkpoint_sha256': network['checkpoint_sha256'],
                         'training_output_scale': network['output_scale'],
                         **strict_gap(enclosure, network['output_scale'], root_program['value'])}
    return {
        'schema': 'ncd.frozen-three-node-truth-gap.v1', 'status': 'refuted-scoped',
        'claim_refuted': 'For this fixed true world and frozen neural mechanism set, one scalar program can simultaneously achieve one-percent-of-training-scale uniform fidelity to the neural root and to the true root structural function on [-1,1]^3',
        'world_sha256': WORLD_SHA, 'composition_manifest_sha256': plan['composition_manifest_sha256'],
        'composition_acceptance_sha256': plan['composition_acceptance_sha256'],
        'world_id': checked_truth(source_dir)['world_id'],
        'learned_graph_equals_true_graph_for_this_instance': True,
        'roots': roots,
        'proof': 'Each true root structural function is zero. Its frozen neural output lies entirely in a strict-sign rational interval more than twice the one-percent training-scale radius from zero. If a scalar program were within one radius of both functions, the triangle inequality would put the functions within two radii, contradicting the checked strict margin. This holds at every point because the roots have no parents.',
        'truth_access': 'only this post-acceptance independent evaluator; the candidate proof bundle contains no true-world file',
        'not_proved': ['all neural networks or all program families are impossible to decompile',
                       'the learned graph is generally correct', 'the child truth mechanism is wrong',
                       'intervention distribution failure in Wasserstein distance',
                       'the original compound causal-decompiler objective is refuted'],
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def verify_bundle(directory):
    directory = Path(directory).resolve()
    plan = validate_plan(read(directory / 'protocol.json'))
    manifest = read(directory / 'manifest.json')
    expected = {'protocol.json', 'certificate.json', 'source/historical_manifest.json',
                'source/world.json', 'source/composition_acceptance.json'}
    if manifest.get('schema') != 'ncd.frozen-truth-gap-bundle.v1' or set(manifest.get('files', {})) != expected:
        raise ValueError('Incomplete independent truth diagnostic')
    if manifest.get('original_objective_achieved') is not False:
        raise ValueError('False original closure')
    for relative, expected_sha in manifest['files'].items():
        if digest((directory / relative).read_bytes()) != expected_sha:
            raise ValueError('Truth diagnostic artifact changed: ' + relative)
    accepted_bytes = (directory / 'source/composition_acceptance.json').read_bytes()
    if digest(accepted_bytes) != plan['composition_acceptance_sha256']:
        raise ValueError('Candidate acceptance changed')
    accepted = parse(accepted_bytes)
    if accepted['status'] != 'accepted-scoped' or accepted['original_objective_achieved'] is not False:
        raise ValueError('Candidate not accepted before truth access')
    if accepted['bundle_manifest_sha256'] != plan['composition_manifest_sha256']:
        raise ValueError('Accepted candidate bundle differs')
    bundle = directory.parent / 'frozen_three_node_composition_v1'
    certificate = derive(bundle, directory / 'source', plan)
    if read(directory / 'certificate.json') != certificate:
        raise ValueError('Independent truth gap recomputation differs')
    return {'status': 'verified-scoped', 'conclusion': 'refuted-scoped',
            'strict_root_gaps': 2, 'original_objective_achieved': False}


def generate(config):
    config = Path(config).resolve()
    plan = validate_plan(read(config))
    root = Path(plan['project_root']).resolve()
    out = (root / plan['output']).resolve()
    if not out.is_relative_to(root) or out == root or (out / 'manifest.json').exists():
        raise ValueError('Unsafe or completed truth diagnostic bundle')
    accepted = root / 'validation/frozen_three_node_acceptance_v1.json'
    comp_bundle = root / 'runs/frozen_three_node_composition_v1'
    original = root / 'runs/active_end_to_end_seed4993'
    acceptance_bytes = accepted.read_bytes()
    if digest(acceptance_bytes) != plan['composition_acceptance_sha256']:
        raise ValueError('Candidate acceptance hash changed before truth access')
    acceptance = parse(acceptance_bytes)
    if acceptance.get('status') != 'accepted-scoped' or acceptance.get('original_objective_achieved') is not False:
        raise ValueError('Candidate not accepted before truth access')
    if acceptance.get('bundle_manifest_sha256') != plan['composition_manifest_sha256'] or digest((comp_bundle / 'manifest.json').read_bytes()) != plan['composition_manifest_sha256']:
        raise ValueError('Candidate bundle not sealed before truth access')
    source = {
        'historical_manifest.json': (original / 'manifest.json').read_bytes(),
        'world.json': (original / WORLD_RELATIVE).read_bytes(),
        'composition_acceptance.json': acceptance_bytes,
    }
    write_once(out / 'protocol.json', config.read_bytes())
    for name, data in source.items():
        write_once(out / 'source' / name, data)
    certificate = derive(comp_bundle, out / 'source', plan)
    write_once(out / 'certificate.json', encoded(certificate))
    names = ['protocol.json', 'certificate.json'] + ['source/' + name for name in source]
    write_once(out / 'manifest.json', encoded({'schema': 'ncd.frozen-truth-gap-bundle.v1',
        'files': {name: digest((out / name).read_bytes()) for name in names},
        'original_objective_achieved': False}))
    return verify_bundle(out)