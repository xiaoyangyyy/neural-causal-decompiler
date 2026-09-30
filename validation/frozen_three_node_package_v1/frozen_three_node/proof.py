"""Independent rational composition of three actual frozen neural mechanisms.

The statement concerns the learned deterministic mechanisms and their extracted
programs. It does not identify the true graph, equations, or exogenous laws.
"""
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import json
import os

from . import __version__

CHILD_MANIFEST_SHA = '309d7a7984e294afd6c8af311b831f0bfebf358092cb97112a7755dbad64f4d4'
ROOT_WEIGHT_SHA = {
    'mechanism_1.pt': '2a88efbbad9a0963d9dc1c55df4909d540b3565adb8d5c7e03571988916e9886',
    'mechanism_2.pt': '77bff2f77190843f2b627231d1b33e65fe176036bedbb0d3d422106d4dd74620',
}
CHILD_ORIGINAL = 'runs/historical_mechanism0_piecewise_v8'
WEIGHT_ORIGINAL = 'runs/active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline'
SOURCE_NAMES = ('child_manifest.json', 'child_protocol.json', 'child_certificate.json',
                'child_program.json', 'mechanism_0.pt', 'mechanism_1.pt', 'mechanism_2.pt')
PACKAGE_NAMES = ('__init__.py', 'proof.py', '__main__.py')


def digest(data):
    return sha256(data).hexdigest()


def parse(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def loaded_source_sha():
    base = Path(__file__).resolve().parent
    return {name: digest((base / name).read_bytes()) for name in PACKAGE_NAMES}


def read(path):
    return parse(Path(path).read_bytes())


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Frozen proof file differs: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def validate_plan(plan):
    if plan.get('schema') != 'ncd.frozen-three-node-plan.v1' or plan.get('version') != __version__:
        raise ValueError('Unknown composition proof protocol')
    if plan.get('output') != 'runs/frozen_three_node_composition_v1' or plan.get('original_objective_achieved') is not False:
        raise ValueError('Scope or output path changed')
    if plan.get('child_manifest_sha256') != CHILD_MANIFEST_SHA or plan.get('root_weight_sha256') != ROOT_WEIGHT_SHA:
        raise ValueError('Frozen historical anchors changed')
    if set(plan.get('source_sha256') or {}) != set(SOURCE_NAMES):
        raise ValueError('Historical source set changed')
    if plan.get('threads') != 2 or plan.get('memory_bytes') != 8589934592:
        raise ValueError('Verifier thread or memory budget changed')
    if plan.get('original_atomic_claim_counts') != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Original claim ledger was incorrectly promoted')
    if 'site-packages' not in Path(__file__).resolve().parts:
        raise ValueError('Proof package must be installed for formal replay')
    if plan.get('loaded_package_sha256') != loaded_source_sha():
        raise ValueError('Installed proof code differs from frozen protocol')
    if not 0 < plan.get('stage_seconds', 0) <= 43200 or plan.get('artifact_budget_bytes') != 8589934592:
        raise ValueError('Invalid proof resource limits')
    return plan


def source_from_original(root):
    original = {
        'child_manifest.json': root / CHILD_ORIGINAL / 'manifest.json',
        'child_protocol.json': root / CHILD_ORIGINAL / 'protocol.json',
        'child_certificate.json': root / CHILD_ORIGINAL / 'certificate.json',
        'child_program.json': root / CHILD_ORIGINAL / 'program.json',
    }
    original.update({'mechanism_%d.pt' % i: root / WEIGHT_ORIGINAL / ('mechanism_%d.pt' % i)
                     for i in range(3)})
    return {name: original[name].read_bytes() for name in SOURCE_NAMES}


def historical_bindings(source):
    if set(source) != set(SOURCE_NAMES) or digest(source['child_manifest.json']) != CHILD_MANIFEST_SHA:
        raise ValueError('Incomplete or changed accepted child proof')
    manifest = parse(source['child_manifest.json'])
    if manifest.get('schema') != 'ncd.piecewise-resume-bundle.v2' or manifest.get('original_objective_achieved') is not False:
        raise ValueError('Wrong accepted child proof schema')
    for source_name, historical_name in (
        ('child_protocol.json', 'protocol.json'),
        ('child_certificate.json', 'certificate.json'),
        ('child_program.json', 'program.json'),
    ):
        if digest(source[source_name]) != manifest['files'].get(historical_name):
            raise ValueError('Changed accepted child artifact: ' + source_name)
    old_plan = parse(source['child_protocol.json'])
    if old_plan.get('schema') != 'ncd.piecewise-mechanism-resume-protocol.v8':
        raise ValueError('Wrong historical child protocol')
    if digest(source['mechanism_0.pt']) != old_plan['checkpoint_sha256']:
        raise ValueError('Changed child checkpoint')
    for name, expected in ROOT_WEIGHT_SHA.items():
        if digest(source[name]) != expected:
            raise ValueError('Changed root checkpoint: ' + name)
    if parse(source['child_certificate.json'])['network']['checkpoint_sha256'] != old_plan['checkpoint_sha256']:
        raise ValueError('Child certificate binds another checkpoint')
    return old_plan


def installed_historical_sources(old_plan):
    import ncd
    import proof_workbench
    if ncd.__version__ != '0.59.0':
        raise ValueError('Wrong installed historical NCD release')
    packages = {'ncd': Path(ncd.__file__).resolve().parent,
                'proof_workbench': Path(proof_workbench.__file__).resolve().parent}
    if any('site-packages' not in base.parts for base in packages.values()):
        raise ValueError('Historical verifier must come from an installed package')
    declared = old_plan['historical_loaded_source_sha256']
    if not declared or not any(p.startswith('proof_workbench/') for p in declared):
        raise ValueError('Incomplete historical installed-source map')
    for relative, expected in declared.items():
        package, file = relative.split('/', 1)
        if package not in packages or '/' in file or digest((packages[package] / file).read_bytes()) != expected:
            raise ValueError('Historical installed proof source mismatch: ' + relative)
    return {package: str(base) for package, base in packages.items()}


def exported_networks(source_dir):
    from ncd.frozen_mechanism_proof import export_mechanism
    networks = [export_mechanism(source_dir / ('mechanism_%d.pt' % i)) for i in range(3)]
    if [n['parents'] for n in networks] != [[1, 2], [], []]:
        raise ValueError('This theorem requires the frozen three-node learned parent graph')
    if any(n['checkpoint_sha256'] != digest((source_dir / ('mechanism_%d.pt' % i)).read_bytes())
           for i, n in enumerate(networks)):
        raise ValueError('Export is not bound to frozen checkpoints')
    return networks


def root_result(network, domain):
    from ncd.frozen_mechanism_proof import neural_interval, certify_mechanism, verify_mechanism
    from ncd.proof_intervals import Interval
    if network['parents']:
        raise ValueError('Root theorem cannot absorb learned parents')
    box = [Interval.from_dict(pair) for pair in domain]
    value = neural_interval(network, box)
    program = {'op': 'constant', 'value': float((value.lo + value.hi) / 2)}
    certificate = certify_mechanism(network, program, domain, '1/100', max_boxes=1, seconds=60)
    if certificate['status'] != 'proved' or verify_mechanism(network, program, certificate)['conclusion'] != 'proved':
        raise ValueError('Root mechanism fidelity not proved')
    if value.lo < -1 or value.hi > 1 or Q(program['value']) < -1 or Q(program['value']) > 1:
        raise ValueError('Root output escaped the composition domain')
    return {'network': network, 'program': program, 'certificate': certificate,
            'network_value_enclosure': value.to_dict()}


def child_lipschitz(network):
    if network['parents'] != [1, 2] or len(network['layers']) != 3:
        raise ValueError('Wrong child network structure')
    if [layer['activation'] for layer in network['layers']] != ['tanh', 'tanh', 'identity']:
        raise ValueError('Wrong child activation order')
    sensitivity = [Q(1)]
    for layer in reversed(network['layers']):
        weights = [[abs(Q(v)) for v in row] for row in layer['weights']]
        if len(weights) != len(sensitivity) or not weights or any(len(row) != len(weights[0]) for row in weights):
            raise ValueError('Invalid layer dimensions')
        sensitivity = [sum((sensitivity[j] * weights[j][i] for j in range(len(weights))), Q(0))
                       for i in range(len(weights[0]))]
    if len(sensitivity) != len(network['parents']) or len(network['std']) != len(sensitivity):
        raise ValueError('Wrong child input normalization dimension')
    result = {str(parent): sensitivity[i] * Q(network['output_scale']) / Q(network['std'][i])
              for i, parent in enumerate(network['parents'])}
    if any(value < 0 for value in result.values()):
        raise ValueError('Negative Lipschitz constant')
    return result


def composition(networks, roots, child_certificate, domain):
    if len(domain) != 3 or any([Q(a), Q(b)] != [Q(-1), Q(1)] for a, b in domain):
        raise ValueError('Joint intervention domain differs from accepted child box')
    if child_certificate['status'] != 'proved' or child_certificate['network'] != networks[0]:
        raise ValueError('Child certificate does not prove this network')
    leaves = [row for row in child_certificate['nodes'] if row.get('kind') == 'leaf']
    if not leaves or child_certificate['unresolved_cells']:
        raise ValueError('Incomplete child program')
    child_normalized = max(max(abs(Q(v)) for v in row['normalized_error']) for row in leaves)
    if child_normalized > Q('1/100') or Q(child_certificate['normalization_scale']) != Q(networks[0]['output_scale']):
        raise ValueError('Child local fidelity bound changed')
    local = [child_normalized * Q(networks[0]['output_scale'])]
    for index in (1, 2):
        root = roots[str(index)]
        if root['network'] != networks[index] or root['certificate']['status'] != 'proved':
            raise ValueError('Root fidelity target changed')
        certified_error = root['certificate']['tree'][0]['error']
        local.append(max(abs(Q(v)) for v in certified_error) * Q(root['certificate']['normalizer']))
    lipschitz = child_lipschitz(networks[0])
    graph = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    masks = []
    for mask in range(8):
        intervened = [j for j in range(3) if mask & (1 << j)]
        errors = [Q(0)] * 3
        for j in (1, 2):
            if j not in intervened:
                errors[j] = local[j]
        if 0 not in intervened:
            errors[0] = local[0] + lipschitz['1'] * errors[1] + lipschitz['2'] * errors[2]
        masks.append({'intervened_nodes': intervened,
                      'coordinate_absolute_error_upper': [str(v) for v in errors],
                      'joint_l1_error_upper': str(sum(errors))})
    return {
        'schema': 'ncd.frozen-three-node-composition.v1', 'status': 'proved-scoped',
        'graph': graph, 'checkpoint_sha256': [n['checkpoint_sha256'] for n in networks],
        'domain': domain, 'intervention_scope': 'all 8 compatible node subsets; each observed do value in [-1,1]',
        'root_network_value_enclosures': {key: roots[key]['network_value_enclosure'] for key in ('1', '2')},
        'local_absolute_error_upper': [str(v) for v in local],
        'child_coordinate_lipschitz_upper': {key: str(v) for key, v in lipschitz.items()},
        'joint_intervention_bounds': masks,
        'worst_joint_l1_error_upper': str(max(Q(row['joint_l1_error_upper']) for row in masks)),
        'proof': 'Induct in learned-DAG order 1,2,0. Fixed do nodes have zero paired error. Root certificates bound free roots and keep both executions inside the child proof box. The child network is coordinatewise Lipschitz by absolute weight products, positive input standard deviations and tanh derivative at most one. Triangle inequality combines the child local program certificate at the program parents with the neural change between paired parents.',
        'semantic_scope': 'exact stored coefficient ideal real evaluation and mathematical Tanh; not floating hardware execution',
        'not_proved': ['learned parents equal true graph', 'frozen neural functions equal true mechanisms',
                       'exogenous noise law or independence', 'distributional intervention fidelity',
                       'other worlds, nodes, or models', 'global MDL minimality'],
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def derive(source_dir):
    from proof_workbench.piecewise_mechanism import program
    from ncd.frozen_mechanism_proof import verify_mechanism
    source_dir = Path(source_dir)
    source = {name: (source_dir / name).read_bytes() for name in SOURCE_NAMES}
    old_plan = historical_bindings(source)
    installed_historical_sources(old_plan)
    networks = exported_networks(source_dir)
    child = parse(source['child_certificate.json'])
    if child['network'] != networks[0] or child['domain'] != old_plan['domain'] or child['epsilon'] != '1/100':
        raise ValueError('Child certificate target/domain changed')
    child_program = parse(source['child_program.json'])
    if child_program != program(child):
        raise ValueError('Child program differs from checked certificate')
    roots = {str(j): root_result(networks[j], child['domain']) for j in (1, 2)}
    for j in (1, 2):
        if verify_mechanism(networks[j], roots[str(j)]['program'], roots[str(j)]['certificate'])['conclusion'] != 'proved':
            raise ValueError('Root checker disagrees')
    composed = composition(networks, roots, child, child['domain'])
    explicit_program = {
        'schema': 'ncd.frozen-three-node-program.v1', 'graph': composed['graph'],
        'domain': child['domain'], 'root_expressions': {key: roots[key]['program'] for key in ('1', '2')},
        'child_piecewise_program': child_program,
        'noise_model': None, 'true_graph_claimed': False, 'original_objective_achieved': False,
    }
    return roots, composed, explicit_program


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / 'manifest.json')
    plan = validate_plan(read(bundle / 'protocol.json'))
    expected = {'protocol.json', 'root_certificates.json', 'composition_certificate.json', 'program.json'} | {
        'source/' + name for name in SOURCE_NAMES}
    if manifest.get('schema') != 'ncd.frozen-three-node-bundle.v1' or set(manifest.get('files', {})) != expected:
        raise ValueError('Incomplete three-node proof bundle')
    if manifest.get('original_objective_achieved') is not False:
        raise ValueError('Original objective improperly promoted')
    for relative, expected_sha in manifest['files'].items():
        if digest((bundle / relative).read_bytes()) != expected_sha:
            raise ValueError('Changed proof artifact: ' + relative)
    source = {name: (bundle / 'source' / name).read_bytes() for name in SOURCE_NAMES}
    historical_bindings(source)
    roots, composed, explicit_program = derive(bundle / 'source')
    for name, expected_value in (('root_certificates.json', roots),
                                 ('composition_certificate.json', composed),
                                 ('program.json', explicit_program)):
        if read(bundle / name) != expected_value:
            raise ValueError('Independent theorem replay disagrees: ' + name)
    if plan['source_sha256'] != {name: digest(source[name]) for name in SOURCE_NAMES}:
        raise ValueError('Proof bundle was created from different source files')
    return {'status': 'verified-scoped', 'conclusion': 'proved-scoped',
            'network_nodes': 3, 'compatible_intervention_masks': 8,
            'worst_joint_l1_error_upper': composed['worst_joint_l1_error_upper'],
            'original_objective_achieved': False}


def generate(config):
    config = Path(config).resolve()
    plan = validate_plan(read(config))
    root = Path(plan['project_root']).resolve()
    if not root.is_dir():
        raise ValueError('Original source root missing')
    out = (root / plan['output']).resolve()
    if not out.is_relative_to(root) or out == root or (out / 'manifest.json').exists():
        raise ValueError('Unsafe or completed output bundle')
    original = source_from_original(root)
    historical_bindings(original)
    if plan['source_sha256'] != {name: digest(original[name]) for name in SOURCE_NAMES}:
        raise ValueError('Frozen source file changed')
    write_once(out / 'protocol.json', config.read_bytes())
    for name, data in original.items():
        write_once(out / 'source' / name, data)
    roots, composed, explicit_program = derive(out / 'source')
    write_once(out / 'root_certificates.json', encoded(roots))
    write_once(out / 'composition_certificate.json', encoded(composed))
    write_once(out / 'program.json', encoded(explicit_program))
    names = ['protocol.json', 'root_certificates.json', 'composition_certificate.json', 'program.json']
    names += ['source/' + name for name in SOURCE_NAMES]
    if sum((out / name).stat().st_size for name in names) > plan['artifact_budget_bytes']:
        raise ValueError('Proof artifact budget exceeded')
    write_once(out / 'manifest.json', encoded({'schema': 'ncd.frozen-three-node-bundle.v1',
        'files': {name: digest((out / name).read_bytes()) for name in names},
        'original_objective_achieved': False}))
    return verify_bundle(out)