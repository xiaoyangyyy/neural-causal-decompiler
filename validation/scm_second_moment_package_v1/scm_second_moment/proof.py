"""Exact Student-5 tail witnesses for five accepted historical SCM worlds.

All coefficient arithmetic uses the exact binary values decoded from JSON.
The theorem is about ideal continuous exogenous laws, not finite device PRNGs.
"""
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import math
import os

from . import __version__

PRIOR_ACCEPTANCE_SHA = '78494f2852e4f036556ae45ab6645531902d2693038aaae725fd853d8d77c9b6'
PRIOR_MANIFEST_SHA = '49ab0a5307f7336974c32bdbad5614fd7a533479af4d0554b52f13ac1b9a1942'
PRIOR_LEDGER_SHA = '2eacf7348e5d8e4ca52295ef20a49e821baa37b8a888ea1787c9d5339cee20b9'
UNITS = (
    'seed_8101_n5_test_noise_7',
    'seed_8101_n8_test_noise_9',
    'seed_8102_n3_test_noise_2',
    'seed_8102_n3_test_noise_4',
    'seed_8102_n5_test_noise_3',
)
PACKAGE_FILES = ('__init__.py', 'proof.py', '__main__.py')
ALPHA = Q(4, 9) * Q(5, 69) ** 3
ZERO = (Q(0), Q(0))


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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def read(path):
    return parse(Path(path).read_bytes())


def installed_sha():
    base = Path(__file__).resolve().parent
    return {name: digest((base / name).read_bytes()) for name in PACKAGE_FILES}


def validate_plan(plan):
    if plan.get('schema') != 'ncd.scm-second-moment-plan.v1' or plan.get('version') != __version__:
        raise ValueError('Unknown Student5 proof protocol')
    if plan.get('output') != 'runs/scm_second_moment_completion_v1' or plan.get('original_objective_achieved') is not False:
        raise ValueError('Proof output or original scope changed')
    if plan.get('prior_acceptance_sha256') != PRIOR_ACCEPTANCE_SHA or plan.get('prior_manifest_sha256') != PRIOR_MANIFEST_SHA or plan.get('prior_ledger_sha256') != PRIOR_LEDGER_SHA:
        raise ValueError('Accepted historical evidence anchor changed')
    if plan.get('unresolved_units') != list(UNITS) or plan.get('original_atomic_claim_counts') != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Historical second-moment or original claim scope changed')
    if plan.get('loaded_package_sha256') != installed_sha() or 'site-packages' not in Path(__file__).resolve().parts:
        raise ValueError('Proof is not running from the frozen installed package')
    if plan.get('memory_bytes') != 8589934592 or plan.get('artifact_budget_bytes') != 8589934592 or plan.get('threads') != 2 or not 0 < plan.get('stage_seconds', 0) <= 43200:
        raise ValueError('Invalid resource budget')
    return plan


def rational(value):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError('Coefficient is not a finite stored real')
    return Q(value)


def add_interval(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul_interval(a, b):
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def trim(poly):
    while len(poly) > 1 and poly[-1] == ZERO:
        poly.pop()
    return poly


def add_poly(a, b):
    result = []
    for index in range(max(len(a), len(b))):
        result.append(add_interval(a[index] if index < len(a) else ZERO,
                                   b[index] if index < len(b) else ZERO))
    return trim(result)


def mul_poly(a, b):
    result = [ZERO for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] = add_interval(result[i + j], mul_interval(x, y))
    return trim(result)


def scale_poly(poly, scalar):
    return [mul_interval(item, (scalar, scalar)) for item in poly]


def order(graph):
    n = len(graph)
    if not n or any(type(row) is not list or len(row) != n or any(type(x) is not int or x not in (0, 1) for x in row) for row in graph):
        raise ValueError('Invalid graph')
    if any(graph[j][j] for j in range(n)):
        raise ValueError('Self-parent')
    remain = set(range(n))
    result = []
    while remain:
        ready = sorted(j for j in remain if not any(graph[i][j] for i in remain))
        if not ready:
            raise ValueError('Cyclic graph')
        j = ready[0]
        remain.remove(j)
        result.append(j)
    return result


def polynomials(world, pivot):
    graph = world['graph']
    topo = order(graph)
    n = len(graph)
    if type(pivot) is not int or not 0 <= pivot < n or any(graph[i][pivot] for i in range(n)):
        raise ValueError('Pivot must be an exogenous root')
    if world['noise_family'] != 'student' or world.get('root_shift') is not False:
        raise ValueError('This theorem needs independent unshifted Student5 noises')
    sigma = rational(world['noise_scale'])
    if not Q(1, 4) <= sigma <= Q(1, 2):
        raise ValueError('Student5 scale is outside the density minorant range')
    if len(world['equations']) != n or len(world['scales']) != n or any(rational(s) != 1 for s in world['scales']):
        raise ValueError('This theorem needs observed scales equal to one')
    result = [None] * n
    for node in topo:
        poly = [ZERO, (Q(1), Q(1))] if node == pivot else [(Q(-1), Q(1))]
        used = set()
        for term in world['equations'][node]:
            op, parents = term['operator'], term['parents']
            arity = {'linear': 1, 'square': 1, 'sin': 1, 'cos': 1,
                     'tanh': 1, 'interaction': 2}
            if op not in arity or type(parents) is not list or len(parents) != arity[op]:
                raise ValueError('Unsupported or malformed structural term')
            if any(type(p) is not int or not 0 <= p < n or not graph[p][node] for p in parents):
                raise ValueError('Equation parent is outside the declared graph')
            used.update(parents)
            scalar = rational(term['coefficient'])
            if not scalar:
                raise ValueError('Zero structural coefficient')
            if op == 'linear':
                piece = result[parents[0]]
            elif op == 'square':
                piece = mul_poly(result[parents[0]], result[parents[0]])
            elif op == 'interaction':
                piece = mul_poly(result[parents[0]], result[parents[1]])
            else:
                piece = [(Q(-1), Q(1))]
            poly = add_poly(poly, scale_poly(piece, scalar))
        if used != {i for i in range(n) if graph[i][node]}:
            raise ValueError('Graph has a parent absent from the equation')
        result[node] = poly
    return result


def witness(world, unit, pivot, target):
    n = len(world['graph'])
    if type(target) is not int or not 0 <= target < n:
        raise ValueError('Wrong target')
    poly = polynomials(world, pivot)[target]
    degree = len(poly) - 1
    lo, hi = poly[-1]
    if degree < 3 or not (lo > 0 or hi < 0):
        raise ValueError('No separated Student5 high-degree coefficient')
    leading = min(abs(lo), abs(hi))
    lower = sum(max(abs(a), abs(b)) for a, b in poly[:-1])
    ratio = 2 * lower / leading
    cutoff = max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    other_event = (2 * ALPHA) ** (n - 1)
    tail_coefficient = other_event * ALPHA * (leading / 2) ** 2
    exponent = 2 * degree - 5
    if exponent < 1 or tail_coefficient <= 0 or cutoff < 1:
        raise ArithmeticError('Invalid divergent Student5 tail witness')
    return {
        'schema': 'ncd.historical-student5-second-moment-witness.v1',
        'unit': unit, 'world_id': world['world_id'], 'pivot_noise_node': pivot,
        'divergent_output_node': target, 'degree': degree,
        'coefficient_intervals': [[str(a), str(b)] for a, b in poly],
        'leading_absolute_lower': str(leading),
        'lower_coefficient_absolute_sum': str(lower),
        'tail_start_integer': cutoff, 'student5_density_minorant': str(ALPHA),
        'other_noise_event_probability_lower': str(other_event),
        'truncated_second_moment_coefficient': str(tail_coefficient),
        'truncated_growth_exponent': exponent,
        'claim': 'The ideal observational population second moment of this output is infinite',
        'bound': 'For M>=T, E[X_target^2] >= C*(M^q-T^q)/q with q=2*degree-5>=1, C>0',
        'semantic_scope': 'Ideal independent continuous Student5 noises; stored binary coefficients interpreted exactly; not finite device PRNG or hardware rounding',
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def choose_witness(world, unit):
    graph = world['graph']
    for pivot in range(len(graph)):
        if any(graph[i][pivot] for i in range(len(graph))):
            continue
        for target in range(len(graph)):
            try:
                return witness(world, unit, pivot, target)
            except ValueError as exc:
                if str(exc) != 'No separated Student5 high-degree coefficient':
                    raise
    raise ValueError('No strict high-degree witness for historical world')


def verify_witness(world, unit, certificate):
    if type(certificate) is not dict:
        raise ValueError('Missing tail certificate')
    expected = witness(world, unit, certificate.get('pivot_noise_node'),
                       certificate.get('divergent_output_node'))
    if certificate != expected:
        raise ValueError('Tail certificate differs from independent exact replay')
    return True


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing immutable proof artifact changed')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def checked_sources(source):
    acceptance_bytes = (source / 'prior_acceptance.json').read_bytes()
    manifest_bytes = (source / 'prior_manifest.json').read_bytes()
    ledger_bytes = (source / 'prior_ledger.json').read_bytes()
    if (digest(acceptance_bytes), digest(manifest_bytes), digest(ledger_bytes)) != (
            PRIOR_ACCEPTANCE_SHA, PRIOR_MANIFEST_SHA, PRIOR_LEDGER_SHA):
        raise ValueError('Accepted previous population evidence changed')
    accepted, manifest, ledger = map(parse, (acceptance_bytes, manifest_bytes, ledger_bytes))
    expected_old = {'worlds': 300, 'empirical_scms': 1800,
                    'first_moment_proved': 300, 'second_moment_proved': 295,
                    'second_moment_unresolved': 5, 'tv_boundaries_refuted_scoped': 1800}
    if accepted['status'] != 'accepted-scoped' or accepted['proof_manifest_sha256'] != PRIOR_MANIFEST_SHA or accepted['results'] != expected_old:
        raise ValueError('Previous accepted population conclusion differs')
    if accepted['second_moment_unresolved_worlds'] != list(UNITS) or accepted['original_objective_achieved'] is not False:
        raise ValueError('Wrong five historical unresolved cases')
    if ledger['second_moment_unresolved_worlds'] != list(UNITS) or ledger['results'] != expected_old:
        raise ValueError('Previous scoped ledger changed')
    if digest(ledger_bytes) != manifest['files']['scoped_ledger.json']:
        raise ValueError('Previous ledger is not in the accepted bundle')
    worlds = {}
    for unit in UNITS:
        relative = 'units/' + unit + '.zip'
        archive_bytes = (source / relative).read_bytes()
        if digest(archive_bytes) != manifest['files'][relative]:
            raise ValueError('Historical unit archive hash mismatch')
        with ZipFile(source / relative) as archive:
            if len(set(archive.namelist())) != len(archive.namelist()):
                raise ValueError('Duplicate unit archive entry')
            world_bytes = archive.read('world.json')
            inner = parse(archive.read('manifest.json'))
        world_name = unit.split('_', 2)[2]
        if digest(world_bytes) != inner['files']['worlds/' + world_name + '/world.json']:
            raise ValueError('World is not bound by original unit manifest')
        world = parse(world_bytes)
        if world['noise_family'] != 'student' or world['split'] != 'test_noise':
            raise ValueError('Wrong historical true-world family')
        worlds[unit] = world
    return worlds


def verify_bundle(directory):
    directory = Path(directory).resolve()
    plan = validate_plan(read(directory / 'protocol.json'))
    manifest = read(directory / 'manifest.json')
    expected = {'protocol.json', 'certificates.json', 'source/prior_acceptance.json',
                'source/prior_manifest.json', 'source/prior_ledger.json'} | {
                    'source/units/' + unit + '.zip' for unit in UNITS}
    if manifest.get('schema') != 'ncd.scm-second-moment-bundle.v1' or set(manifest.get('files', {})) != expected or manifest.get('original_objective_achieved') is not False:
        raise ValueError('Incomplete second-moment proof bundle')
    for relative, expected_sha in manifest['files'].items():
        if digest((directory / relative).read_bytes()) != expected_sha:
            raise ValueError('Changed proof artifact: ' + relative)
    worlds = checked_sources(directory / 'source')
    certificates = read(directory / 'certificates.json')
    if set(certificates) != set(UNITS):
        raise ValueError('Missing or extra historical tail witness')
    for unit in UNITS:
        verify_witness(worlds[unit], unit, certificates[unit])
    if plan['unresolved_units'] != list(UNITS):
        raise ValueError('Proof plan unit set changed')
    return {'status': 'verified-scoped', 'prior_finite_second_moments': 295,
            'new_infinite_second_moments': 5, 'classified_observational_worlds': 300,
            'original_objective_achieved': False}


def generate(config):
    config = Path(config).resolve()
    plan = validate_plan(read(config))
    root = Path(plan['project_root']).resolve()
    out = (root / plan['output']).resolve()
    if not out.is_relative_to(root) or out == root or (out / 'manifest.json').exists():
        raise ValueError('Unsafe or completed proof output')
    previous = root / 'runs/scm_population_proof_v1'
    sources = {
        'prior_acceptance.json': (root / 'validation/scm_population_acceptance_v1.json').read_bytes(),
        'prior_manifest.json': (previous / 'manifest.json').read_bytes(),
        'prior_ledger.json': (previous / 'scoped_ledger.json').read_bytes(),
    }
    for unit in UNITS:
        sources['units/' + unit + '.zip'] = (previous / 'units' / (unit + '.zip')).read_bytes()
    write_once(out / 'protocol.json', config.read_bytes())
    for name, data in sources.items():
        write_once(out / 'source' / name, data)
    worlds = checked_sources(out / 'source')
    certificates = {unit: choose_witness(worlds[unit], unit) for unit in UNITS}
    write_once(out / 'certificates.json', encoded(certificates))
    names = ['protocol.json', 'certificates.json'] + ['source/' + name for name in sources]
    if sum((out / name).stat().st_size for name in names) > plan['artifact_budget_bytes']:
        raise ValueError('Proof artifact budget exceeded')
    write_once(out / 'manifest.json', encoded({
        'schema': 'ncd.scm-second-moment-bundle.v1',
        'files': {name: digest((out / name).read_bytes()) for name in names},
        'original_objective_achieved': False,
    }))
    return verify_bundle(out)