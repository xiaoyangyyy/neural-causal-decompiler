"""Finite-difference exclusion of low-degree programs for one frozen Tanh network.

A strict eight-point rational interval certificate rules out every real
polynomial of total degree at most six on the declared input box.
"""
from fractions import Fraction as Q
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import os

from . import __version__

CHECKPOINT_SHA = '20b62f0bd0f6ba0e36ee678fbd7401e8425fdb8720e28c80b4b588382e671b61'
ORIGINAL = 'runs/active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline/mechanism_0.pt'
POINTS = tuple(str(Q(-37, 40) + j * Q(11, 40)) for j in range(8))
WEIGHTS = tuple((-1) ** (7 - j) * comb(7, j) for j in range(8))
PACKAGE_FILES = ('__init__.py', 'proof.py', '__main__.py')


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


def loaded_package_sha():
    root = Path(__file__).resolve().parent
    return {name: digest((root / name).read_bytes()) for name in PACKAGE_FILES}


def installed_ncd_sha():
    import ncd
    if ncd.__version__ != '0.59.0':
        raise ValueError('Wrong historical neural interval checker version')
    root = Path(ncd.__file__).resolve().parent
    if 'site-packages' not in root.parts:
        raise ValueError('Historical NCD must come from an installed package')
    return {path.relative_to(root).as_posix(): digest(path.read_bytes())
            for path in sorted(root.rglob('*.py'))}


def validate_plan(plan):
    if plan.get('schema') != 'ncd.polynomial-degree-barrier-plan.v1' or plan.get('version') != __version__:
        raise ValueError('Wrong degree barrier protocol')
    if plan.get('output') != 'runs/frozen_mechanism_polynomial_barrier_v1' or plan.get('original_objective_achieved') is not False:
        raise ValueError('Wrong proof output or original claim scope')
    if plan.get('checkpoint_sha256') != CHECKPOINT_SHA or plan.get('degree_bound') != 6:
        raise ValueError('Frozen target or polynomial degree changed')
    if plan.get('points') != list(POINTS) or plan.get('epsilon_normalized') != '1/100':
        raise ValueError('Finite difference witness or error threshold changed')
    if plan.get('input_domain') != [['-1', '1']] * 3 or plan.get('original_atomic_claim_counts') != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Declared domain or original ledger changed')
    if 'site-packages' not in Path(__file__).resolve().parts or plan.get('loaded_package_sha256') != loaded_package_sha():
        raise ValueError('Degree barrier checker is not the installed frozen package')
    if plan.get('installed_ncd_source_sha256') != installed_ncd_sha():
        raise ValueError('Historical neural interval implementation changed')
    if plan.get('memory_bytes') != 8589934592 or plan.get('artifact_budget_bytes') != 8589934592 or plan.get('threads') != 2 or not 0 < plan.get('stage_seconds', 0) <= 43200:
        raise ValueError('Invalid resource budget')
    return plan


def finite_difference_intervals(intervals):
    if len(intervals) != 8:
        raise ValueError('Exactly eight point enclosures required')
    lower, upper = Q(0), Q(0)
    for weight, pair in zip(WEIGHTS, intervals):
        if len(pair) != 2:
            raise ValueError('Malformed point interval')
        lo, hi = map(Q, pair)
        if lo > hi:
            raise ValueError('Reversed point interval')
        lower += weight * (lo if weight >= 0 else hi)
        upper += weight * (hi if weight >= 0 else lo)
    return lower, upper


def separated_margin(intervals, output_scale):
    delta_lo, delta_hi = finite_difference_intervals(intervals)
    separation = delta_lo if delta_lo > 0 else -delta_hi if delta_hi < 0 else Q(0)
    scale = Q(output_scale)
    if scale <= 0:
        raise ValueError('Nonpositive training output scale')
    budget = sum(abs(w) for w in WEIGHTS) * scale / 100
    if separation <= budget:
        raise ValueError('The checked eight points do not exclude this program family')
    return delta_lo, delta_hi, separation, budget, separation - budget


def derive(checkpoint):
    from ncd.frozen_mechanism_proof import export_mechanism, neural_interval
    from ncd.proof_intervals import Interval
    checkpoint = Path(checkpoint)
    if digest(checkpoint.read_bytes()) != CHECKPOINT_SHA:
        raise ValueError('Wrong frozen neural checkpoint')
    network = export_mechanism(checkpoint)
    if network['checkpoint_sha256'] != CHECKPOINT_SHA or network['parents'] != [1, 2]:
        raise ValueError('Wrong two-parent frozen network')
    values = []
    for point in POINTS:
        t = Q(point)
        if not Q(-1) <= t <= Q(1):
            raise ValueError('Finite difference point outside the closed box')
        box = [Interval.point(Q(0)), Interval.point(t), Interval.point(t)]
        values.append(neural_interval(network, box).to_dict())
    low, high, separation, budget, margin = separated_margin(values, network['output_scale'])
    return {
        'schema': 'ncd.frozen-polynomial-degree-barrier.v1', 'status': 'refuted-scoped',
        'claim_refuted': 'A real polynomial in the three declared inputs of total degree at most six uniformly approximates this frozen child neural mechanism within one percent of its training output scale on [-1,1]^3',
        'checkpoint_sha256': CHECKPOINT_SHA,
        'network_export_sha256': digest(encoded(network)),
        'parents': [1, 2], 'input_domain': [['-1', '1']] * 3,
        'degree_bound': 6, 'epsilon_normalized': '1/100',
        'output_training_scale': network['output_scale'],
        'points': list(POINTS), 'difference_weights': list(WEIGHTS),
        'network_point_enclosures': values,
        'seventh_difference_enclosure': [str(low), str(high)],
        'seventh_difference_absolute_lower': str(separation),
        'all_point_error_difference_upper': str(budget),
        'strict_incompatibility_margin_lower': str(margin),
        'proof': 'Restrict any total-degree-at-most-six polynomial to (x0,x1,x2)=(0,t,t). Its seventh equally spaced finite difference is exactly zero. If every point error were at most output_scale/100, the seventh difference of the neural values would have absolute value at most the sum of absolute binomial weights times that radius. The independently recomputed rational enclosures violate that bound strictly.',
        'semantic_scope': 'Ideal real evaluation of exact stored binary weights and mathematical Tanh; no device-rounding guarantee',
        'not_refuted': ['non-polynomial short programs', 'piecewise programs', 'higher-degree polynomials',
                        'other frozen networks or worlds', 'true causal mechanism recovery',
                        'global MDL minimality or uniqueness'],
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing immutable proof file differs')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def verify_bundle(directory):
    directory = Path(directory).resolve()
    plan = validate_plan(read(directory / 'protocol.json'))
    manifest = read(directory / 'manifest.json')
    expected_files = {'protocol.json', 'certificate.json', 'source/mechanism_0.pt'}
    if manifest.get('schema') != 'ncd.polynomial-degree-barrier-bundle.v1' or set(manifest.get('files', {})) != expected_files or manifest.get('original_objective_achieved') is not False:
        raise ValueError('Incomplete polynomial barrier bundle')
    for relative, expected in manifest['files'].items():
        if digest((directory / relative).read_bytes()) != expected:
            raise ValueError('Changed proof artifact: ' + relative)
    if digest((directory / 'source/mechanism_0.pt').read_bytes()) != plan['checkpoint_sha256']:
        raise ValueError('Proof bundle checkpoint differs from protocol')
    if read(directory / 'certificate.json') != derive(directory / 'source/mechanism_0.pt'):
        raise ValueError('Independent neural interval replay disagrees')
    return {'status': 'verified-scoped', 'conclusion': 'refuted-scoped',
            'degree_bound': 6, 'points': 8, 'original_objective_achieved': False}


def generate(config):
    config = Path(config).resolve()
    plan = validate_plan(read(config))
    root = Path(plan['project_root']).resolve()
    out = (root / plan['output']).resolve()
    if not root.is_dir() or not out.is_relative_to(root) or out == root or (out / 'manifest.json').exists():
        raise ValueError('Unsafe or completed proof output')
    checkpoint = (root / ORIGINAL).read_bytes()
    if digest(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('Historical frozen checkpoint changed')
    write_once(out / 'protocol.json', config.read_bytes())
    write_once(out / 'source/mechanism_0.pt', checkpoint)
    write_once(out / 'certificate.json', encoded(derive(out / 'source/mechanism_0.pt')))
    names = ('protocol.json', 'certificate.json', 'source/mechanism_0.pt')
    if sum((out / name).stat().st_size for name in names) > plan['artifact_budget_bytes']:
        raise ValueError('Proof artifact budget exceeded')
    write_once(out / 'manifest.json', encoded({
        'schema': 'ncd.polynomial-degree-barrier-bundle.v1',
        'files': {name: digest((out / name).read_bytes()) for name in names},
        'original_objective_achieved': False,
    }))
    return verify_bundle(out)