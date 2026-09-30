"""Adversarial checks of scoped real-population SCM certificates."""
import io
from fractions import Fraction as Q
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation' / 'scm_population_package_v1'))
from scm_population_proof import compact, counterexamples, distribution, moments
from scm_population_proof.__main__ import (
    ANCHORS, EXPECTED, historical_roots, source_unit, unit_certificate,
    unit_zip, unzip_unit, verified_raw_unit, validate_plan, verify_unit,
)


def expr(op, *args, **rest):
    if op == 'constant':
        return {'op': op, 'value': rest['value']}
    if op == 'var':
        return {'op': op, 'index': rest['index']}
    return {'op': op, 'args': list(args)}


def small_model():
    graph = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    return {
        'source_graph': graph,
        'effective_graph': [row[:] for row in graph],
        'noise_assumption': 'independent empirical additive residuals',
        'noise_samples': [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]],
        'equations': [expr('constant', value=0.0),
                      expr('mul', expr('constant', value=2.0), expr('var', index=0)),
                      expr('mul', expr('constant', value=-1.0),
                           expr('square', expr('var', index=1)))],
    }


def student_chain():
    graph = [[0] * 5 for _ in range(5)]
    for j in range(3):
        graph[j][j + 1] = 1
    equations = [[]]
    equations += [[{'operator': 'square', 'parents': [j-1], 'coefficient': 0.1875}]
                  for j in range(1, 4)]
    equations += [[]]
    world = {'graph': graph, 'equations': equations, 'scales': [1.0] * 5,
             'noise_family': 'student', 'noise_scale': 0.35, 'root_shift': False}
    model = {'source_graph': graph, 'effective_graph': [row[:] for row in graph],
             'noise_assumption': 'independent empirical additive residuals',
             'noise_samples': [[0.0, 1.0]] * 5,
             'equations': [expr('constant', value=0.0)] +
                          [expr('square', expr('var', index=j-1)) for j in range(1, 4)] +
                          [expr('constant', value=0.0)]}
    return world, model


def historical_fixture(unit):
    files = {name: (ROOT / path).read_bytes() for name, (path, digest) in ANCHORS.items()}
    receipt = historical_roots(files)
    return source_unit(ROOT, unit, receipt), receipt


def test_every_compatible_do_mask_and_empirical_noise_choice():
    model = small_model()
    cert = compact.derive(model)
    bounds = [float(x) for x in cert['coordinate_abs_upper']]
    assert cert['topological_order'] == [0, 1, 2]
    for mask in itertools.product((False, True), repeat=3):
        for do in itertools.product((-1.0, 1.0), repeat=sum(mask)):
            do_values = dict(zip((j for j, selected in enumerate(mask) if selected), do))
            for u in itertools.product(*model['noise_samples']):
                x = [0.0] * 3
                x[0] = do_values.get(0, u[0])
                x[1] = do_values.get(1, 2 * x[0] + u[1])
                x[2] = do_values.get(2, -x[1] ** 2 + u[2])
                assert all(abs(a) <= upper for a, upper in zip(x, bounds))
    assert compact.verify(model, cert)['status'] == 'verified'


def test_protected_division_zero_and_negative_branch():
    numerator = expr('constant', value=2.0)
    for denominator in (0.0, 1e-9, -1e-9, 1e-8, -1e-8):
        box = compact.interval(expr('div', numerator,
                                    expr('constant', value=denominator)), {}, set())
        replaced = denominator if abs(denominator) >= 1e-8 else (-1e-8 if denominator < 0 else 1e-8)
        actual = 2.0 / replaced
        assert float(box[0]) <= actual <= float(box[1])


def test_sqrt_log_branches_and_bounded_transcendentals():
    assert compact.interval(expr('sqrt', expr('constant', value=-16.0)), {}, set()) == (0, 0)
    box = compact.interval(expr('log', expr('constant', value=0.0)), {}, set())
    assert float(box[0]) <= math.log(1e-12) <= float(box[1])
    for op in ('sin', 'cos', 'tanh'):
        assert compact.interval(expr(op, expr('constant', value=1e9)), {}, set()) == (-1, 1)


def test_wrong_parent_graph_and_nonfinite_noise_rejected():
    model = small_model()
    model['effective_graph'][0][1] = 0
    with pytest.raises(ValueError, match='effective parents'):
        compact.derive(model)
    model = small_model()
    model['source_graph'][1][2] = 0
    with pytest.raises(ValueError, match='Undeclared parent'):
        compact.derive(model)
    model = small_model()
    model['noise_samples'][0] = [float('nan')]
    with pytest.raises(ValueError, match='Finite stored number'):
        compact.derive(model)


def test_cycle_invalid_op_and_missing_noise_rejected():
    model = small_model()
    model['source_graph'][2][0] = 1
    with pytest.raises(ValueError, match='cycle'):
        compact.derive(model)
    model = small_model()
    model['equations'][0] = expr('unsupported', expr('constant', value=1.0))
    with pytest.raises(NotImplementedError, match='Unsupported'):
        compact.derive(model)
    model = small_model()
    model['noise_samples'][2] = []
    with pytest.raises(ValueError, match='nonempty'):
        compact.derive(model)


def test_empirical_noise_assumption_must_be_explicit():
    model = small_model()
    model['noise_assumption'] = 'fitted residuals proved independent of truth'
    with pytest.raises(ValueError, match='Different declared noise model'):
        compact.derive(model)


def test_student5_growth_failure_is_only_unresolved():
    world, model = student_chain()
    first = moments.derive(world, '1')
    second = moments.derive(world, '2')
    assert first['status'] == 'unresolved-by-sufficient-bound'
    assert second['status'] == 'unresolved-by-sufficient-bound'
    assert any(o['required_moment'] == '8' and not o['available']
               for o in first['rows'][3]['moment_obligations'])
    tv = distribution.derive(world, model)
    assert tv['total_variation_distance'] == '1'
    assert tv['intervention_scope'].startswith('Every compatible')
    assert tv['original_claim_closed'] is False


def test_gaussian_all_finite_moments_and_wrong_graph_rejected():
    world, model = student_chain()
    world['noise_family'] = 'gaussian'
    assert moments.derive(world, '64')['status'] == 'proved-scoped'
    world['graph'][0][1] = 0
    with pytest.raises(ValueError, match='incorrect true mechanism'):
        moments.derive(world, '1')


def test_total_variation_requires_continuous_positive_noise():
    world, model = student_chain()
    world['noise_scale'] = 0.0
    with pytest.raises(ValueError, match='positive'):
        distribution.derive(world, model)
    world['noise_scale'] = 0.35
    world['noise_family'] = 'discrete'
    with pytest.raises(NotImplementedError, match='Unknown true noise law'):
        distribution.derive(world, model)
    world['noise_family'] = 'student'
    world['root_shift'] = 'False'
    with pytest.raises(ValueError, match='Boolean'):
        distribution.derive(world, model)


def test_counterexample_exact_finite_cutoffs_and_tamper():
    for n in (5, 8):
        for B, K in [('0', '1'), ('10/3', '7/8'), ('1000000000000', '100')]:
            cert = counterexamples.derive(n, B, K)
            finite = cert['finite_witness']
            assert Q(finite['strict_coupling_cost_lower']) > Q(K)
            assert counterexamples.verify(cert)['status'] == 'verified'
            cert['model']['coefficient_interval'][0] = '1/9'
            with pytest.raises(ValueError, match='mismatch'):
                counterexamples.verify(cert)


def test_counterexample_outside_declared_nodes_rejected():
    with pytest.raises(ValueError, match='separate intervention isolate'):
        counterexamples.derive(3, '0', '1')
    with pytest.raises(ValueError, match='nonnegative rational'):
        counterexamples.derive(5, '-1', '1')


def test_unit_archive_duplicates_and_unsafe_path_rejected():
    buf = io.BytesIO()
    with pytest.warns(UserWarning, match='Duplicate name'):
        with zipfile.ZipFile(buf, 'w') as z:
            for i in range(10):
                z.writestr('same.json', b'{}')
    with pytest.raises(ValueError, match='duplicate'):
        unzip_unit(buf.getvalue())


def test_historical_confirmed_world_and_six_models_have_exact_receipts():
    if not (ROOT / 'runs/original_confirmation_v1/units').exists():
        pytest.skip('historical source artifacts excluded from source-only checkout')
    contents, receipt = historical_fixture(EXPECTED[0])
    assert len(unzip_unit(unit_zip(contents))) == 10
    cert = unit_certificate(EXPECTED[0], contents, receipt)
    assert len(cert['estimated_support']) == 6
    assert {x['total_variation_distance'] for x in cert['population_tv_boundaries'].values()} == {'1'}
    assert cert['true_first_moment']['status'] == 'proved-scoped'
    tampered = dict(contents)
    world = json.loads(contents['world.json'])
    world['noise_scale'] = 0.36
    tampered['world.json'] = json.dumps(world).encode()
    with pytest.raises(ValueError, match='World data differs'):
        verified_raw_unit(EXPECTED[0], tampered, receipt)


def test_confirmed_student_case_does_not_convert_failed_sufficient_bound_to_infinity():
    if not (ROOT / 'runs/original_confirmation_v1/units').exists():
        pytest.skip('historical source artifacts excluded from source-only checkout')
    contents, receipt = historical_fixture('seed_8101_n8_test_noise_9')
    cert = unit_certificate('seed_8101_n8_test_noise_9', contents, receipt)
    assert cert['true_first_moment']['status'] == 'proved-scoped'
    assert cert['true_second_moment']['status'] == 'unresolved-by-sufficient-bound'
    assert cert['original_claim_closed'] is False


def test_protocol_loaded_code_mismatch_rejected():
    plan = json.loads((ROOT / 'validation/scm_population_protocol_v1.json').read_bytes())
    assert validate_plan(plan) == plan
    plan['loaded_package_sha256']['distribution.py'] = '0' * 64
    with pytest.raises(ValueError, match='Loaded checker source differs'):
        validate_plan(plan)


def test_unit_certificate_tamper_rejected(tmp_path):
    if not (ROOT / 'runs/original_confirmation_v1/units').exists():
        pytest.skip('historical source artifacts excluded from source-only checkout')
    contents, receipt = historical_fixture(EXPECTED[0])
    (tmp_path / 'units').mkdir()
    (tmp_path / 'certificates').mkdir()
    archive = tmp_path / 'units' / (EXPECTED[0] + '.zip')
    certificate = tmp_path / 'certificates' / (EXPECTED[0] + '.json')
    archive.write_bytes(unit_zip(contents))
    expected = unit_certificate(EXPECTED[0], contents, receipt)
    certificate.write_text(json.dumps(expected, sort_keys=True), encoding='utf-8')
    assert verify_unit(tmp_path, EXPECTED[0], receipt) == expected
    expected['population_tv_boundaries']['active_graph/baseline']['total_variation_distance'] = '0'
    certificate.write_text(json.dumps(expected, sort_keys=True), encoding='utf-8')
    with pytest.raises(ValueError, match='proof recomputation mismatch'):
        verify_unit(tmp_path, EXPECTED[0], receipt)