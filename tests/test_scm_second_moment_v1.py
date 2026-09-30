"""Checks for strict Student5 second-moment tail witnesses and provenance."""
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile
import copy
import json
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation/scm_second_moment_package_v1'))
from scm_second_moment.proof import (
    UNITS, checked_sources, choose_witness, polynomials, verify_witness,
)

PRIOR = ROOT / 'runs/scm_population_proof_v1'


def world(unit):
    with ZipFile(PRIOR / 'units' / (unit + '.zip')) as archive:
        return json.loads(archive.read('world.json'))


def test_all_five_historical_unresolved_worlds_have_strict_divergent_tails():
    expected = [3, 4, 4, 3, 4]
    for unit, degree in zip(UNITS, expected):
        actual = world(unit)
        certificate = choose_witness(actual, unit)
        assert verify_witness(actual, unit, certificate)
        assert certificate['degree'] == degree
        assert Q(certificate['leading_absolute_lower']) > 0
        assert Q(certificate['truncated_second_moment_coefficient']) > 0
        assert certificate['truncated_growth_exponent'] == 2 * degree - 5
        changed = dict(certificate, degree=2)
        with pytest.raises(ValueError, match='differs'):
            verify_witness(actual, unit, changed)


def test_accepted_source_binding_and_unit_archive_tamper(tmp_path):
    source = tmp_path / 'source'
    (source / 'units').mkdir(parents=True)
    shutil.copyfile(ROOT / 'validation/scm_population_acceptance_v1.json',
                    source / 'prior_acceptance.json')
    shutil.copyfile(PRIOR / 'manifest.json', source / 'prior_manifest.json')
    shutil.copyfile(PRIOR / 'scoped_ledger.json', source / 'prior_ledger.json')
    for unit in UNITS:
        shutil.copyfile(PRIOR / 'units' / (unit + '.zip'),
                        source / 'units' / (unit + '.zip'))
    assert set(checked_sources(source)) == set(UNITS)
    archive = source / 'units' / (UNITS[0] + '.zip')
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='archive hash'):
        checked_sources(source)


def test_false_student_or_structural_premises_rejected():
    actual = world(UNITS[0])
    changed = copy.deepcopy(actual)
    changed['noise_family'] = 'gaussian'
    with pytest.raises(ValueError, match='Student5'):
        polynomials(changed, 1)
    changed = copy.deepcopy(actual)
    changed['scales'][0] = 2
    with pytest.raises(ValueError, match='scales'):
        polynomials(changed, 1)
    changed = copy.deepcopy(actual)
    changed['equations'][0][0]['coefficient'] = 0
    with pytest.raises(ValueError, match='Zero structural'):
        polynomials(changed, 1)
    with pytest.raises(ValueError, match='exogenous root'):
        polynomials(actual, 0)