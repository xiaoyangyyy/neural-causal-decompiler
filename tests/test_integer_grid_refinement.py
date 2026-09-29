"""Exact replay of state-only and joint integer-grid improvements."""
from copy import deepcopy
from pathlib import Path
import pytest
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.integer_grid_refinement import verify_refinement
from ncd.io import read_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
BASELINE = ROOT/'validation'/'automatic_affine128_dev'/'certificate.json'
OUTPUT = ROOT/'runs'/'integer_grid_refinement_v1'


def test_exact_refinement_replays_two_independent_upper_models():
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    baseline = read_json(BASELINE)
    results = []
    for name in ('state_only','state_action'):
        proposal = read_json(OUTPUT/name/'proposal.json')
        certificate = read_json(OUTPUT/name/'certificate.json')
        results.append(verify_refinement(system,baseline,proposal,certificate))
    assert [r['new_upper_bound'] for r in results] == ['27648','27216']
    assert [r['action_bins'] for r in results] == [128,512]
    assert [r['numeric_tuples_searched'] for r in results] == [4851,9702]
    proposal = read_json(OUTPUT/'state_action'/'proposal.json')
    certificate = read_json(OUTPUT/'state_action'/'certificate.json')
    tampered = deepcopy(proposal)
    tampered['candidate_upper_bound'] = '27215'
    with pytest.raises(ValueError,match='proposal replay mismatch'):
        verify_refinement(system,baseline,tampered,certificate)
    for key,value in (('epsilon','1'),('packing_axes',3)):
        tampered_certificate = deepcopy(certificate)
        tampered_certificate[key] = value
        with pytest.raises(ValueError,match='certificate parameter mismatch'):
            verify_refinement(system,baseline,proposal,tampered_certificate)
