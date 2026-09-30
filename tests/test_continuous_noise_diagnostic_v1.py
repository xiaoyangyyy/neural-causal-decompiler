"""Checks that continuous-noise calibration matches the executed SCM equations."""
from pathlib import Path
import sys
import shutil

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from run_continuous_noise_diagnostic_v1 import empirical_joint_energy, symbolic_residuals
from verify_continuous_noise_posthoc_v1 import verify
from ncd.cdir import Node
from ncd.mechanisms import ExplicitSCM


def test_symbolic_residuals_use_executable_equations():
    equations = [
        Node("constant", value=2.),
        Node("add", (
            Node("var", index=0),
            Node("constant", value=-1.),
        )),
    ]
    graph = np.array([[0, 1], [0, 0]], dtype=bool)
    scm = ExplicitSCM(equations, [np.array([0.]), np.array([0.])], graph)
    observed = np.array([[3., 2.5], [4., 1.5], [-1., 0.]])
    result = symbolic_residuals(observed, scm)
    expected = np.column_stack((
        observed[:, 0] - 2.,
        observed[:, 1] - (observed[:, 0] - 1.),
    ))
    np.testing.assert_array_equal(result, expected)

def test_joint_energy_detects_dependence_with_equal_marginals():
    x = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
    y = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    assert empirical_joint_energy(x, x) == 0.
    assert empirical_joint_energy(x, y) > .1


def test_posthoc_replay_rejects_tampered_frozen_mechanism(tmp_path):
    source = ROOT / "validation" / "continuous_noise_posthoc_package_v1"
    target = tmp_path / "bundle"
    shutil.copytree(source, target)
    checkpoint = target / "model" / "baseline" / "mechanism_0.pt"
    raw = checkpoint.read_bytes()
    checkpoint.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="Bundled file hash mismatch"):
        verify(target)
