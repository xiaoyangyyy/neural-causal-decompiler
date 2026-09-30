"""Independent-do graph candidate behavior on exact and malformed groups."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncd.independent_do_graph import recover_independent_do_means


def linear_groups():
    direct = np.array([[0,0,0],[.6,0,0],[.4,.5,0]])
    total = np.linalg.inv(np.eye(3)-direct)
    groups = {}
    for i in range(3):
        for level in (-1,1):
            groups[(i,level)] = np.tile(level*total[:,i], (4,1))
    return groups


def test_exact_population_group_means_recover_graph():
    result = recover_independent_do_means(linear_groups(), 3)
    assert result["graph_source_target"] == [[0,1,1],[0,0,1],[0,0,0]]
    assert result["acyclic"] is True
    assert result["exogenous_pairing_used"] is False
    assert result["finite_sample_graph_guarantee"] is False


def test_missing_or_wrong_do_groups_rejected():
    groups = linear_groups()
    groups.pop((0,-1))
    with pytest.raises(ValueError, match="group"):
        recover_independent_do_means(groups, 3)
    groups = linear_groups()
    groups[(0,1)][:,0] = 0
    with pytest.raises(ValueError, match="coordinate"):
        recover_independent_do_means(groups, 3)


def test_singular_empirical_total_effect_rejected():
    groups = linear_groups()
    for level in (-1,1):
        groups[(1,level)][:] = 0
        groups[(1,level)][:,1] = level
        groups[(2,level)][:] = 0
        groups[(2,level)][:,2] = level
    # Make two columns identical off the do diagonal is impossible while all
    # do coordinates track; a non-finite or singular case must still be safe.
    groups[(0,1)][:,1] = 1
    groups[(0,-1)][:,1] = -1
    groups[(1,1)][:,0] = 1
    groups[(1,-1)][:,0] = -1
    with pytest.raises(ValueError, match="singular"):
        recover_independent_do_means(groups, 3)