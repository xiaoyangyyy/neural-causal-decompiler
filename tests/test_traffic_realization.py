import numpy as np

from ncd.traffic_realization import (
    ACTIONS,
    TrafficConfig,
    run_certified_traffic,
    traffic_step,
    verify_certified_traffic,
)


def test_traffic_actions_have_declared_dynamical_effects():
    state = np.full(4, 0.5)
    normal = traffic_step(state, ACTIONS.index("normal"))
    demand = traffic_step(state, ACTIONS.index("demand_up"))
    incident = traffic_step(state, ACTIONS.index("incident_link1"))
    closure = traffic_step(state, ACTIONS.index("closure_link3"))
    assert demand[0] > normal[0]
    assert incident[1] > normal[1]
    assert closure[3] > normal[3]
    assert np.all((0 <= closure) & (closure <= 1))


def test_quick_traffic_training_certificate_and_replay(tmp_path):
    output = tmp_path / "traffic"
    summary = run_certified_traffic(output, TrafficConfig.quick(seed=53))
    assert summary["all_training_pairs_fit"]
    assert summary["all_certificates_verified"]
    assert summary["active_exact_cases"] == 1
    assert summary["limited_unresolved_pairs"] > 0
    checked = verify_certified_traffic(output)
    assert checked["status"] == "verified"
    assert checked["certificates_verified"] == 2
