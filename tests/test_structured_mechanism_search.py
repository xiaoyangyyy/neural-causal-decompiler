import numpy as np
from ncd.mechanisms import structured_symbolic_fit
from ncd.mechanism_search_experiment import MechanismSearchConfig,run_mechanism_search,verify_mechanism_search


def test_structured_fit_recovers_one_operator_per_parent_and_interaction():
    rng=np.random.default_rng(4)
    x=rng.uniform(-2,2,(1200,3))
    y=.2+.7*np.sin(x[:,0])-.4*x[:,1]**2+.3*x[:,0]*x[:,1]
    expression,atoms,history=structured_symbolic_fit(x,y,(0,1),max_terms=5,beam_width=48)
    assert [a["atom"] for a in atoms]==["constant","sin:0","square:1","interaction:0,1"]
    assert np.mean((expression.evaluate(x)-y)**2)<1e-20
    assert history


def test_mechanism_search_quick_run_replays(tmp_path):
    config=MechanismSearchConfig.quick()
    summary=run_mechanism_search(tmp_path/"run",config)
    assert summary["world_count"]==1
    assert set(summary["aggregate"])=={"baseline","structured"}
    verified=verify_mechanism_search(tmp_path/"run")
    assert verified["status"]=="verified"
    assert verified["worlds"]==1