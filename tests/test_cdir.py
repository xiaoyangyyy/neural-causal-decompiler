import numpy as np
import pytest
from ncd.cdir import Node,statistical_primitives,canonical_expression,algebraic_equivalence,numeric_equivalence

def test_composed_residual_and_statistical_typing():
    rng=np.random.default_rng(1);x=rng.normal(size=64);d=np.column_stack([x,x*x+rng.normal(0,.1,64)])
    primitives=statistical_primitives()
    e=primitives["resdep_xy"]
    assert e.args[1].op=="residual"
    assert e.args[1].args[1].op=="regress"
    assert e.evaluate(d)<primitives["resdep_yx"].evaluate(d)
    restored=Node.from_dict(e.to_dict())
    assert restored.evaluate(d)==e.evaluate(d)
    with pytest.raises(ValueError):Node("lt",(Node("var"),Node("constant",value=0)))
    with pytest.raises(ValueError):Node("and",(Node("constant"),Node("constant")))

def test_arithmetic_equivalence_and_protected_division():
    x,y=Node("var",index=0),Node("var",index=1)
    a=Node("mul",(x,Node("add",(x,y))))
    b=Node("add",(Node("square",(x,)),Node("mul",(x,y))))
    assert algebraic_equivalence(a,b)
    assert canonical_expression(a)==canonical_expression(b)
    data=np.random.default_rng(2).normal(size=(32,2))
    assert numeric_equivalence(a,b,data)["equivalent_on_probes"]
    div=Node("div",(Node("constant",value=1),Node("constant",value=0)))
    assert np.isfinite(div.evaluate(data))
    with pytest.raises(ValueError):algebraic_equivalence(div,x)
