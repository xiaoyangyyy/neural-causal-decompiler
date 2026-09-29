"""Actual PySR multi-output distillation, exported as safe executable ASTs."""
from pathlib import Path
import numpy as np
from .cdir import Node
from .io import save_json,read_json
from .research_runtime import configure_julia

def from_sympy(expr,variables):
    import sympy as sp
    if expr.is_Number:return Node("constant",value=float(expr))
    if expr.is_Symbol:
        if str(expr) not in variables:raise ValueError("Unknown PySR variable")
        return Node("var",index=variables.index(str(expr)))
    args=[from_sympy(a,variables) for a in expr.args]
    if expr.func in (sp.Add,sp.Mul):
        op="add" if expr.func==sp.Add else "mul";node=args[0]
        for child in args[1:]:node=Node(op,(node,child))
        return node
    if expr.func==sp.Pow:
        exponent=expr.args[1]
        if not exponent.is_Integer or int(exponent)<1 or int(exponent)>20:raise ValueError("Unsupported PySR power")
        node=args[0]
        for _ in range(int(exponent)-1):node=Node("mul",(node,args[0]))
        return node
    for func,op in ((sp.sin,"sin"),(sp.cos,"cos"),(sp.tanh,"tanh"),(sp.Abs,"abs")):
        if expr.func==func:return Node(op,tuple(args))
    raise ValueError(f"Unsupported PySR expression: {expr.func}")

class SymbolicScores:
    def __init__(self,expressions,names):self.expressions=list(expressions);self.names=tuple(names)
    def scores(self,x):
        return np.column_stack([np.broadcast_to(e.evaluate(x),(len(x),)) for e in self.expressions])
    def predict(self,x):return self.scores(x).argmax(1)
    def to_dict(self):
        return {"names":list(self.names),"expressions":[e.to_dict() for e in self.expressions],
                "complexity":sum(e.complexity for e in self.expressions),"engine":"PySR"}
    @classmethod
    def from_dict(cls,d):return cls([Node.from_dict(e) for e in d["expressions"]],d["names"])

def _fit_pysr_in_process(x,teacher_probabilities,names,directory,*,iterations=12,seed=42):
    configure_julia()
    from pysr import PySRRegressor
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    variables=[f"f{i}" for i in range(x.shape[1])]
    regressor=PySRRegressor(niterations=iterations,populations=3,population_size=24,ncycles_per_iteration=100,
        maxsize=13,binary_operators=["+","-","*"],unary_operators=["abs"],parallelism="serial",
        deterministic=True,random_state=seed,model_selection="best",progress=False,verbosity=0,
        input_stream="devnull",output_directory=str(directory/"native"),run_id="distillation")
    regressor.fit(np.asarray(x,dtype=np.float64),np.asarray(teacher_probabilities,dtype=np.float64),variable_names=variables)
    equations=regressor.sympy()
    if not isinstance(equations,list):equations=[equations]
    model=SymbolicScores([from_sympy(e,variables) for e in equations],names)
    native=regressor.predict(x)
    np.testing.assert_allclose(model.scores(x),native,rtol=1e-4,atol=1e-4)
    save_json(directory/"program.json",{**model.to_dict(),"equations":[str(e) for e in equations],
        "seed":seed,"iterations":iterations,"supervision":"frozen_neural_probabilities",
        "export_matches_native":True})
    np.savez_compressed(directory/"training.npz",features=x,teacher_probabilities=teacher_probabilities,
                        exported_scores=model.scores(x))
    return model

def fit_pysr(x,teacher_probabilities,names,directory,*,iterations=12,seed=42):
    """Run Julia in a separate process to avoid torch/Julia loader conflicts."""
    import subprocess
    import sys
    directory=Path(directory).resolve()
    if (directory/"program.json").exists():raise FileExistsError("PySR evidence already exists")
    directory.mkdir(parents=True,exist_ok=True)
    save_json(directory/"request.json",{"names":list(names),"iterations":iterations,"seed":seed})
    np.savez_compressed(directory/"request.npz",features=x,probabilities=teacher_probabilities)
    subprocess.run([sys.executable,"-u","-m","ncd.pysr_worker",str(directory)],check=True)
    return SymbolicScores.from_dict(read_json(directory/"program.json"))
