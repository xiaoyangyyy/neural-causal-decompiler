"""Occurrence-addressed program execution and typed intermediate interventions.

Addresses are scoped to the exact serialized program, not algebraic equivalence.
Rule expressions have their original full-batch semantics; visited masks say
which rows actually reach each occurrence. Node conditionals execute one branch.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .cdir import Node
from .rules import Rule

def _identity(program):
    return hashlib.sha256(json.dumps(program.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _numeric(value,shape):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("Intervention shape/type mismatch")
    return float(a) if not shape else a.astype(float,copy=True)

def _typed(value,kind,shape):
    if kind=="bool":
        a=np.asarray(value)
        if a.shape!=shape or a.dtype.kind!="b":raise ValueError("Boolean intervention required")
        return bool(a) if not shape else a.copy()
    return _numeric(value,shape)

def _apply(node,values,rows):
    # Rebind operands to distinct leaves rather than a structural-node cache:
    # two equal subexpressions may receive different occurrence interventions.
    columns=[];args=[]
    for child,value in zip(node.args,values):
        if child.kind=="bool":
            args.append(Node("lt",(Node("constant",value=0.),Node("constant",value=1. if value else -1.))))
        elif child.kind=="scalar":args.append(Node("constant",value=float(value)))
        elif child.kind=="vector":
            args.append(Node("var",index=len(columns)));columns.append(np.asarray(value))
        elif child.kind=="matrix":
            start=len(columns);a=np.asarray(value)
            columns.extend(a[:,j] for j in range(a.shape[1]))
            args.append(Node("columns",tuple(Node("var",index=j) for j in range(start,len(columns)))))
        else:raise ValueError("Unsupported operand type")
    data=np.column_stack(columns) if columns else np.empty((rows,0))
    return Node(node.op,tuple(args)).evaluate(data)

@dataclass
class ExecutionTrace:
    program_id:str
    output:object
    records:dict

    def to_dict(self):
        def plain(x):return np.asarray(x).tolist()
        return {"program_id":self.program_id,"output":plain(self.output),
                "records":{k:{"kind":v["kind"],"value":plain(v["value"]),
                              "visited":plain(v["visited"]),"intervened":v["intervened"]}
                           for k,v in self.records.items()}}

class ProgramExecutor:
    def __init__(self,program):
        if not isinstance(program,(Node,Rule)):raise TypeError("Expected Node or Rule")
        self.program=program;self.program_id=_identity(program);self.catalog={};self._nodes={}
        def expression(node,path):
            self.catalog[path]={"kind":node.kind,"op":node.op}
            self._nodes[path]=node
            for i,child in enumerate(node.args):expression(child,path+"/"+str(i))
        def tree(t,path):
            if "label" in t:
                self.catalog[path]={"kind":"label","op":"return"};return
            expression(Node.from_dict(t["expr"]),path+"/expr")
            self.catalog[path+"/predicate"]={"kind":"bool","op":"lt"}
            tree(t["left"],path+"/left");tree(t["right"],path+"/right")
        if isinstance(program,Node):expression(program,"root")
        else:tree(program.tree,"root")

    def address(self,path):
        if path not in self.catalog:raise ValueError("Unknown program occurrence")
        return self.program_id+":"+path

    def execute(self,data,interventions=None):
        data=np.asarray(data,dtype=float)
        if data.ndim!=2 or not np.isfinite(data).all():raise ValueError("Finite matrix required")
        if isinstance(self.program,Rule) and data.shape[1]!=len(self.program.names):
            raise ValueError("Feature schema mismatch")
        patches={}
        for address,value in (interventions or {}).items():
            prefix,sep,path=address.partition(":")
            if not sep or prefix!=self.program_id or path not in self.catalog:raise ValueError("Foreign or unknown intervention")
            if self.catalog[path]["kind"]=="label":raise ValueError("Output-label intervention is not an intermediate intervention")
            patches[path]=value
        # No silently overridden descendant interventions.
        for a in patches:
            if any(b.startswith(a+"/") for b in patches if a!=b):raise ValueError("Overlapping interventions")
        for path in patches:
            if path.endswith("/predicate"):
                expr_path=path[:-len("predicate")]+"expr"
                if any(p==expr_path or p.startswith(expr_path+"/") for p in patches):
                    raise ValueError("Predicate override shadows expression intervention")
        records={};rows=len(data)

        def record(path,kind,value,visited):
            records[self.address(path)]={"kind":kind,"value":np.asarray(value).copy(),
                                       "visited":np.asarray(visited,dtype=bool).copy(),"intervened":path in patches}
            return value

        def expr(node,path,visited):
            shape=() if node.kind in ("scalar","bool") else (rows,)
            if node.kind=="matrix" and path in patches:shape=np.asarray(node.evaluate(data)).shape
            if path in patches:
                return record(path,node.kind,_typed(patches[path],node.kind,shape),visited)
            if node.op in ("var","constant"):value=node.evaluate(data)
            elif node.op=="if":
                condition=expr(node.args[0],path+"/0",visited)
                branch=1 if condition else 2
                value=expr(node.args[branch],path+"/"+str(branch),visited)
            else:
                values=[expr(child,path+"/"+str(i),visited) for i,child in enumerate(node.args)]
                value=_apply(node,values,rows)
            return record(path,node.kind,value,visited)

        def walk(t,path,visited):
            if "label" in t:return record(path,"label",np.full(rows,t["label"],int),visited)
            p=path+"/predicate"
            if p in patches:decision=_typed(patches[p],"bool",(rows,))
            else:
                value=expr(Node.from_dict(t["expr"]),path+"/expr",visited)
                decision=np.broadcast_to(value<t["threshold"],(rows,)).copy()
            record(p,"bool",decision,visited)
            left=walk(t["left"],path+"/left",visited&decision)
            right=walk(t["right"],path+"/right",visited&~decision)
            return np.where(decision,left,right)

        if isinstance(self.program,Node):output=expr(self.program,"root",True)
        else:output=walk(self.program.tree,"root",np.ones(rows,bool))
        # Valid but causally unreachable patches are explicitly reported as unused.
        for path in patches:
            if self.address(path) not in records:
                meta=self.catalog[path]
                kind=meta["kind"]
                shape=() if kind in ("scalar","bool") else (rows,)
                if kind=="matrix":
                    shape=np.asarray(self._nodes[path].evaluate(data)).shape
                value=_typed(patches[path],kind,shape)
                record(path,kind,value,False)
        return ExecutionTrace(self.program_id,output,records)
