"""Executable factorized skeleton and orientation graph programs."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .rules import Rule
from .graph_model import SWAP_LABELS
from .graphs import acyclic_projection

@dataclass(frozen=True)
class FactorizedGraphProgram:
    skeleton:Rule
    orientation:Rule
    aggregation:str="and"
    def __post_init__(self):
        if self.skeleton.names!=self.orientation.names or self.aggregation not in ("and","or"):raise ValueError("Invalid factorized graph program")
    @property
    def complexity(self):return self.skeleton.complexity+self.orientation.complexity+1
    def predict(self,features):
        x=np.asarray(features,float);n=x.shape[0]
        if x.shape!=(n,n,len(self.skeleton.names)):raise ValueError("Factorized feature mismatch")
        raw=np.zeros((n,n),bool);scores=np.zeros((n,n),float)
        for i,j in combinations(range(n),2):
            pair=np.stack([x[i,j],x[j,i]])
            sk=self.skeleton.predict(pair)!=0
            present=bool(np.all(sk) if self.aggregation=="and" else np.any(sk))
            if not present:continue
            orient=self.orientation.predict(pair);left=int(orient[0]);right=int(SWAP_LABELS[int(orient[1])])
            cls=left if left==right and left in (1,2,3) else 3
            if cls==1:raw[i,j]=True;scores[i,j]=1.
            elif cls==2:raw[j,i]=True;scores[j,i]=1.
            else:raw[i,j]=raw[j,i]=True
        directed=raw&~raw.T;dag,_=acyclic_projection(np.where(directed,scores,0.),threshold=1e-12)
        return dag|(raw&raw.T)
    def to_dict(self):return {"skeleton":self.skeleton.to_dict(),"orientation":self.orientation.to_dict(),"aggregation":self.aggregation,"complexity":self.complexity,"semantics":"skeleton then swap-consistent orientation then acyclic projection"}
    @classmethod
    def from_dict(cls,d):
        p=cls(Rule.from_dict(d["skeleton"]),Rule.from_dict(d["orientation"]),d["aggregation"])
        if "complexity" in d and d["complexity"]!=p.complexity:raise ValueError("Factorized complexity mismatch")
        return p