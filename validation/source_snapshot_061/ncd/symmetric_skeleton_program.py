"""Graph program with an unordered, swap-invariant skeleton rule."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .rules import Rule
from .graph_model import SWAP_LABELS
from .graphs import acyclic_projection

def symmetric_pair_features(features):
    x=np.asarray(features,float)
    if x.ndim!=3 or x.shape[0]!=x.shape[1]:raise ValueError("Expected square pair-feature tensor")
    rows=[]
    for i,j in combinations(range(x.shape[0]),2):rows.append(np.r_[np.minimum(x[i,j],x[j,i]),np.maximum(x[i,j],x[j,i])])
    return np.stack(rows) if rows else np.empty((0,2*x.shape[2]))

@dataclass(frozen=True)
class SymmetricSkeletonGraphProgram:
    skeleton:Rule
    orientation:Rule
    def __post_init__(self):
        if len(self.skeleton.names)!=2*len(self.orientation.names):raise ValueError("Invalid symmetric graph program schemas")
    @property
    def complexity(self):return self.skeleton.complexity+self.orientation.complexity+1
    def predict(self,features):
        x=np.asarray(features,float);n=x.shape[0]
        if x.shape!=(n,n,len(self.orientation.names)):raise ValueError("Symmetric feature mismatch")
        present=self.skeleton.predict(symmetric_pair_features(x))!=0;raw=np.zeros((n,n),bool);scores=np.zeros((n,n),float)
        for keep,(i,j) in zip(present,combinations(range(n),2)):
            if not keep:continue
            votes=self.orientation.predict(np.stack([x[i,j],x[j,i]]));left=int(votes[0]);right=int(SWAP_LABELS[int(votes[1])]);cls=left if left==right and left in (1,2,3) else 3
            if cls==1:raw[i,j]=True;scores[i,j]=1.
            elif cls==2:raw[j,i]=True;scores[j,i]=1.
            else:raw[i,j]=raw[j,i]=True
        directed=raw&~raw.T;dag,_=acyclic_projection(np.where(directed,scores,0.),threshold=1e-12)
        return dag|(raw&raw.T)
    def to_dict(self):return {"skeleton":self.skeleton.to_dict(),"orientation":self.orientation.to_dict(),"complexity":self.complexity,"semantics":"symmetric decoded skeleton then swap-consistent orientation then acyclic projection"}
    @classmethod
    def from_dict(cls,d):
        p=cls(Rule.from_dict(d["skeleton"]),Rule.from_dict(d["orientation"]))
        if "complexity" in d and d["complexity"]!=p.complexity:raise ValueError("Symmetric complexity mismatch")
        return p
