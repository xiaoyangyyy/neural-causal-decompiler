"""Executable graph-global sparse ranking program."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .rules import Rule
from .graph_model import SWAP_LABELS
from .graphs import acyclic_projection
from .symmetric_skeleton_program import symmetric_pair_features

COUNT_FEATURES=("nodes","score_mean","score_std","score_min","score_max","score_q25","score_q50","score_q75")

def count_features(scores,nodes):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError("Invalid edge scores")
    return np.array([nodes,s.mean(),s.std(),s.min(),s.max(),*np.quantile(s,[.25,.5,.75])],float)

@dataclass(frozen=True)
class GraphGlobalRankingProgram:
    edge_names:tuple
    edge_mean:tuple
    edge_scale:tuple
    edge_coefficients:tuple
    edge_intercept:float
    count_mean:tuple
    count_scale:tuple
    count_coefficients:tuple
    count_intercept:float
    orientation:Rule
    edge_c:float
    count_alpha:float
    def __post_init__(self):
        size=len(self.edge_names);vectors=(self.edge_mean,self.edge_scale,self.edge_coefficients)
        if size!=2*len(self.orientation.names) or any(len(x)!=size for x in vectors):raise ValueError("Invalid edge schema")
        if any(len(x)!=len(COUNT_FEATURES) for x in (self.count_mean,self.count_scale,self.count_coefficients)):raise ValueError("Invalid count schema")
        values=np.r_[self.edge_mean,self.edge_scale,self.edge_coefficients,self.edge_intercept,self.count_mean,self.count_scale,self.count_coefficients,self.count_intercept,self.edge_c,self.count_alpha]
        if not np.isfinite(values).all() or np.any(np.asarray(self.edge_scale)<=0) or np.any(np.asarray(self.count_scale)<=0) or self.edge_c<=0 or self.count_alpha<=0:raise ValueError("Invalid ranking parameters")
    @property
    def edge_nonzero(self):return int(np.count_nonzero(self.edge_coefficients))
    @property
    def count_nonzero(self):return int(np.count_nonzero(self.count_coefficients))
    @property
    def complexity(self):return self.orientation.complexity+self.edge_nonzero+self.count_nonzero+3
    def edge_scores(self,features):
        rows=symmetric_pair_features(features);return ((rows-np.asarray(self.edge_mean))/np.asarray(self.edge_scale))@np.asarray(self.edge_coefficients)+self.edge_intercept
    def predicted_count(self,scores,nodes):
        z=(count_features(scores,nodes)-np.asarray(self.count_mean))/np.asarray(self.count_scale)
        value=float(z@np.asarray(self.count_coefficients)+self.count_intercept)
        return int(np.clip(np.rint(value),0,len(scores)))
    def predict(self,features):
        x=np.asarray(features,float);n=x.shape[0]
        if x.shape!=(n,n,len(self.orientation.names)):raise ValueError("Ranking feature mismatch")
        pairs=list(combinations(range(n),2));scores=self.edge_scores(x);k=self.predicted_count(scores,n)
        order=np.lexsort((np.arange(len(scores)), -scores));selected=np.zeros(len(scores),bool);selected[order[:k]]=True
        raw=np.zeros((n,n),bool);direction_scores=np.zeros((n,n),float)
        for keep,(i,j),score in zip(selected,pairs,scores):
            if not keep:continue
            votes=self.orientation.predict(np.stack([x[i,j],x[j,i]]));left=int(votes[0]);right=int(SWAP_LABELS[int(votes[1])]);cls=left if left==right and left in (1,2,3) else 3
            magnitude=float(score-scores.min()+1.)
            if cls==1:raw[i,j]=True;direction_scores[i,j]=magnitude
            elif cls==2:raw[j,i]=True;direction_scores[j,i]=magnitude
            else:raw[i,j]=raw[j,i]=True
        directed=raw&~raw.T;dag,_=acyclic_projection(np.where(directed,direction_scores,0.),threshold=1e-12)
        return dag|(raw&raw.T)
    def to_dict(self):
        return {"edge_names":list(self.edge_names),"edge_mean":list(self.edge_mean),"edge_scale":list(self.edge_scale),"edge_coefficients":list(self.edge_coefficients),"edge_intercept":self.edge_intercept,"count_names":list(COUNT_FEATURES),"count_mean":list(self.count_mean),"count_scale":list(self.count_scale),"count_coefficients":list(self.count_coefficients),"count_intercept":self.count_intercept,"orientation":self.orientation.to_dict(),"edge_c":self.edge_c,"count_alpha":self.count_alpha,"edge_nonzero":self.edge_nonzero,"count_nonzero":self.count_nonzero,"complexity":self.complexity,"semantics":"sparse symmetric edge scores -> affine edge count -> deterministic top-k -> orientation -> acyclic projection"}
    @classmethod
    def from_dict(cls,d):
        if tuple(d["count_names"])!=COUNT_FEATURES:raise ValueError("Count feature mismatch")
        p=cls(tuple(d["edge_names"]),tuple(d["edge_mean"]),tuple(d["edge_scale"]),tuple(d["edge_coefficients"]),float(d["edge_intercept"]),tuple(d["count_mean"]),tuple(d["count_scale"]),tuple(d["count_coefficients"]),float(d["count_intercept"]),Rule.from_dict(d["orientation"]),float(d["edge_c"]),float(d["count_alpha"]))
        for key in ("edge_nonzero","count_nonzero","complexity"):
            if key in d and d[key]!=getattr(p,key):raise ValueError("Ranking metadata mismatch")
        return p
