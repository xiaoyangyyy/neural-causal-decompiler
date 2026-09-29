"""Active intervention response features for multivariate graph discovery."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .graph_model import GRAPH_FEATURES,pair_features
from .factorized_node_context_graph import FactorizedNodeContextGraphDiscoverer

ACTIVE_FEATURES=("do_contrast_mean","do_contrast_rms","do_plus_change","do_minus_change")
ACTIVE_FEATURE_COUNT=len(GRAPH_FEATURES)+len(ACTIVE_FEATURES)


def intervention_response_features(world,observational,base_features=None):
    """Append ordered intervention-response features without reading graph truth."""
    data=np.asarray(observational,dtype=float)
    if data.ndim!=2 or data.shape!=(world.samples,world.nodes) or not np.isfinite(data).all():
        raise ValueError("Invalid observational sample for active features")
    base=pair_features(data) if base_features is None else np.asarray(base_features,dtype=float)
    if base.shape!=(world.nodes,world.nodes,len(GRAPH_FEATURES)):raise ValueError("Invalid base graph features")
    n=world.nodes
    response=np.zeros((n,n,len(ACTIVE_FEATURES)),dtype=float)
    means=data.mean(0);scales=np.maximum(data.std(0),1e-6)
    for source in range(n):
        plus=world.sample(interventions={source:float(means[source]+scales[source])})
        minus=world.sample(interventions={source:float(means[source]-scales[source])})
        for target in range(n):
            if source==target:continue
            denom=scales[target]
            contrast=(plus[:,target]-minus[:,target])/denom
            response[source,target]=[
                float(contrast.mean()),float(np.sqrt(np.mean(contrast**2))),
                float(np.mean(np.abs(plus[:,target]-data[:,target]))/denom),
                float(np.mean(np.abs(minus[:,target]-data[:,target]))/denom)]
    result=np.concatenate((base,response),-1)
    if not np.isfinite(result).all():raise ValueError("Non-finite active graph feature")
    return result


def padded_observational_features(observational,base_features=None):
    base=pair_features(observational) if base_features is None else np.asarray(base_features,dtype=float)
    return np.concatenate((base,np.zeros((*base.shape[:-1],len(ACTIVE_FEATURES)))),axis=-1)


class ActiveFactorizedGraphDiscoverer(FactorizedNodeContextGraphDiscoverer):
    """Factorized node-context model with equal-width active feature input."""
    def __init__(self,width=48):
        super().__init__(width)
        self.mean=torch.zeros(ACTIVE_FEATURE_COUNT)
        self.std=torch.ones(ACTIVE_FEATURE_COUNT)
        self.encoder=nn.Sequential(nn.Linear(ACTIVE_FEATURE_COUNT,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())


def load_active_factorized_graph(path):
    state=torch.load(Path(path),map_location="cpu",weights_only=True)
    if state.get("architecture") not in ("active_input_observational_control_v1","active_input_intervention_v1"):
        raise ValueError("Unknown active graph architecture")
    model=ActiveFactorizedGraphDiscoverer(state["width"])
    model.load_state_dict(state["state_dict"]);model.eval();return model
