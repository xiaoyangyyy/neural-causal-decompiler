"""Swap-structured orientation head for factorized node-context teachers."""
import torch
from torch import nn
from .factorized_node_context_graph import FactorizedNodeContextGraphDiscoverer


class AntisymmetricDirectionGraphDiscoverer(FactorizedNodeContextGraphDiscoverer):
    """Factorized graph head with an exact orientation swap constraint.

    The skeleton path is inherited unchanged.  Direction is an odd function of
    the ordered-pair representation difference, while the undirected score is a
    function of the symmetric representation.  Consequently exchanging the two
    nodes swaps the directed logits and leaves the undirected logit unchanged.
    """
    def __init__(self,width=48):
        super().__init__(width)
        del self.orientation_head
        self.direction_head=nn.Linear(width,1,bias=False)
        self.undirected_head=nn.Linear(width,1)

    def raw_components(self,h):
        z=self.head_trunk(h)
        skeleton=self.skeleton_head(z).squeeze(-1)
        reverse=z.transpose(1,2)
        antisymmetric=.5*(z-reverse)
        symmetric=.5*(z+reverse)
        direction=self.direction_head(antisymmetric).squeeze(-1)
        undirected=self.undirected_head(symmetric).squeeze(-1)
        orientation=torch.stack((direction,-direction,undirected),-1)
        return skeleton,orientation
