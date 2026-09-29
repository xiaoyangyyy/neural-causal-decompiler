"""Factorized skeleton/orientation head for node-context graph teachers."""
import torch
from torch import nn
from .graph_model import SWAP_LABELS
from .node_context_graph import NodeContextGraphDiscoverer


class FactorizedNodeContextGraphDiscoverer(NodeContextGraphDiscoverer):
    def __init__(self,width=48):
        super().__init__(width)
        first=self.head[0];activation=self.head[1]
        self.head_trunk=nn.Sequential(first,activation)
        self.head=nn.Identity()
        self.skeleton_head=nn.Linear(width,1)
        self.orientation_head=nn.Linear(width,3)

    def raw_components(self,h):
        z=self.head_trunk(h)
        return self.skeleton_head(z).squeeze(-1),self.orientation_head(z)

    def from_hidden(self,h):
        skeleton,orientation=self.raw_components(h)
        conditional=torch.log_softmax(orientation,-1)
        centered=conditional-conditional.max(-1,keepdim=True).values
        scores=torch.cat((torch.zeros_like(skeleton)[...,None],skeleton[...,None]+centered),-1)
        return .5*(scores+scores.transpose(1,2)[...,SWAP_LABELS])

    def forward(self,x):return self.from_hidden(self.representation(x))
