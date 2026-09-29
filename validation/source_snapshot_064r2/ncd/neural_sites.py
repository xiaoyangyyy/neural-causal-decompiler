"""Exact vector-valued cuts in the existing Discoverer, without teacher changes."""
import numpy as np
import torch
from torch import nn
from .model import Discoverer,SWAP
from .joint_alignment import frozen_teacher

SITES=("representation","head_linear","head_tanh")

class SiteDecoder(nn.Module):
    def __init__(self,teacher,site):
        super().__init__()
        if not isinstance(teacher,Discoverer) or site not in SITES:raise ValueError("Unsupported neural cut")
        self.teacher=teacher;self.site=site

    def from_hidden(self,base,swapped):
        if self.site=="representation":return self.teacher.from_hidden(base,swapped)
        suffix=self.teacher.head[1:] if self.site=="head_linear" else self.teacher.head[2:]
        return .5*(suffix(base)+suffix(swapped)[:,SWAP])

    def extract(self,data,batch_size=128):
        outputs=[]
        with frozen_teacher(self),torch.no_grad():
            for offset in range(0,len(data),batch_size):
                x=torch.as_tensor(data[offset:offset+batch_size],dtype=torch.float32)
                a,b=self.teacher.hidden_pair(x)
                if self.site!="representation":
                    prefix=self.teacher.head[:1] if self.site=="head_linear" else self.teacher.head[:2]
                    a,b=prefix(a),prefix(b)
                outputs.append(torch.cat([a,b],1).numpy())
        if not outputs:raise ValueError("Empty extraction data")
        return np.concatenate(outputs).astype(float)

    def logits(self,h):
        h=torch.as_tensor(h,dtype=torch.float32);width=h.shape[1]//2
        if h.ndim!=2 or h.shape[1]%2:raise ValueError("Invalid hidden cut")
        return self.from_hidden(h[:,:width],h[:,width:])
