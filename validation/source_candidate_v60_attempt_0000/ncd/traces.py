"""Reusable bounded PyTorch circuit capture and invocation-specific patching."""
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class CircuitTrace:
    activations:dict
    output:torch.Tensor

    def save(self,path):
        arrays={f"{name.replace('.','__')}__call_{i}":value.cpu().numpy()
                for name,values in self.activations.items() for i,value in enumerate(values)}
        np.savez_compressed(path,**arrays)
        return {key:{"shape":list(value.shape),"elements":int(value.size)} for key,value in arrays.items()}

class TorchCircuit:
    def __init__(self,model,layers,max_elements=2_000_000):
        self.model=model;self.layers=tuple(layers);self.max_elements=max_elements
        available=dict(model.named_modules())
        if any(name not in available for name in layers):raise ValueError("Unknown circuit layer")
        self.modules={name:available[name] for name in layers}

    @contextmanager
    def _hooks(self,callback):
        handles=[];states={m:m.training for m in self.model.modules()}
        self.model.eval()
        try:
            for name,module in self.modules.items():
                handles.append(module.register_forward_hook(lambda m,i,o,name=name:callback(name,o)))
            yield
        finally:
            for handle in handles:handle.remove()
            for module,state in states.items():module.training=state

    def capture(self,data):
        activations={name:[] for name in self.layers};total=0
        def hook(name,output):
            nonlocal total
            if not isinstance(output,torch.Tensor):raise ValueError("Circuit layer must return a tensor")
            total+=output.numel()
            if total>self.max_elements:raise ValueError("Circuit trace exceeds budget")
            activations[name].append(output.detach().clone())
        with self._hooks(hook),torch.no_grad():output=self.model(data).detach().clone()
        return CircuitTrace(activations,output)

    def intervene(self,data,source,layer,invocation=0,subspace=None):
        if layer not in source.activations or invocation>=len(source.activations[layer]):raise ValueError("Missing source invocation")
        counts={name:0 for name in self.layers};applied=False
        def hook(name,output):
            nonlocal applied
            current=counts[name];counts[name]+=1
            if name!=layer or current!=invocation:return None
            reference=source.activations[layer][invocation].to(output)
            if reference.shape!=output.shape:raise ValueError("Activation shape mismatch")
            applied=True
            if subspace is None:return reference.clone()
            q=torch.as_tensor(subspace,dtype=output.dtype,device=output.device)
            if q.ndim!=2 or q.shape[0]!=output.shape[-1]:raise ValueError("Subspace shape mismatch")
            return output+(reference-output)@q@q.T
        with self._hooks(hook),torch.no_grad():result=self.model(data).detach().clone()
        if not applied:raise ValueError("Requested invocation did not execute")
        return result
