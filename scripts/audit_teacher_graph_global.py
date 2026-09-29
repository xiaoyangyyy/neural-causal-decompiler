from pathlib import Path
import json,numpy as np
from ncd.relational_graph import load_relational_graph
from ncd.graph_model import graph_probabilities,decode_graph
for source_name in ('relational_seed493','relational_seed494'):
 source=Path('runs')/source_name;print(source_name)
 for mode in ('without_relations','with_relations'):
  model=load_relational_graph(source/'models'/mode/'graph_teacher.pt');print(' ',mode)
  for n in (3,5,8):
   for split in ('extraction','refinement'):
    with np.load(source/'datasets'/f'n{n}_{split}'/'features.npz') as z:f=z['features']
    gs=[decode_graph(p)[0] for p in graph_probabilities(model,f)];counts=np.array([np.sum(g|g.T)//2 for g in gs]);degrees=np.concatenate([(g|g.T).sum(0) for g in gs]);print(f'   n{n} {split}: edges {counts.min()}..{counts.max()} mean={counts.mean():.2f} modes={dict(zip(*np.unique(counts,return_counts=True)))} degree_mean={degrees.mean():.2f}')
