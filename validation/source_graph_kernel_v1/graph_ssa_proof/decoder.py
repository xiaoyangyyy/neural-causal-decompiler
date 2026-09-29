"""Explicit serialized graph tail; no neural framework or source checkout call."""
import ast,hashlib
APPROVED_AST_HASHES={'decode_graph': '2acb1d8c3e777bcf871b8694738e9ae931a06678a7965640a0e36b2cdfc4b2cb', 'dag_completion': '7a6d16fe2aa47e364b345db3f092751eeae95e60d13a0862e7f0e50b5bab640a', 'topological_order': '34c4602e6002a97e80a6f5b378be270d4bb94005031e28c9fc174f8a86eb8fb1', 'directed_path': '282163a6850bd583f052159d594ef8fffa7f4ea2e5791b29a106afba32c8344c', 'acyclic_projection': '76807003bc93d480c9a57962837e947688387325528d5f7be449f7f734a7aeed'}
PRIMARY_FUNCTION_TEXT={'decode_graph': "def decode_graph(probabilities):\n    p = np.asarray(probabilities)\n    n = len(p)\n    labels = p.argmax(-1)\n    raw = np.zeros((n, n), bool)\n    for i, j in combinations(range(n), 2):\n        cls = int(labels[i, j])\n        if cls == 1:\n            raw[i, j] = True\n        elif cls == 2:\n            raw[j, i] = True\n        elif cls == 3:\n            raw[i, j] = raw[j, i] = True\n    directed = raw & ~raw.T\n    scores = np.where(directed, p[:, :, 1], 0)\n    dag, rejected = acyclic_projection(scores, threshold=1e-12)\n    partial = dag | raw & raw.T\n    return (partial, {'removed_cycle_edges': rejected, 'raw_graph': raw.astype(int).tolist(), 'semantics': 'partially_directed_prediction_not_guaranteed_completed_PDAG'})", 'dag_completion': 'def dag_completion(partial):\n    """One explicit acyclic completion, not an identification of ambiguous edges."""\n    p = np.asarray(partial, dtype=bool)\n    directed = p & ~p.T\n    order = topological_order(directed)\n    rank = {v: i for i, v in enumerate(order)}\n    completed = directed.copy()\n    chosen = []\n    for i, j in combinations(range(len(p)), 2):\n        if p[i, j] and p[j, i]:\n            u, v = (i, j) if rank[i] < rank[j] else (j, i)\n            completed[u, v] = True\n            chosen.append([u, v])\n    topological_order(completed)\n    return (completed, chosen)', 'topological_order': "def topological_order(graph):\n    a = np.asarray(graph, dtype=bool)\n    if a.ndim != 2 or a.shape[0] != a.shape[1] or np.diag(a).any():\n        raise ValueError('Expected loop-free square directed graph')\n    degree = a.sum(0).astype(int)\n    todo = sorted(np.flatnonzero(degree == 0).tolist())\n    result = []\n    while todo:\n        v = todo.pop(0)\n        result.append(v)\n        for child in np.flatnonzero(a[v]):\n            degree[child] -= 1\n            if degree[child] == 0:\n                todo.append(int(child))\n                todo.sort()\n    if len(result) != len(a):\n        raise ValueError('Directed cycle')\n    return result", 'directed_path': 'def directed_path(pdag, x, y):\n    a = np.asarray(pdag, dtype=bool)\n    directed = a & ~a.T\n    seen = {x}\n    stack = [x]\n    while stack:\n        u = stack.pop()\n        for v in np.flatnonzero(directed[u]):\n            if v == y:\n                return True\n            if int(v) not in seen:\n                seen.add(int(v))\n                stack.append(int(v))\n    return False', 'acyclic_projection': 'def acyclic_projection(edge_probabilities, threshold=0.5):\n    """Explicit greedy projection; never report it as the raw neural output."""\n    prob = np.asarray(edge_probabilities, dtype=float)\n    if prob.ndim != 2 or prob.shape[0] != prob.shape[1] or (not np.isfinite(prob).all()):\n        raise ValueError(\'Invalid edge probabilities\')\n    n = len(prob)\n    result = np.zeros((n, n), bool)\n    rejected = []\n    edges = sorted(((float(prob[i, j]), i, j) for i in range(n) for j in range(n) if i != j and prob[i, j] >= threshold), reverse=True)\n    for score, i, j in edges:\n        if result[j, i] or directed_path(result, j, i):\n            rejected.append([i, j, score])\n        else:\n            result[i, j] = True\n    topological_order(result)\n    return (result, rejected)'}

def validate_decoder(spec):
 if not isinstance(spec,dict) or set(spec)!=set(APPROVED_AST_HASHES):raise ValueError('Missing graph decision/completion function')
 for name,text in spec.items():
  if not isinstance(text,str) or len(text)>20000:raise ValueError('Graph function serialization')
  module=ast.parse(text)
  if len(module.body)!=1 or not isinstance(module.body[0],ast.FunctionDef) or module.body[0].name!=name:raise ValueError('Unexpected executable graph-tail statement')
  if hashlib.sha256(ast.dump(module.body[0],include_attributes=False).encode()).hexdigest()!=APPROVED_AST_HASHES[name]:raise ValueError('Frozen graph decision algorithm changed')
 return True


def graph_tail(probabilities,spec=None):
 from itertools import combinations
 import numpy as np
 spec=PRIMARY_FUNCTION_TEXT if spec is None else spec
 validate_decoder(spec)
 # Exactly the five audited function bodies, with their complete dependency
 # closure. No imports, external files, model objects or hidden NN suffix.
 namespace={'np':np,'combinations':combinations}
 module=ast.Module(body=[ast.parse(text).body[0] for text in spec.values()],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),'<explicit-graph-tail>','exec'),namespace)
 partial,decode=namespace['decode_graph'](probabilities)
 inferred,choices=namespace['dag_completion'](partial)
 return {'partial_graph':partial,'inferred_dag':inferred,'completion_choices':choices,**decode,
         'completion_is_not_causal_identification':True}
