"""DAG / partially directed graph semantics and PC-stable skeleton discovery.

Convention: graph[i,j]=True denotes i->j. A CPDAG has both entries for an
undirected edge. True DAGs are used only for generator labels/evaluation.
"""
from itertools import combinations
import numpy as np
from scipy.stats import norm

def topological_order(graph):
    a=np.asarray(graph,dtype=bool)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or np.diag(a).any():
        raise ValueError("Expected loop-free square directed graph")
    degree=a.sum(0).astype(int)
    todo=sorted(np.flatnonzero(degree==0).tolist()); result=[]
    while todo:
        v=todo.pop(0);result.append(v)
        for child in np.flatnonzero(a[v]):
            degree[child]-=1
            if degree[child]==0: todo.append(int(child));todo.sort()
    if len(result)!=len(a): raise ValueError("Directed cycle")
    return result

def random_dag(nodes,rng,expected_degree=1.5):
    if not 2<=nodes<=8 or not 0<=expected_degree<=nodes-1: raise ValueError("Invalid DAG size/degree")
    order=rng.permutation(nodes)
    p=expected_degree/(nodes-1)
    a=np.zeros((nodes,nodes),dtype=bool)
    for i,j in combinations(range(nodes),2):
        if rng.random()<p: a[order[i],order[j]]=True
    return a

def d_separated(graph,x,y,condition=()):
    """Exact ancestral moral-graph criterion; used as a test/label oracle."""
    a=np.asarray(graph,dtype=bool);topological_order(a)
    z=set(condition)
    if x==y or x in z or y in z: raise ValueError("Invalid separation query")
    ancestors={x,y}|z
    while True:
        more=set(np.flatnonzero(a[:,list(ancestors)].any(1)).tolist())
        if more<=ancestors:break
        ancestors|=more
    moral=(a|a.T).copy()
    for child in ancestors:
        parents=list(set(np.flatnonzero(a[:,child]))&ancestors)
        for u,v in combinations(parents,2):moral[u,v]=moral[v,u]=True
    reachable={x};stack=[x]
    while stack:
        u=stack.pop()
        for v in set(np.flatnonzero(moral[u])) & (ancestors-z):
            if v not in reachable:reachable.add(v);stack.append(v)
    return y not in reachable

def directed_path(pdag,x,y):
    a=np.asarray(pdag,dtype=bool)
    directed=a&~a.T;seen={x};stack=[x]
    while stack:
        u=stack.pop()
        for v in np.flatnonzero(directed[u]):
            if v==y:return True
            if int(v) not in seen:seen.add(int(v));stack.append(int(v))
    return False

def orient_colliders(skeleton,sepsets):
    """Orient unshielded triples only when middle is absent from a known sepset."""
    p=np.asarray(skeleton,dtype=bool).copy()
    proposals=set()
    for z in range(len(p)):
        for x,y in combinations(np.flatnonzero(p[z]),2):
            key=(min(int(x),int(y)),max(int(x),int(y)))
            if not p[x,y] and key in sepsets and z not in sepsets[key]:
                proposals.update(((int(x),z),(int(y),z)))
    for u,v in sorted(proposals):
        # Conflicting finite-sample orientation requests remain unresolved.
        if (v,u) not in proposals:p[v,u]=False
    return p

def meek_closure(pdag):
    """Sound R1/R2/R3 orientation closure. Conflicts/cycles remain unresolved."""
    p=np.asarray(pdag,dtype=bool).copy()
    def adj(a,b):return p[a,b] or p[b,a]
    def arrow(a,b):return p[a,b] and not p[b,a]
    def line(a,b):return p[a,b] and p[b,a]
    while True:
        proposals=set()
        for a,b in zip(*np.where(p&p.T)):
            # R1 c -> a -- b, c nonadjacent b
            r1=any(arrow(c,a) and not adj(c,b) for c in range(len(p)) if c not in (a,b))
            # R2 directed path a -> ... -> b
            r2=directed_path(p,a,b)
            candidates=[c for c in range(len(p)) if c not in (a,b) and line(a,c) and arrow(c,b)]
            r3=any(not adj(c,d) for c,d in combinations(candidates,2))
            if r1 or r2 or r3:proposals.add((int(a),int(b)))
        applied=False
        for a,b in sorted(proposals):
            if (b,a) not in proposals and line(a,b) and not directed_path(p,b,a):
                p[b,a]=False;applied=True
        if not applied:break
    return p

def cpdag(graph):
    """Exact equivalence-class graph for <=8-node DAGs via oracle sepsets."""
    a=np.asarray(graph,dtype=bool);topological_order(a)
    skeleton=a|a.T;sepsets={}
    for x,y in combinations(range(len(a)),2):
        if skeleton[x,y]:continue
        candidates=[i for i in range(len(a)) if i not in (x,y)]
        found=False
        for size in range(len(candidates)+1):
            for z in combinations(candidates,size):
                if d_separated(a,x,y,z):sepsets[(x,y)]=z;found=True;break
            if found:break
    return meek_closure(orient_colliders(skeleton,sepsets))

def partial_correlation(data,x,y,condition=()):
    data=np.asarray(data,dtype=float)
    if condition:
        z=np.column_stack([np.ones(len(data)),data[:,list(condition)]])
        pair=data[:,[x,y]]-z@np.linalg.lstsq(z,data[:,[x,y]],rcond=None)[0]
    else:pair=data[:,[x,y]]-data[:,[x,y]].mean(0)
    scale=np.linalg.norm(pair[:,0])*np.linalg.norm(pair[:,1])
    return float(np.clip(pair[:,0]@pair[:,1]/max(scale,1e-15),-.999999,.999999))

def fisher_p(data,x,y,condition=()):
    if len(data)<=len(condition)+3:raise ValueError("Insufficient rows for Fisher test")
    r=partial_correlation(data,x,y,condition)
    z=np.arctanh(r)*np.sqrt(len(data)-len(condition)-3)
    return float(2*norm.sf(abs(z)))

def pc_stable(data,alpha=.01,max_condition=2):
    n=data.shape[1];adj=np.ones((n,n),bool);np.fill_diagonal(adj,False);sepsets={};tests=[]
    for size in range(min(max_condition,n-2)+1):
        snapshot=adj.copy();removals=[]
        for x,y in combinations(range(n),2):
            if not snapshot[x,y]:continue
            sets=set()
            for side,other in ((x,y),(y,x)):
                neighbors=[v for v in np.flatnonzero(snapshot[side]) if v!=other]
                sets.update(combinations(neighbors,size))
            for z in sorted(sets):
                p=fisher_p(data,x,y,z);tests.append({"x":x,"y":y,"condition":list(map(int,z)),"p":p})
                if p>alpha:removals.append((x,y));sepsets[(x,y)]=tuple(map(int,z));break
        for x,y in removals:adj[x,y]=adj[y,x]=False
    skeleton=adj.copy();collider_pdag=orient_colliders(skeleton,sepsets);pdag=meek_closure(collider_pdag)
    return {"pdag":pdag,"skeleton":skeleton,"collider_pdag":collider_pdag,
            "sepsets":{f"{x},{y}":list(z) for (x,y),z in sepsets.items()},"tests":tests,
            "assumption":"Gaussian partial-correlation CI; nonlinear validity not guaranteed"}

def acyclic_projection(edge_probabilities,threshold=.5):
    """Explicit greedy projection; never report it as the raw neural output."""
    prob=np.asarray(edge_probabilities,dtype=float)
    if prob.ndim!=2 or prob.shape[0]!=prob.shape[1] or not np.isfinite(prob).all():
        raise ValueError("Invalid edge probabilities")
    n=len(prob);result=np.zeros((n,n),bool);rejected=[]
    edges=sorted(((float(prob[i,j]),i,j) for i in range(n) for j in range(n) if i!=j and prob[i,j]>=threshold),reverse=True)
    for score,i,j in edges:
        if result[j,i] or directed_path(result,j,i):rejected.append([i,j,score])
        else:result[i,j]=True
    topological_order(result)
    return result,rejected
