"""Exact bounded convex polytopes by rational edge clipping.

Vertices and inward halfspaces represent the same bounded convex set.
A crossing pair is an edge iff its common tight normals have rank d-1.
This preserves lower-dimensional closed contacts without floating LP status.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product


def evaluate(form,point):
    return form[-1]+sum((a*b for a,b in zip(form[:-1],point)),Q(0))


def rank(rows,dimension):
    matrix=[list(row) for row in rows]
    result=0
    for column in range(dimension):
        pivot=next((j for j in range(result,len(matrix)) if matrix[j][column]),None)
        if pivot is None:
            continue
        matrix[result],matrix[pivot]=matrix[pivot],matrix[result]
        divisor=matrix[result][column]
        matrix[result]=[x/divisor for x in matrix[result]]
        for j in range(result+1,len(matrix)):
            if matrix[j][column]:
                factor=matrix[j][column]
                matrix[j]=[a-factor*b for a,b in zip(matrix[j],matrix[result])]
        result+=1
        if result==dimension:
            break
    return result


@dataclass(frozen=True)
class Polytope:
    vertices: tuple
    constraints: tuple

    @property
    def dimension(self):
        return len(self.constraints[0])-1


def box(low,high):
    low,high=tuple(map(Q,low)),tuple(map(Q,high))
    if not low or len(low)!=len(high) or any(a>=b for a,b in zip(low,high)):
        raise ValueError('Nondegenerate rational box required')
    dimension=len(low)
    forms=[]
    for i,(a,b) in enumerate(zip(low,high)):
        unit=tuple(Q(int(j==i)) for j in range(dimension))
        forms.extend((unit+(-a,),tuple(-x for x in unit)+(b,)))
    return Polytope(tuple(sorted(product(*zip(low,high)))),tuple(forms))


def clip(poly,form,rank_cache=None):
    """Intersection with form>=0; empty intersections have no vertices."""
    if not poly.vertices:
        return poly
    form=tuple(map(Q,form))
    d=poly.dimension
    if len(form)!=d+1:
        raise ValueError('Halfspace dimension mismatch')
    scores=tuple(evaluate(form,p) for p in poly.vertices)
    if min(scores)>=0:
        return poly
    constraints=poly.constraints+(form,)
    if max(scores)<0:
        return Polytope((),constraints)
    positive=[j for j,v in enumerate(scores) if v>0]
    negative=[j for j,v in enumerate(scores) if v<0]
    result={p for p,s in zip(poly.vertices,scores) if s>=0}
    if positive and negative:
        active={j:frozenset(k for k,f in enumerate(poly.constraints) if evaluate(f,poly.vertices[j])==0)
                for j in positive+negative}
        cache={} if rank_cache is None else rank_cache
        for i in positive:
            for j in negative:
                common=active[i]&active[j]
                if len(common)<d-1:
                    continue
                normals=tuple(sorted({poly.constraints[k][:-1] for k in common}))
                if normals not in cache:
                    cache[normals]=rank(normals,d)
                if cache[normals]!=d-1:
                    continue
                scale=scores[i]/(scores[i]-scores[j])
                result.add(tuple(a+scale*(b-a) for a,b in zip(poly.vertices[i],poly.vertices[j])))
    if not result:
        raise ValueError('Nonempty halfspace intersection lost all vertices')
    output=Polytope(tuple(sorted(result)),constraints)
    if any(evaluate(f,p)<0 for f in output.constraints for p in output.vertices):
        raise ValueError('Clipped vertex violates an inward halfspace')
    return output
