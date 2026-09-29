"""Exact CDIR polynomial semantics using actual binary64 constant values.

Canonical coefficient form proves function equality for this algebraic subset.
Protected division, nonlinear statistics and internal causal equivalence are not
covered. Decimal pretty-printing is not a sound conversion of a float constant.
"""
from fractions import Fraction as Q
import json
from .cdir import Node
from .io import digest


def canonical_polynomial(expression,variables):
    if type(variables)is not int or not 1<=variables<=8:raise ValueError('Polynomial input dimension')
    node=Node.from_dict(expression);zero=(0,)*variables
    def add(a,b,sign=1):
        result=dict(a)
        for key,value in b.items():result[key]=result.get(key,Q(0))+sign*value
        return {key:v for key,v in result.items() if v}
    def multiply(a,b):
        result={}
        for ka,va in a.items():
            for kb,vb in b.items():
                key=tuple(x+y for x,y in zip(ka,kb));result[key]=result.get(key,Q(0))+va*vb
        return {key:v for key,v in result.items() if v}
    def walk(n):
        if n.op=='constant':return {} if n.value==0 else {zero:Q(n.value)}
        if n.op=='var':
            if not 0<=n.index<variables:raise ValueError('Polynomial variable outside domain')
            key=list(zero);key[n.index]=1;return {tuple(key):Q(1)}
        if n.op not in ('add','sub','mul','square'):raise ValueError('Operator outside exact polynomial subset')
        a=[walk(v) for v in n.args]
        if n.op in ('add','sub'):return add(a[0],a[1],1 if n.op=='add' else -1)
        return multiply(a[0],a[0] if n.op=='square' else a[1])
    return [{'powers':list(k),'coefficient':str(v)} for k,v in sorted(walk(node).items())]


def prove_polynomial_equivalence(left,right,variables):
    a,b=canonical_polynomial(left,variables),canonical_polynomial(right,variables)
    return {'schema':'ncd.exact-binary-polynomial-equivalence.v1','status':'proved' if a==b else 'refuted',
        'left':left,'right':right,'variables':variables,'left_normal_form':a,'right_normal_form':b,
        'scope':'equality as mathematical functions on all real inputs with the exact binary64 values of the serialized CDIR constants',
        'canonical_serialization':'sorted powers and reduced rational coefficients; one representation per polynomial function',
        'device_rounding_covered':False,'internal_causal_equivalence_covered':False}


def verify_polynomial_equivalence(certificate):
    if certificate!=prove_polynomial_equivalence(certificate['left'],certificate['right'],certificate['variables']):raise ValueError('Polynomial equivalence certificate mismatch')
    return {'status':'verified','conclusion':certificate['status'],'scope':certificate['scope']}


