"""Generate a replayable exact interval-chain exclusion certificate."""
from __future__ import annotations
import argparse
import gzip
import json
import time
from itertools import combinations
from pathlib import Path
from ncd.continuous_separation import _digest
from ncd.exact_interval_lower import (
    _benchmark, base_constraints, chain_constraints,
    verify_interval_chain_exclusion,
)
import numpy as np
from scipy.optimize import linprog
from sympy import Matrix, Rational


def _rational(value):
    return Rational(value.numerator, value.denominator)


def farkas(rows, states):
    """Use floating LP only to propose a dual support; validate it exactly."""
    matrix = np.array([list(row) for row, _ in rows], dtype=float)
    vector = np.array([float(bound) for _, bound in rows])
    primal = linprog(np.zeros(2*states), A_ub=matrix, b_ub=vector,
                     bounds=(None,None), method='highs')
    if primal.status != 2:
        return None
    n = len(rows)
    dual = linprog(vector, A_eq=np.vstack((matrix.T, np.ones(n))),
                   b_eq=np.r_[np.zeros(2*states), 1.],
                   bounds=(0,None), method='highs')
    if dual.status != 0 or dual.fun >= -1e-9:
        return None
    support = np.flatnonzero(dual.x > 1e-8)
    exact = Matrix([[_rational(rows[k][0][j]) for k in support]
                    for j in range(2*states)] + [[1 for _ in support]])
    rhs = Matrix([0]*(2*states)+[1])
    try:
        solution, params = exact.gauss_jordan_solve(rhs)
    except ValueError:
        return None
    if params:
        solution = solution.subs({v:0 for v in params})
    weights = tuple(Fraction(int(v.p),int(v.q)) for v in solution)
    if (any(w < 0 for w in weights)
            or sum(weights,Fraction(0)) != 1
            or any(sum((rows[k][0][j]*w for k,w in zip(support,weights)),Fraction(0)) != 0
                   for j in range(2*states))
            or sum((rows[k][1]*w for k,w in zip(support,weights)),Fraction(0)) >= 0):
        return None
    return tuple(int(k) for k in support), weights
from fractions import Fraction


def generate(states: int, progress: int = 1000):
    system, lam, beta = _benchmark()
    m = states
    base = base_constraints(m, Fraction(0.101))
    chains = [c for length in range(1,m+1) for c in combinations(range(m),length)]
    choices = {(i,c):chain_constraints(m,i,c,lam,beta)
               for i in range(m) for c in chains}
    order = [k//2 if k%2==0 else m-1-k//2 for k in range(m)]
    tokens=[]
    start=time.perf_counter()
    def walk(depth,rows):
        if depth==m:
            raise RuntimeError('Found an unclosed complete branch')
        i=order[depth]
        for chain in chains:
            candidate=rows+choices[i,chain]
            proof=farkas(candidate,m)
            if proof is None:
                tokens.append(None)
                walk(depth+1,candidate)
            else:
                support,weights=proof
                tokens.append([[int(k),str(w)] for k,w in zip(support,weights)])
            if progress and len(tokens)%progress==0:
                print(f'tokens={len(tokens)} depth={depth} seconds={time.perf_counter()-start:.1f}',flush=True)
    walk(0,base)
    certificate={
        'schema':'ncd.exact-interval-chain-lower.v1',
        'system_sha256':_digest(system.to_dict()),
        'epsilon':repr(0.101),
        'excluded_states':m,
        'proof':tokens,
    }
    print('generated',len(tokens),'in',round(time.perf_counter()-start,3),'seconds',flush=True)
    print(verify_interval_chain_exclusion(certificate),flush=True)
    return certificate

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('states',type=int)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--progress',type=int,default=1000)
    args=parser.parse_args()
    cert=generate(args.states,args.progress)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(cert,separators=(',',':')).encode('utf-8')
    with args.out.open('wb') as handle:
        with gzip.GzipFile(filename='',fileobj=handle,mode='wb',compresslevel=9,mtime=0) as zipped:
            zipped.write(payload)
    print('saved',args.out,args.out.stat().st_size,'bytes',flush=True)
