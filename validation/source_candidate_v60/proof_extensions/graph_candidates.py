"""Explicit lazy DAG families; retain ambiguity and contradictory neural predictions."""
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
import numpy as np
from ncd.proof_intervals import Interval,as_interval
from ncd.graphs import topological_order
from ncd.io import digest

SWAP=[0,2,1,3]


def graph_family(scores):
    n=len(scores)
    if n not in (2,3,5,8) or any(len(r)!=n for r in scores) or any(len(v)!=4 for r in scores for v in r):raise ValueError('Candidate score dimensions')
    values=[[[as_interval(v) for v in token] for token in row] for row in scores]
    def possible(token):return {c for c in range(4) if token[c].hi>=max(v.lo for v in token)}
    pairs=[];forced=np.zeros((n,n),bool)
    for i,j in combinations(range(n),2):
        # Include both ordered predictions; this operation commutes with relabeling even on ties.
        labels=possible(values[i][j])|{SWAP[c] for c in possible(values[j][i])}
        states=set(labels)-{3}
        if 3 in labels:states|={1,2}
        states=sorted(states)
        pairs.append({'pair':[i,j],'possible_neural_classes':sorted(labels),'allowed_dag_states':states})
        if states==[1]:forced[i,j]=True
        elif states==[2]:forced[j,i]=True
    witness=None;conflict=False
    try:
        order=topological_order(forced);rank={v:i for i,v in enumerate(order)};witness=forced.copy()
        for row in pairs:
            if row['allowed_dag_states']==[1,2]:
                i,j=row['pair'];a,b=(i,j) if rank[i]<rank[j] else (j,i);witness[a,b]=True
    except ValueError:conflict=True
    return {'schema':'ncd.lazy-neural-dag-family.v1','nodes':n,'pairs':pairs,
        'family_definition':'all acyclic loop-free graphs satisfying every allowed pair state',
        'forced_graph':forced.astype(int).tolist(),'forced_cycle_conflict':conflict,
        'nonempty_witness':None if witness is None else witness.astype(int).tolist(),
        'witness_is_not_identification_or_equivariant_selection':True,
        'true_causal_graph_coverage_proved':False,'is_certified_cpdag':False}


def family_contains(family,graph):
    a=np.asarray(graph,bool);n=family['nodes']
    if a.shape!=(n,n):return False
    try:topological_order(a)
    except ValueError:return False
    for row in family['pairs']:
        i,j=row['pair'];state=1 if a[i,j] else 2 if a[j,i] else 0
        if state not in row['allowed_dag_states']:return False
    return True


def certify_decoder_boundary(case='confidence_cycle'):
    from ncd.graph_model import decode_graph
    n=3;scores=np.zeros((n,n,4));scores[:,:,0]=1
    if case=='label_tie':
        scores[0,1]=scores[1,0]=[Q(1,8),Q(3,8),Q(3,8),Q(1,8)];permutation=[1,0,2]
    elif case=='confidence_cycle':
        for i,j in ((0,1),(1,2),(2,0)):
            scores[i,j]=[Q(1,8),Q(1,2),Q(1,4),Q(1,8)];scores[j,i]=scores[i,j,SWAP]
        permutation=[2,1,0]
    else:raise ValueError('Unknown decoder boundary case')
    before,diagnostic=decode_graph(scores)
    permuted=scores[np.ix_(permutation,permutation)];after,other=decode_graph(permuted)
    expected=before[np.ix_(permutation,permutation)]
    if np.array_equal(after,expected):raise ValueError('Not a decoder counterexample')
    source=Path(__file__).resolve().parents[1]/'ncd'
    return {'schema':'ncd.decoder-equivariance-boundary.v1','status':'refuted','case':case,
        'probabilities':[[[str(Q(float(v))) for v in token] for token in row] for row in scores],
        'permutation':permutation,'decoded_original':before.astype(int).tolist(),'decoded_permuted':after.astype(int).tolist(),
        'required_equivariant_output':expected.astype(int).tolist(),'original_diagnostic':diagnostic,'permuted_diagnostic':other,
        'scope':'legacy decode_graph is not node-permutation-equivariant on all valid reverse-symmetric probability inputs',
        'exact_arithmetic':'dyadic probabilities; integer graph operations; strict confidence comparisons or exact ties',
        'source_sha256':{name:digest(source/name) for name in ('graph_model.py','graphs.py')},
        'actual_trained_network_reachability_certified':False,'true_causal_identification_refuted':False}


def verify_decoder_boundary(certificate):
    if certificate!=certify_decoder_boundary(certificate['case']):raise ValueError('Decoder boundary certificate mismatch')
    return {'status':'verified','conclusion':'refuted','scope':certificate['scope'],'original_claim_closed':False}


def verify_graph_family(scores,family):
    if family!=graph_family(scores):raise ValueError('Graph candidate family mismatch')
    if family['nonempty_witness'] is not None and not family_contains(family,family['nonempty_witness']):raise ValueError('Invalid DAG family witness')
    return {'status':'verified','forced_cycle_conflict':family['forced_cycle_conflict'],'true_causal_graph_coverage_proved':False}
