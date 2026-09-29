"""Candidate-only polynomial CEGIS with verified neural counterexamples.

This is bounded synthesis, not a claim that a neural mechanism always admits a
short polynomial, or that successful extraction recovers the true SCM.
"""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import time
import numpy as np
from .io import read_json,save_json,digest
from .frozen_mechanism_proof import export_mechanism,certify_mechanism,verify_mechanism


def candidate_from_neural(model,points,parents,max_terms=8):
    from .mechanisms import neural_values
    parents=tuple(parents)
    if len(parents)>2:raise ValueError('Polynomial candidate budget allows at most two parents')
    exponents=[e for e in product(range(5),repeat=len(parents)) if sum(e)<=4]
    if not parents:exponents=[()]
    exponents=sorted(exponents,key=lambda e:(sum(e),e))
    matrix=np.column_stack([np.prod([points[:,p]**power for p,power in zip(parents,e)],axis=0)
                           if any(e) else np.ones(len(points)) for e in exponents])
    targets=neural_values(model,points)
    selected=[0]
    for _ in range(min(max_terms,len(exponents))-1):
        co=np.linalg.lstsq(matrix[:,selected],targets,rcond=None)[0]
        residual=targets-matrix[:,selected]@co
        scores=np.abs(matrix.T@residual)/np.maximum(np.linalg.norm(matrix,axis=0),1e-12)
        scores[selected]=-np.inf
        selected.append(int(np.argmax(scores)))
    co=np.linalg.lstsq(matrix[:,selected],targets,rcond=None)[0]
    def multiply(a,b):return {'op':'mul','args':[a,b]}
    terms=[]
    for index,coefficient in zip(selected,co):
        term={'op':'constant','value':float(coefficient)}
        for parent,power in zip(parents,exponents[index]):
            for _ in range(power):term=multiply(term,{'op':'var','index':parent})
        terms.append(term)
    expression=terms[0]
    for term in terms[1:]:expression={'op':'add','args':[expression,term]}
    return expression


def run_cegis(case,checkpoint,source_scm,node,domain,epsilon='1/100',rounds=3,max_boxes=32,seconds=300,seed=8100):
    from .mechanisms import load_mechanism
    case=Path(case);network=export_mechanism(checkpoint);model=load_mechanism(checkpoint)
    program=read_json(source_scm)['equations'][node]
    save_json(case/'network.json',network)
    rng=np.random.default_rng(seed)
    low=np.array([float(Q(x[0])) for x in domain]);high=np.array([float(Q(x[1])) for x in domain])
    points=rng.uniform(low,high,(64,len(domain)))
    history=[];deadline=time.monotonic()+seconds
    for index in range(rounds+1):
        save_json(case/f'candidate_{index}.json',program)
        proof=certify_mechanism(network,program,domain,epsilon,max_boxes=max_boxes,seconds=max(.001,deadline-time.monotonic()))
        proof.update(network_sha256=digest(case/'network.json'),program_sha256=digest(case/f'candidate_{index}.json'))
        save_json(case/f'proof_{index}.json',proof)
        verify_mechanism(network,program,proof)
        history.append({'round':index,'program':f'candidate_{index}.json','program_sha256':digest(case/f'candidate_{index}.json'),
            'proof':f'proof_{index}.json','proof_sha256':digest(case/f'proof_{index}.json'),'conclusion':proof['status']})
        if proof['status']!='refuted' or index==rounds or time.monotonic()>=deadline:break
        point=np.array([float(Q(x)) for x in proof['counterexample']['point']])
        # Only exact-checked neural witnesses enter new candidate fitting.
        points=np.vstack([points,np.tile(point,(16,1))])
        program=candidate_from_neural(model,points,network['parents'])
    certificate={'schema':'ncd.mechanism-cegis.v1','status':history[-1]['conclusion'],'history':history,
        'network_sha256':digest(case/'network.json'),'checkpoint_sha256':digest(checkpoint),
        'source_scm_sha256':digest(source_scm),'node':node,'domain':domain,'epsilon':str(Q(epsilon)),
        'seed':seed,'round_budget':rounds,'max_boxes':max_boxes,'seconds':seconds,
        'grammar':'total-degree <=4 polynomials, at most two parent coordinates and eight fitted terms',
        'global_minimum_proved':False,'teacher_oracle_equations_used':False,
        'scope':'this frozen mechanism, candidate grammar and declared closed box; candidate failure is not synthesis impossibility'}
    return certificate


def verify_cegis(case,certificate):
    case=Path(case);network=read_json(case/'network.json')
    if export_mechanism(case/'checkpoint.pt')!=network or digest(case/'network.json')!=certificate['network_sha256']:
        raise ValueError('CEGIS frozen checkpoint changed')
    if digest(case/'checkpoint.pt')!=certificate['checkpoint_sha256'] or digest(case/'source_scm.json')!=certificate['source_scm_sha256']:
        raise ValueError('CEGIS source binding changed')
    history=certificate['history']
    if not history or len(history)>certificate['round_budget']+1:raise ValueError('Invalid CEGIS rounds')
    from .mechanisms import load_mechanism
    model=load_mechanism(case/'checkpoint.pt')
    rng=np.random.default_rng(certificate['seed'])
    low=np.array([float(Q(x[0])) for x in certificate['domain']]);high=np.array([float(Q(x[1])) for x in certificate['domain']])
    points=rng.uniform(low,high,(64,len(low)))
    previous=None
    for index,row in enumerate(history):
        if row['round']!=index or row['program']!=f'candidate_{index}.json' or row['proof']!=f'proof_{index}.json':raise ValueError('CEGIS history changed')
        for field in ('program','proof'):
            if digest(case/row[field])!=row[field+'_sha256']:raise ValueError('CEGIS history artifact changed')
        program=read_json(case/row['program']);proof=read_json(case/row['proof'])
        if proof['domain']!=certificate['domain'] or proof['epsilon']!=certificate['epsilon'] or proof['normalizer']!=network['output_scale']:
            raise ValueError('CEGIS fidelity contract changed')
        if proof['network_sha256']!=certificate['network_sha256'] or proof['program_sha256']!=row['program_sha256']:raise ValueError('CEGIS dependency changed')
        result=verify_mechanism(network,program,proof)
        if result['conclusion']!=row['conclusion']:raise ValueError('False CEGIS round conclusion')
        if index==0 and program!=read_json(case/'source_scm.json')['equations'][certificate['node']]:raise ValueError('CEGIS did not begin from historical candidate')
        if index:
            if history[index-1]['conclusion']!='refuted':raise ValueError('Refinement without checked counterexample')
            point=np.array([float(Q(x)) for x in previous['counterexample']['point']])
            points=np.vstack([points,np.tile(point,(16,1))])
            if program!=candidate_from_neural(model,points,network['parents']):raise ValueError('Candidate was not regenerated from the frozen neural queries and checked witnesses')
        previous=proof
    if certificate['status']!=history[-1]['conclusion'] or certificate['global_minimum_proved'] or certificate['teacher_oracle_equations_used']:
        raise ValueError('False CEGIS overall conclusion')
    return {'status':'verified','conclusion':certificate['status'],'rounds':len(history),'global_minimum_proved':False}
