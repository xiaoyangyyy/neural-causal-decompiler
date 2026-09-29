"""A feature-family impossibility certificate for an actual frozen Discoverer.

The family uses the mathematical semantics of statistics.extract_one; device
rounding is not included. No claim about richer raw-data CDIR programs follows.
"""
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
from ncd.io import digest
from ncd.proof_intervals import Interval
from ncd.discovery_fidelity_proof import export_discoverer,discoverer_logits


def template(scale_x='1/8',scale_y='1/4',repeat_count=6):
    sx,sy=Q(scale_x),Q(scale_y)
    if sx<Q(1,1000) or sy<Q(1,1000) or type(repeat_count)is not int or repeat_count<2 or repeat_count%2:
        raise ValueError('Positive scales above normalization floors and even multiplicity required')
    yy=[5,-5]*4+[1,-1,5,-5,5,-5,7,-7]
    base=[[(sx if i<8 else -sx),sy*v] for i,v in enumerate(yy)]
    return base,[row[:] for row in base for _ in range(repeat_count)]


def equality_witness(scale_x='1/8',scale_y='1/4',repeat_count=6):
    base,data=template(scale_x,scale_y,repeat_count);sx,sy=Q(scale_x),Q(scale_y)
    reflected=[[-x,y] for x,y in data]
    def moments(rows):
        return {f'{i},{j}':str(sum((x**i*y**j for x,y in rows),Q(0))/len(rows))
            for i,j in [(1,0),(0,1),(2,0),(0,2),(1,1),(3,0),(0,3),(2,1),(1,2),(4,0),(0,4)]}
    ma,mb=moments(data),moments(reflected)
    if ma!=mb or any(Q(ma[k]) for k in ('1,0','0,1','1,1','3,0','0,3','2,1','1,2')):
        raise ValueError('Feature moment equality failed')
    if Q(ma['2,0'])!=sx*sx or Q(ma['0,2'])!=25*sy*sy:
        raise ValueError('Normalization variance mismatch')
    standardized=[[x/sx,y/(5*sy)] for x,y in data]
    centers=[];fold_counts=[]
    for column in (0,1):
        order=sorted(range(len(data)),key=lambda i:(standardized[i][column],standardized[i][1-column]))
        whole=Counter(tuple(row) for row in standardized)
        for parity in (0,1):
            fold=Counter(tuple(standardized[i]) for i in order[parity::2])
            if any(2*fold[row]!=count for row,count in whole.items()):raise ValueError('Fold balance failed')
        train=sorted(standardized[i][column] for i in order[1::2])
        if sum(train,Q(0)) or sum((v*v for v in train),Q(0))!=len(train):raise ValueError('Train standardization failed')
        c=[]
        for q in (.1,.3,.5,.7,.9):
            position=Q(q)*(len(train)-1);k=position.numerator//position.denominator;f=position-k
            c.append(train[k] if not f else train[k]*(1-f)+train[k+1]*f)
        if c!=[Q(-1),Q(-1),Q(0),Q(1),Q(1)]:raise ValueError('Quantile reflection identity failed')
        centers.append([str(v) for v in c]);fold_counts.append(len(train))
    # Positive diagonal ridge makes every Gram system positive definite. Under
    # x -> -x the 14 basis columns undergo a signed permutation; the diagonal
    # regularizer commutes with it (the exceptional bias column stays fixed).
    return {'moments_a':ma,'moments_b':mb,'train_counts':fold_counts,'train_centers':centers,
        'basis_reflection_permutation':[0,1,2,3,4,5,6,7,8,13,12,11,10,9],
        'basis_reflection_signs':[1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,1],
        'ridge_diagonal':[str(Q(1e-6))]+[str(Q(.1))]*13,
        'identities':[
            'Every sorted parity fold contains half the multiplicity of every distinct row.',
            'All training means are zero and standardized training variances are one.',
            'Predictor reflection is a signed basis permutation preserving the ridge diagonal.',
            'Response reflection negates unique ridge coefficients, predictions and residuals.',
            'Cross-fit residual_y is unchanged; residual_x is negated at corresponding rows.',
            'RBF kernels depend only on squared distances; centering and normalization preserve equality.',
            'All 14 features agree: zero odd/mixed moments, equal even moments, scales and residual MSE.'
        ],'feature_count':14,'equality':'exact mathematical equality',
        'device_rounding_included':False,'datasets_are_row_permutations':Counter(map(tuple,data))==Counter(map(tuple,reflected))}


def _bindings(checkpoint):
    names=['ncd/statistics.py','ncd/model.py','ncd/discovery_fidelity_proof.py','ncd/proof_intervals.py',
        'proof_extensions/feature_collision.py']
    return {'checkpoint_sha256':digest(checkpoint),'source_sha256':{p:digest(p) for p in names}}


def certify_feature_collision(checkpoint,scale_x='1/8',scale_y='1/4',repeat_count=6):
    witness=equality_witness(scale_x,scale_y,repeat_count)
    base,data=template(scale_x,scale_y,repeat_count)
    network=export_discoverer(checkpoint);enclosures=[];labels=[]
    # Exact replication invariance of population mean/std, encoder averages,
    # encoder squared averages, and log scales reduces 96 rows to 16.
    for rows in (base,[[-x,y] for x,y in base]):
        logits=discoverer_logits(network,[[Interval.point(v) for v in row] for row in rows])
        winners=[i for i,v in enumerate(logits) if all(i==j or v.lo>w.hi for j,w in enumerate(logits))]
        labels.append(winners[0] if len(winners)==1 else None)
        enclosures.append([v.to_dict() for v in logits])
    different=all(v is not None for v in labels) and labels[0]!=labels[1]
    return {'schema':'ncd.fixed-statistic-family-impossibility.v1','status':'refuted' if different else 'unresolved',
        'bindings':_bindings(checkpoint),'checkpoint':str(checkpoint),'scale_x':str(Q(scale_x)),
        'scale_y':str(Q(scale_y)),'repeat_count':repeat_count,'samples':len(data),
        'data_a':[[str(v) for v in row] for row in data],
        'data_b':[[str(-x),str(y)] for x,y in data],'feature_equality_witness':witness,
        'network_logits':enclosures,'strict_network_labels':labels,
        'refuted_statement':'Some deterministic function of the fixed 14 mathematical statistics features matches this frozen network on both specified datasets.',
        'candidate_family':'All deterministic functions of these 14 features, independent of grammar, constants or program length.',
        'quantifier_argument':'Equal feature vectors force every deterministic program in the family to return the same label; certified distinct network labels contradict full-domain fidelity.',
        'source_truth_used':False,'hardware_semantics_covered':False,
        'richer_raw_data_programs_covered':False,'distributional_recovery_covered':False,
        'original_requirement_not_automatically_closed':True}


def verify_feature_collision(certificate,checkpoint=None):
    checkpoint=checkpoint or certificate['checkpoint']
    expected=certify_feature_collision(checkpoint,certificate['scale_x'],certificate['scale_y'],certificate['repeat_count'])
    if certificate!=expected:raise ValueError('Feature collision certificate or checkpoint mismatch')
    return {'status':'verified','conclusion':expected['status'],'strict_network_labels':expected['strict_network_labels'],
        'scope':'specified frozen network and mathematical 14-feature family'}
