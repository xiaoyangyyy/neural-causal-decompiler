"""Whole-machine lower bounds from continuum coverage and common successors.

A deterministic finite machine has a state-only output map and a common
input transition. The proof is independent of grids, partitions and a
chosen recurrent simulation relation. It excludes an 81-state machine
using its forced decoder bands, then an exact one-step neural witness.
"""
from __future__ import annotations
from fractions import Fraction as Q
from .affine_observability import _affine_network
from .continuous_compositional_realization import value
from .continuous_separation import _digest


def _direct_observation(system):
    if system.state_dim<4:
        raise ValueError('Four direct observed coordinates required')
    matrix,bias,_=_affine_network(system.observation)
    identity=tuple(tuple(Q(int(i==j)) for j in range(system.state_dim)) for i in range(4))
    if matrix!=identity or any(bias):
        raise ValueError('Four direct observed coordinates with zero bias required')


def _proof(system):
    _direct_observation(system)
    epsilon=Q(17,100)
    t=Q(319,1000)
    bands=((1-5*epsilon,epsilon),(1-3*epsilon,3*epsilon),(1-epsilon,5*epsilon))
    # With 81 states, the 81 separated anchors exhaust all decoder labels.
    # On any axis slice the other three anchors exclude every label but
    # three. Continuous coverage forces their two gaps to be <=2*epsilon.
    # Combining those gaps with endpoint coverage gives these tight bands.
    if not (Q(1,2)>2*epsilon and t<1-4*epsilon
            and all(a<=b for a,b in bands)
            and bands[0][1]<bands[1][0]<bands[2][0]):
        raise ValueError('Decoder-band theorem preconditions failed')
    action=[Q(1)]+[Q(0)]*(system.action_dim-1)
    witness=None
    for axis in range(4):
        left=[Q(0)]*system.state_dim
        left[(axis-1)%system.state_dim]=Q(1)
        right=list(left)
        right[axis]=t
        initial_left=value(system.observation,left)
        initial_right=value(system.observation,right)
        expected_left=left[:4]
        expected_right=right[:4]
        if initial_left!=expected_left or initial_right!=expected_right:
            raise ValueError('Initial observation identity replay failed')
        left_next=value(system.observation,value(system.transition,left+action))
        right_next=value(system.observation,value(system.transition,right+action))
        lo,hi=left_next[axis],right_next[axis]
        if not (lo<1-4*epsilon and hi>2*epsilon):
            continue
        required=(hi-epsilon,lo+epsilon)
        if not (required[0]>bands[0][1] and required[1]<bands[1][0]):
            raise ValueError('Common-successor interval is not in the decoder gap')
        witness={'observed_axis':axis,'common_initial_anchor':[str(x) for x in initial_left],
            'left_initial_state':[str(x) for x in left],
            'right_initial_state':[str(x) for x in right],
            'left_initial_observation':[str(x) for x in initial_left],
            'right_initial_observation':[str(x) for x in initial_right],
            'common_action':[str(x) for x in action],
            'left_next_observation':[str(x) for x in left_next],
            'right_next_observation':[str(x) for x in right_next],
            'required_successor_decoder_interval':[str(x) for x in required],
            'lower_gap_margin':str(hi-2*epsilon),
            'upper_gap_margin':str(1-4*epsilon-lo),
            'initial_middle_label_exclusion_margin':str(1-4*epsilon-t)}
        break
    if witness is None:
        raise ValueError('No transition-consistency witness found; lower remains unresolved')
    return {'schema':'ncd.transition-consistency-lower.v1','status':'certified',
        'system_sha256':_digest(system.to_dict()),'epsilon':str(epsilon),
        'state_dim':system.state_dim,'action_dim':system.action_dim,
        'packing_lower_bound':81,'excluded_machine_state_count':81,'lower_bound':82,
        'packing':{'observed_axes':[0,1,2,3],'axis_values':['0','1/2','1'],
            'point_count':81,'minimum_separation':'1/2','twice_epsilon':str(2*epsilon)},
        'continuum_coverage':{'candidate_machine_states':81,
            'packing_anchors_exhaust_all_labels':True,
            'labels_per_observed_axis_slice':3,
            'other_anchor_exclusion_margin':str(Q(1,2)-2*epsilon),
            'adjacent_decoder_gap_upper':str(2*epsilon),
            'necessary_decoder_bands':[[str(x) for x in band] for band in bands],
            'forbidden_decoder_gap':[str(epsilon),str(1-3*epsilon)]},
        'witness':witness,
        'scope':'any deterministic time-homogeneous finite machine with a state-only output map; arbitrary initial selector and partitions; full unit initial/action cubes; error at times zero and one',
        'boundary':'whole-machine lower 82; exact global minimum, code length and end-to-end causal decompilation remain open'}


def certify_transition_lower(system):
    return _proof(system)


def verify_transition_lower(system,certificate):
    if certificate.get('schema')!='ncd.transition-consistency-lower.v1':
        raise ValueError('Unsupported transition-consistency lower certificate')
    expected=_proof(system)
    if certificate!=expected:
        raise ValueError('Transition-consistency lower replay mismatch')
    return {'status':'verified','packing_lower_bound':81,'lower_bound':82}
