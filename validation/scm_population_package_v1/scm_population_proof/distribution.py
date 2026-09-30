"""Population total-variation boundary for finite empirical SCM laws."""
from .compact import derive as support, fingerprint, rational
from .moments import derive as moments


def derive(world, model, support_certificate=None):
    finite = support(model)
    if support_certificate is not None and finite != support_certificate:
        raise ValueError('Empirical finite-support proof does not match the model')
    true_moment = moments(world, '1')
    if world['noise_family'] not in ('gaussian', 'laplace', 'student'):
        raise ValueError('Noise density not approved for this theorem')
    if rational(world['noise_scale']) <= 0:
        raise ValueError('Continuous true noise scale must be positive and finite')
    if type(world.get('root_shift')) is not bool:
        raise ValueError('True root shift must be an explicit Boolean')
    n = len(world['graph'])
    if finite['nodes'] != n:
        raise ValueError('SCM and true world dimensions differ')
    atoms = 1
    for values in model['noise_samples']:
        atoms *= len(values)
    return {
        'schema': 'ncd.population-empirical-tv-boundary.v1',
        'status': 'refuted-scoped',
        'claim': 'The finite empirical-residual SCM is within total-variation distance strictly below 1 of the ideal continuous true SCM law',
        'true_world_sha256': fingerprint(world), 'estimated_model_sha256': finite['model_sha256'],
        'support_certificate_sha256': fingerprint(finite),
        'node_count': n, 'estimated_joint_atom_count_upper': str(atoms),
        'intervention_scope': 'Every compatible intervention subset leaving at least one node un-intervened; each observed do value is in [-1,1]',
        'total_variation_distance': '1',
        'witness_event': 'The finite set of all estimated joint outcomes for the fixed intervention',
        'proof': 'The estimated joint law assigns probability one to its finite outcome set. Any free true coordinate equals a function of its parents plus its own independent continuous noise; its conditional and marginal laws have no atoms. Thus the true joint law assigns probability zero to that finite set.',
        'true_noise_families': true_moment['noise_laws'],
        'semantic_scope': 'ideal continuous population laws, not deterministic seeded NumPy PRNG or floating device measures',
        'not_proved': ['small Wasserstein distance fails', 'fully intervened laws are different',
                       'the true graph or mechanisms were recovered',
                       'the original compound causal-decompilation objective is resolved'],
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def verify(world, model, certificate, support_certificate=None):
    expected = derive(world, model, support_certificate)
    if certificate != expected:
        raise ValueError('Total-variation boundary certificate mismatch')
    return {'status': 'verified', 'conclusion': 'refuted-scoped',
            'original_objective_achieved': False}
