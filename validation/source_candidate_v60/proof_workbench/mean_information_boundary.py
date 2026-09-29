"""All post-normalization readouts lose the raw sample mean.

An architecture-level information obstruction, not one failed fitted map.
The certificate binds an existing teacher, program occurrence and fit scale.
"""
from fractions import Fraction as Q
from pathlib import Path
import json
from ncd.io import digest
from ncd.proof_intervals import Interval
from ncd.discovery_fidelity_proof import export_discoverer
from proof_extensions.statistical_frontend import erf_interval

MODEL_SOURCE='ee732b8f06c153951e6f3e559c9511591e8e6e632d815766c5192aff50bb3602'


def target_binding(run):
    from ncd.rules import Rule
    from ncd.raw_program_trace import RawDiscoveryExecutor
    run=Path(run);rule=Rule.from_dict(json.loads((run/'program.json').read_text()))
    executor=RawDiscoveryExecutor(rule,trace_dependence=True,trace_regression=True)
    probe=json.loads((run/'sites/head_tanh/probe.json').read_text())
    matches=[]
    for i,address in enumerate(probe['addresses']):
        path=address.split(':',1)[1];members=executor._all_groups.get(path,())
        if members and all(executor._nodes.get(m) and executor._nodes[m].to_dict()=={'op':'mean','args':[{'op':'var','index':0}]} for m in members):
            matches.append({'address':address,'members':list(members),'fit_variance':str(Q(probe['target_variance'][i]))})
    if len(matches)!=1 or Q(matches[0]['fit_variance'])<=0:raise ValueError('Historical raw mean group not uniquely bound')
    return matches[0]


def _centered(data):
    n=len(data);means=[sum((row[j] for row in data),Q(0))/n for j in range(2)]
    centered=[[row[j]-means[j] for j in range(2)] for row in data]
    variance=[sum((row[j]**2 for row in centered),Q(0))/n for j in range(2)]
    return means,centered,variance


def certify_mean_information(run='runs/oblique_seed1193',samples=96,shift='1',epsilon='1/100',requested_success='99/100'):
    run=Path(run);shift=Q(shift);epsilon=Q(epsilon);success=Q(requested_success)
    if type(samples)is not int or samples<16 or shift<=0 or epsilon<0 or not 0<success<=1:raise ValueError('Mean-information contract')
    if digest('ncd/model.py')!=MODEL_SOURCE:raise ValueError('Unreviewed normalization implementation')
    network=export_discoverer(run/'teacher.pt');target=target_binding(run)
    data=[[Q(2*i-samples+1,32),Q(i%11-5,4)] for i in range(samples)]
    moved=[[x+shift,y] for x,y in data]
    ma,ca,va=_centered(data);mb,cb,vb=_centered(moved)
    if ca!=cb or va!=vb or mb[0]-ma[0]!=shift:raise ValueError('Translation witness identity failed')
    normalizer=Interval.point(target['fit_variance']).sqrt()
    deterministic_lower=Interval.point(shift)/(2*normalizer)
    # Under an iid bivariate Gaussian SCM, the sample mean is independent of
    # every centered row. All three cut states are functions of centered rows.
    # A symmetric interval of given length captures most N(0,1/n) mass when
    # centered at zero; independent randomized predictions cannot improve it.
    success_bound=erf_interval(Interval.point(epsilon)*Interval.point(Q(samples)*Q(target['fit_variance'])/2).sqrt())
    dependencies=['ncd/model.py','ncd/neural_sites.py','ncd/raw_program_trace.py','ncd/cdir.py',
        'proof_extensions/statistical_frontend.py','proof_workbench/mean_information_boundary.py']
    return {'schema':'ncd.post-normalization-mean-impossibility.v1','status':'refuted' if deterministic_lower.lo>epsilon and success_bound.hi<success else 'unresolved',
        'run':run.as_posix(),'checkpoint_sha256':network['checkpoint_sha256'],'program_sha256':digest(run/'program.json'),
        'probe_sha256':digest(run/'sites/head_tanh/probe.json'),'source_sha256':{p:digest(p) for p in dependencies},
        'target':target,'samples':samples,'shift':str(shift),'epsilon':str(epsilon),'requested_success':str(success),
        'data_a':[[str(v) for v in row] for row in data],'data_b':[[str(v) for v in row] for row in moved],
        'means_a':[str(v) for v in ma],'means_b':[str(v) for v in mb],'common_variances':[str(v) for v in va],
        'normalized_minimax_absolute_error_lower':deterministic_lower.to_dict(),
        'statistical_success_probability_upper':str(success_bound.hi),'statistical_failure_probability_lower':str(1-success_bound.hi),
        'statistical_normalized_mse_lower':str(1/(Q(samples)*Q(target['fit_variance']))),
        'gaussian_scm':'X=U0; Y=X/2+U1; independent iid standard Gaussian noises per row',
        'mapping_family':'every deterministic function of any or all post-normalization internal states; independent randomization also cannot improve the statistical bound',
        'cut_sites':['representation','head_linear','head_tanh'],
        'architecture_identity':'Subtracting each column mean makes both centered matrices equal. Standard deviations, clipping, encoders, averages, log scales, and all head prefixes then agree, including swapped-column computation.',
        'deterministic_argument':'Equal hidden states force a common prediction; two true raw means differ by shift, so at least one normalized absolute error is shift/(2*fit_std) or greater.',
        'gaussian_argument':'For iid joint Gaussian rows, Cov(sample_mean, each centered row)=0, hence Gaussian independence. The raw X mean is N(0,1/n) independent of all cut states. Conditional interval probability is maximized at prediction zero and bounded by erf(epsilon*fit_std*sqrt(n/2)).',
        'execution_coverage':'RawDiscoveryExecutor evaluates all feature roots eagerly; every listed mean occurrence is evaluated on both valid datasets before Rule branches.',
        'pre_normalization_or_raw_input_maps_covered':False,'other_numeric_intermediates_refuted':False,
        'device_rounding_covered':False,'full_original_R5_closed':False}


def verify_mean_information(certificate):
    expected=certify_mean_information(certificate['run'],certificate['samples'],certificate['shift'],certificate['epsilon'],certificate['requested_success'])
    if expected!=certificate:raise ValueError('Mean information certificate, source, checkpoint or scale mismatch')
    return {'status':'verified','conclusion':certificate['status'],'mapping_family_excluded':certificate['mapping_family'],
        'statistical_success_probability_upper':certificate['statistical_success_probability_upper'],'original_R5_closed':False}
