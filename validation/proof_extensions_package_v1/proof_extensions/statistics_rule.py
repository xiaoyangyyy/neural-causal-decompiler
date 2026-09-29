"""Correct historical Rule fidelity through its actual statistics.extract_one input."""
from fractions import Fraction as Q
from ncd.proof_intervals import Interval
from ncd.frozen_mechanism_proof import expression_interval
from ncd.discovery_fidelity_proof import discoverer_logits
from .statistical_frontend import raw_feature


def labels_and_trace(program,data):
    from ncd.rules import Rule
    from ncd.statistics import FEATURES
    Rule.from_dict(program)
    if tuple(program['names'])!=FEATURES:raise ValueError('Actual statistics schema mismatch')
    cache={};paths=[]
    def expression(node):
        if node['op']=='var':
            i=node['index']
            if i not in cache:cache[i]=raw_feature(data,i)
            return cache[i]
        if node['op']=='constant':return Interval.point(node['value'])
        args=[expression(a) for a in node.get('args',[])]
        return expression_interval({'op':node['op'],'args':[{'op':'var','index':i} for i in range(len(args))]},args)
    def visit(tree,path):
        if 'label' in tree:
            paths.append({'path':path,'label':tree['label']});return {tree['label']}
        value=expression(tree['expr']);threshold=Q(tree['threshold'])
        if value.hi<threshold:return visit(tree['left'],path+['left'])
        if value.lo>=threshold:return visit(tree['right'],path+['right'])
        return visit(tree['left'],path+['left'])|visit(tree['right'],path+['right'])
    labels=visit(program['tree'],[])
    return labels,{'features':{str(i):value.to_dict() for i,value in sorted(cache.items())},'reachable_paths':paths,
        'frontend':'ncd.statistics.extract_one, not ncd.cdir.crossfit_prediction'}


def certify_rule_box(network,program,domain):
    try:
        data=[[Interval.from_dict(x) for x in row] for row in domain]
        labels,trace=labels_and_trace(program,data);logits=discoverer_logits(network,data)
        winners=[j for j in range(4) if all(logits[j].lo>logits[k].hi for k in range(4) if j!=k)]
        status='proved' if len(winners)==1 and labels==set(winners) else 'refuted' if len(winners)==len(labels)==1 else 'unresolved'
        return {'schema':'ncd.actual-statistics-rule-fidelity.v2','status':status,'domain':domain,
            'logit_enclosures':[v.to_dict() for v in logits],'rule_labels':sorted(labels),'neural_winners':winners,'trace':trace,
            'scope':'complete declared raw dataset box for this frozen mathematical network and this fixed actual-statistics Rule',
            'does_not_close_full_original_domain':True,'device_rounding_certified':False}
    except ValueError as exc:
        return {'schema':'ncd.actual-statistics-rule-fidelity.v2','status':'unresolved','domain':domain,'reason':str(exc),
            'does_not_close_full_original_domain':True,'device_rounding_certified':False}


def verify_rule_box(network,program,certificate):
    if certificate!=certify_rule_box(network,program,certificate['domain']):raise ValueError('Actual statistics Rule certificate mismatch')
    return {'status':'verified','conclusion':certificate['status'],'original_claim_closed':False}
