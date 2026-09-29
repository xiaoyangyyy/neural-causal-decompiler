"""Full mathematical export and interval execution of actual frozen graph teachers.

Includes node contexts, masked multihead attention, learned layer normalization,
factorized orientation scores and reverse-pair symmetrization. No oracle graph.
"""
from pathlib import Path
from fractions import Fraction as Q
from ncd.io import digest
from ncd.proof_intervals import Interval,affine,hyperbolic_tangent,attention
from ncd.discovery_fidelity_proof import average,variance,stack
from .statistical_frontend import maximum,graph_features,standardize,negative_exponential


def export_graph(checkpoint):
    import torch
    from ncd.graph_model import load_graph_model
    from ncd.node_context_graph import load_node_context_graph
    from ncd.active_intervention_graph import load_active_factorized_graph
    from ncd.factorized_node_context_graph import FactorizedNodeContextGraphDiscoverer
    saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=True);architecture=saved.get('architecture','pair_attention_v1')
    if architecture in ('active_input_observational_control_v1','active_input_intervention_v1'):model=load_active_factorized_graph(checkpoint)
    elif architecture=='node_context_v1':model=load_node_context_graph(checkpoint)
    elif architecture=='factorized_node_context_v1':
        model=FactorizedNodeContextGraphDiscoverer(saved['width']);model.load_state_dict(saved['state_dict']);model.eval()
    elif architecture=='pair_attention_v1':model=load_graph_model(checkpoint)
    else:raise ValueError('Unsupported graph teacher architecture '+architecture)
    state=model.state_dict()
    def numbers(v):
        def walk(x):return [walk(a) for a in x] if isinstance(x,list) else str(Q(x))
        return walk(v.detach().cpu().tolist())
    def layer(prefix,activation):return {'weights':numbers(state[prefix+'.weight']),'bias':numbers(state[prefix+'.bias']),'activation':activation}
    def layers(prefix):return [layer(prefix+'.0','tanh'),layer(prefix+'.2','tanh')]
    source=Path(__file__).resolve().parents[1]/'ncd'
    names=['graph_model.py','model.py']
    if hasattr(model,'context'):names.append('node_context_graph.py')
    if hasattr(model,'skeleton_head'):names.append('factorized_node_context_graph.py')
    if 'active_input' in architecture:names.append('active_intervention_graph.py')
    return {'schema':'ncd.frozen-graph-math.v1','checkpoint_sha256':digest(checkpoint),'architecture':architecture,
       'source_sha256':{name:digest(source/name) for name in names},'width':model.width,
       'mean':numbers(model.mean),'std':numbers(model.std),'encoder':layers('encoder'),
       'context':layers('context') if hasattr(model,'context') else None,
       'attention':{'heads':model.attention.num_heads,'in_weights':numbers(state['attention.in_proj_weight']),
           'in_bias':numbers(state['attention.in_proj_bias']),'out':layer('attention.out_proj','identity')},
       'norm':{'weights':numbers(model.norm.weight),'bias':numbers(model.norm.bias),'epsilon':str(Q(model.norm.eps))},
       'head':({'trunk':[layer('head_trunk.0','tanh')],'skeleton':layer('skeleton_head','identity'),'orientation':layer('orientation_head','identity')}
           if hasattr(model,'skeleton_head') else {'plain':[layer('head.0','tanh'),layer('head.2','identity')]}),
       'semantics':'exact serialized coefficients and mathematical operators; device rounding excluded'}


def stable_attention(query,keys,values):
    if not query or not keys or len(keys)!=len(values) or any(len(k)!=len(query) for k in keys) or len({len(v) for v in values})!=1:raise ValueError('Attention dimensions')
    scale=Interval.point(len(query)).sqrt()
    scores=[sum((a*b for a,b in zip(query,key)),Interval.point(0))/scale for key in keys]
    peak=maximum(scores)
    centered=[Interval((v-peak).lo,min(Q(0),(v-peak).hi)) for v in scores]
    exps=[negative_exponential(v) for v in centered];total=sum(exps,Interval.point(0))
    # At least one shifted score is zero for every actual input, so total>=1.
    total=Interval(max(Q(1),total.lo),total.hi);probs=[v/total for v in exps]
    result=[]
    for j in range(len(values[0])):
        value=sum((p*row[j] for p,row in zip(probs,values)),Interval.point(0))
        # Attention is a convex combination even when separate probability bounds overlap.
        result.append(Interval(max(value.lo,min(row[j].lo for row in values)),min(value.hi,max(row[j].hi for row in values))))
    return result


def layer_norm(values,weights,bias,epsilon):
    if len(values)!=len(weights) or len(values)!=len(bias) or Q(epsilon)<=0:raise ValueError('Layer-normalization contract')
    mean=average(values);scale=(variance(values)+Q(epsilon)).sqrt()
    limit=Interval.point(len(values)-1).sqrt().hi
    normalized=[((v-mean)/scale).clip(-limit,limit) for v in values]
    # Sum of centered coordinates is zero: |z_j|<=sqrt(width-1).
    return [v*Q(w)+Q(b) for v,w,b in zip(normalized,weights,bias)]


def graph_logits(network,features,trace=False):
    if network.get('schema')!='ncd.frozen-graph-math.v1':raise ValueError('Unsupported graph export')
    n=len(features);w=network['width'];f=len(network['mean']);heads=network['attention']['heads']
    if n<2 or any(len(row)!=n for row in features) or any(len(token)!=f for row in features for token in row):raise ValueError('Graph feature dimensions')
    if w%heads or any(Q(s)<=0 for s in network['std']):raise ValueError('Graph normalization/head dimensions')
    encoded=[[stack(network['encoder'],[((x-Q(m))/Q(s)).clip(-20,20) for x,m,s in zip(token,network['mean'],network['std'])]) for token in row] for row in features]
    if any(len(token)!=w for row in encoded for token in row):raise ValueError('Graph encoded width')
    enriched=encoded
    if network['context'] is not None:
        outgoing=[[average([encoded[i][j][a] for j in range(n) if j!=i]) for a in range(w)] for i in range(n)]
        incoming=[[average([encoded[j][i][a] for j in range(n) if j!=i]) for a in range(w)] for i in range(n)]
        global_mean=[average([encoded[i][j][a] for i in range(n) for j in range(n) if i!=j]) for a in range(w)]
        enriched=[]
        for i in range(n):
            row=[]
            for j in range(n):
                inputs=encoded[i][j]+outgoing[i]+incoming[i]+outgoing[j]+incoming[j]+global_mean
                context=stack(network['context'],inputs)
                row.append([a+b for a,b in zip(encoded[i][j],context)])
            enriched.append(row)
    flat=[v for row in enriched for v in row];a=network['attention']
    projected=[affine(a['in_weights'],a['in_bias'],v) for v in flat]
    if any(len(v)!=3*w for v in projected):raise ValueError('Attention projection dimensions')
    active=[i*n+j for i in range(n) for j in range(n) if i!=j];head_width=w//heads
    normalized=[];attended=[]
    for index,p in enumerate(projected):
        concatenated=[]
        for h in range(heads):
            lo=h*head_width;hi=lo+head_width
            q=p[lo:hi];keys=[projected[k][w+lo:w+hi] for k in active];values=[projected[k][2*w+lo:2*w+hi] for k in active]
            concatenated+=stable_attention(q,keys,values)
        context=affine(a['out']['weights'],a['out']['bias'],concatenated);attended.append(context)
        v=[x+y for x,y in zip(flat[index],context)];ln=network['norm']
        normalized.append(layer_norm(v,ln['weights'],ln['bias'],ln['epsilon']))
    raw=[]
    for values in normalized:
        head=network['head']
        if 'plain' in head:raw.append(stack(head['plain'],values))
        else:
            trunk=stack(head['trunk'],values);s=affine(head['skeleton']['weights'],head['skeleton']['bias'],trunk)
            o=affine(head['orientation']['weights'],head['orientation']['bias'],trunk)
            if len(s)!=1 or len(o)!=3:raise ValueError('Factorized head dimensions')
            # logsoftmax(o)-max(logsoftmax(o)) == o-max(o), exactly.
            peak=maximum(o)
            centered=[Interval((v-peak).lo,min(Q(0),(v-peak).hi)) for v in o]
            raw.append([Interval.point(0)]+[s[0]+v for v in centered])
    if any(len(v)!=4 for v in raw):raise ValueError('Graph class dimensions')
    swap=[0,2,1,3]
    logits=[[[ (raw[i*n+j][c]+raw[j*n+i][swap[c]])/2 for c in range(4)] for j in range(n)] for i in range(n)]
    if trace:return logits,{'encoder':encoded,'enriched':enriched,'attention':attended,'norm':normalized,'raw_logits':raw}
    return logits


def graph_input_features(network,observations,responses=None):
    features=graph_features(observations);n=len(features);f=len(network['mean'])
    if f==20:
        if responses is not None:raise ValueError('Response data supplied to observational-only architecture')
        return features
    if f!=24:raise ValueError('Unsupported graph feature schema')
    for i in range(n):
        for j in range(n):features[i][j]+=[Interval.point(0)]*4
    if network['architecture']=='active_input_observational_control_v1':
        if responses is not None:raise ValueError('Response data prohibited in observational control')
        return features
    if responses is None or len(responses)!=n:raise ValueError('Active graph needs plus/minus allowed do-sample arrays per source')
    cols=list(zip(*observations));scales=[]
    from .statistical_frontend import floor_interval
    for c in cols:scales.append(floor_interval(variance(c).sqrt(),Q(1e-6)))
    for source,pair in enumerate(responses):
        if len(pair)!=2:raise ValueError('Expected plus/minus response datasets')
        plus,minus=pair
        if len(plus)!=len(observations) or len(minus)!=len(observations) or any(len(r)!=n for r in plus+minus):raise ValueError('Active sample dimensions')
        for target in range(n):
            if source==target:continue
            scale=scales[target];contrast=[(a[target]-b[target])/scale for a,b in zip(plus,minus)]
            features[source][target][-4:]=[average(contrast),average([v.square() for v in contrast]).sqrt(),
                average([(a[target]-b[target]).abs() for a,b in zip(plus,observations)])/scale,
                average([(a[target]-b[target]).abs() for a,b in zip(minus,observations)])/scale]
    return features


def certify_graph_box(network,domain,program_labels,input_kind='features',responses=None):
    def intervals(x):
        if isinstance(x,list) and len(x)==2 and all(isinstance(v,(str,int,float)) and not isinstance(v,bool) for v in x):return Interval.from_dict(x)
        if not isinstance(x,list):raise ValueError('Invalid graph proof domain')
        return [intervals(v) for v in x]
    try:
        if input_kind not in ('features','raw_statistics'):raise ValueError('Unsupported graph proof input kind')
        values=intervals(domain)
        features=values if input_kind=='features' else graph_input_features(network,values,intervals(responses) if responses is not None else None)
        logits=graph_logits(network,features);n=len(logits)
        if len(program_labels)!=n or any(len(r)!=n for r in program_labels):raise ValueError('Candidate graph label dimensions')
        pairs=[];status='proved'
        for i in range(n):
            for j in range(i+1,n):
                label=program_labels[i][j]
                if type(label)is not int or label not in range(4):raise ValueError('Invalid candidate pair label')
                winner=[c for c in range(4) if all(logits[i][j][c].lo>logits[i][j][d].hi for d in range(4) if d!=c)]
                result='proved' if winner==[label] else 'refuted' if winner else 'unresolved'
                if result=='refuted':status='refuted'
                elif result=='unresolved' and status!='refuted':status='unresolved'
                pairs.append({'pair':[i,j],'candidate_label':label,'winners':winner,'status':result,'logits':[v.to_dict() for v in logits[i][j]]})
        return {'schema':'ncd.graph-box-fidelity.v1','status':status,'domain':domain,'input_kind':input_kind,'responses':responses,
            'program_labels':program_labels,'pairs':pairs,'scope':'all unordered pair labels on the complete declared input box; fixed candidate only',
            'does_not_prove_true_graph_or_global_program':True,'semantics':network['semantics']}
    except ValueError as exc:
        return {'schema':'ncd.graph-box-fidelity.v1','status':'unresolved','domain':domain,'input_kind':input_kind,'responses':responses,
            'program_labels':program_labels,'reason':str(exc),'does_not_prove_true_graph_or_global_program':True}


def verify_graph_box(network,certificate):
    expected=certify_graph_box(network,certificate['domain'],certificate['program_labels'],certificate['input_kind'],certificate['responses'])
    if certificate!=expected:raise ValueError('Graph fidelity certificate mismatch')
    return {'status':'verified','conclusion':certificate['status'],'original_requirement_closed':False}
