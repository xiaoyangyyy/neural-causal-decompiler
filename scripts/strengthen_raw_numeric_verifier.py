from pathlib import Path
p=Path('ncd/raw_numeric_experiment.py');s=p.read_text(encoding='utf-8')
insert='''def _trace_values(trace,addresses):
    values=[];visited=[]
    for address in addresses:
        record=trace.records[address];values.append(np.asarray(record['value']));visited.append(np.asarray(record['visited'],dtype=bool))
    return np.column_stack(values),np.column_stack(visited)
'''
s=s.replace('def _schema(executor,addresses):',insert+'\ndef _schema(executor,addresses):')
s=s.replace("trace=executor.execute(data[split]);features[split]=trace.features;natural[split]=numeric_values(executor,data[split],addresses)","trace=executor.execute(data[split]);features[split]=trace.features;natural[split]=_trace_values(trace,addresses)")
s=s.replace("seen|=ids;data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=numeric_values(executor,data[split],addresses)","seen|=ids;data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses)")
old='''        with np.load(folder/'bases.npz') as z:bases={k:z[k] for k in z.files}
        for name,basis in bases.items():
            numeric=audit_numeric_values(probe,ht,basis,test,*natural['alignment_test'],test_targets,test_visited);behavioral=measure_joint_mapping(decoder,ht,basis,test)
'''
new='''        with np.load(folder/'bases.npz') as z:bases={k:z[k] for k in z.files}
        expected_methods={}
        for name,weight,shuffle,offset in (('numeric',c.numeric_weight,False,0),('behavior_only',0.,False,0),('shuffled',c.numeric_weight,True,1000)):
            basis,record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                seed=c.seed+offset,numeric_weight=weight,shuffle_targets=shuffle);expected_methods[name]=basis
            if record!=read_json(folder/(name+'_training.json')):raise ValueError("Raw mapping training replay mismatch")
        rng=np.random.default_rng(c.seed+2000);expected_methods['random']=np.linalg.qr(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)))[0].reshape(hf.shape[1],len(addresses),c.rank)
        if set(bases)!=set(expected_methods):raise ValueError("Raw method coverage mismatch")
        for name,basis in bases.items():
            np.testing.assert_allclose(basis,expected_methods[name],rtol=0,atol=1e-7)
            numeric=audit_numeric_values(probe,ht,basis,test,*natural['alignment_test'],test_targets,test_visited);behavioral=measure_joint_mapping(decoder,ht,basis,test)
'''
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
