from pathlib import Path
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');s+='''

def test_grouped_raw_interventions_use_world_level_pair_engine():
    from ncd.mechanism_pairs import compatible_combinations,execution_conditioned_pairs
    data=np.concatenate([worlds() for _ in range(6)],axis=0);e=RawDiscoveryExecutor(rule());addresses=raw_scalar_groups(e)
    masks=compatible_combinations(e,addresses,max_order=2)
    pairs,record=execution_conditioned_pairs(e,data,addresses,masks,count=4,seed=17)
    assert len(pairs.base)==4 and record['neural_or_truth_filtering'] is False
    natural=e.execute(data)
    for i in range(4):
        patches={a:natural.records[a]['value'][pairs.sources[i,j]:pairs.sources[i,j]+1]
                 for j,a in enumerate(addresses) if pairs.masks[i,j]}
        replay=e.execute(data[pairs.base[i:i+1]],patches)
        assert replay.output[0]==pairs.target[i]
''';p.write_text(s,encoding='utf-8')
