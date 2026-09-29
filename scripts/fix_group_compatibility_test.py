from pathlib import Path
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');old='''    variance=next(a for a in groups if e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op=='var_stat')
    square_root=next(a for a in groups if e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op=='sqrt' and
        any(v.startswith(p+'/') for p in e.catalog[square_root.split(':',1)[1]]['members'] for v in e.catalog[variance.split(':',1)[1]]['members']))
''';new='''    def op(a):return e._nodes[e.catalog[a.split(':',1)[1]]['members'][0]].op
    candidates=[]
    for variance in [a for a in groups if op(a)=='var_stat']:
        for square_root in [a for a in groups if op(a)=='sqrt']:
            vm=e.catalog[variance.split(':',1)[1]]['members'];sm=e.catalog[square_root.split(':',1)[1]]['members']
            if any(v.startswith(p+'/') for p in sm for v in vm):candidates.append((variance,square_root))
    variance,square_root=candidates[0]
''';assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
