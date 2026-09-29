from pathlib import Path
import json
s=json.loads(Path('runs/relational_program_693_694/summary.json').read_text(encoding='utf-8'))
print(json.dumps(s['primary'],indent=2))
for mode in ('without_relations','with_relations'):
 for variant in ('local_composed','local_primitive','relational_primitive'):
  p=json.loads(Path(f'runs/relational_program_693_694/programs/{mode}/{variant}.json').read_text())
  indices=[]
  def walk(t):
   if 'label' in t:return
   def expr(e):
    if e['op']=='var':indices.append(e['index'])
    for a in e.get('args',[]):expr(a)
   expr(t['expr']);walk(t['left']);walk(t['right'])
  walk(p['tree'])
  print(mode,variant,'complexity',p['complexity'],'features',[p['names'][i] for i in sorted(set(indices))])
