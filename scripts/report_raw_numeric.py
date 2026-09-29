from pathlib import Path
import json,datetime
root=Path('.')
def label(ast):
 op=ast['op']
 if op in ('mean','var_stat') and ast['args'][0]['op']=='var':return op+'_'+('x' if ast['args'][0]['index']==0 else 'y')
 if op=='sqrt':return 'std_'+('x' if ast['args'][0]['args'][0]['index']==0 else 'y')
 if op=='div':return 'variance_ratio'
 return 'corr'
lines=['# Raw statistical operation decompilation: seeds 793 / 794','',
'Protocol: `docs/RAW_NUMERIC_PROTOCOL.md`. Both formal runs were independently replayed from worlds through AST execution, pair construction, mapping retraining, and metric reconstruction.','',
'Each run contains 1,024 fit and 4,096 test worlds, 384 non-reused training pairs, 1,024 non-reused test pairs, eight structural-equivalence groups, and three neural cuts.','',
'## Aggregate numerical and behavioral results','',
'| Seed | Site | Method | Natural probe NMSE | Target NMSE | No-intervention target NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
for seed in (793,794):
 s=json.loads((root/f'runs/raw_numeric_seed{seed}/summary.json').read_text())
 for site,methods in s['sites'].items():
  for method,row in methods.items():
   ns=row['numeric_nodes'];nat=sum(x['natural_n']*x['natural_nmse'] for x in ns)/sum(x['natural_n'] for x in ns)
   def agg(part,key):
    count=sum(x[part]['n'] for x in ns);return sum(x[part]['n']*x[part][key] for x in ns if x[part]['n'])/count
   b=row['behavioral'];acc='N/A' if b['informative_accuracy'] is None else f"{b['informative_accuracy']:.1%}"
   lines.append(f"| {seed} | {site} | {method} | {nat:.3f} | {agg('targeted','nmse'):.3f} | {agg('targeted','no_intervention_nmse'):.3f} | {agg('collateral','nmse'):.3f} | {b['informative_pairs']} | {acc} |")
lines+=['','## Numeric method by operation group','',
'| Seed | Site | Group | Natural NMSE | Target NMSE | No-intervention NMSE |','|---:|---|---|---:|---:|---:|']
for seed in (793,794):
 s=json.loads((root/f'runs/raw_numeric_seed{seed}/summary.json').read_text());names=[label(g['ast']) for g in s['groups']]
 for site in s['sites']:
  for name,row in zip(names,s['sites'][site]['numeric']['numeric_nodes']):
   lines.append(f"| {seed} | {site} | {name} | {row['natural_nmse']:.3f} | {row['targeted']['nmse']:.3f} | {row['targeted']['no_intervention_nmse']:.3f} |")
lines+=['','## Interpretation','',
'- This is the first experiment in the repository that intervenes below the final statistical feature leaves. An internal variance exchange creates a hybrid computation from source variance and target-world downstream operations.',
'- Seed 793 shows moderate numeric target errors around one normalized variance. Behavior-only mappings have much larger numeric errors, confirming that behavioral agreement alone does not identify numerical semantics.',
'- In seed 794, the variance-ratio group has target NMSE around 1,900 and dominates the aggregate. Numeric training only slightly improves its no-intervention error. Several other groups improve, but the full eight-operation mechanism is not recovered.',
'- Behavior interchange accuracy remains low and behavior-only is often higher. Numerical recovery and categorical intervention fidelity are therefore not interchangeable claims.',
'- Dependence-kernel bandwidth/centering and cross-fit regression internals are still opaque operations in the current CDIR trace. Complete raw statistical algorithm recovery remains open.',
'- Seeds 793/794 are now observed results and cannot serve as untouched confirmation data for a revised method.']
(root/'RESULTS_RAW_NUMERIC.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
accept={'state':'verified','runs':{'793':{'worlds':5120,'groups':8,'sites':['representation','head_linear','head_tanh'],'train_pairs':384,'test_pairs':1024},'794':{'worlds':5120,'groups':8,'sites':['representation','head_linear','head_tanh'],'train_pairs':384,'test_pairs':1024}},'verification':'two independent verify_raw_numeric processes exited 0 after replaying worlds, ASTs, pairs, probes, all learned bases, controls and metrics','science_not_certified':True}
(root/'validation/raw_numeric_acceptance.json').write_text(json.dumps(accept,indent=2),encoding='utf-8')
p=root/'WORK_STATUS.md';t=p.read_text(encoding='utf-8');t+='''
## 原始统计运算干预
- ncd/raw_program_trace.py 从原始样本执行 14 个 CDIR 特征 AST，再执行冻结规则；内部标量可按 occurrence 或结构等价组干预。
- 八个组覆盖 mean/variance/std/correlation/variance ratio；祖先/后代组合在展开真实 AST 路径后被排除。
- ncd/fixed_numeric_mapping.py 使用一次预计算的非复用配对池训练映射，避免每个步骤重复计算原始统计量。
- quick 792：512 worlds、48/96 pairs，独立重放通过。失败的 quick 791 保留，原因是旧组兼容检查未展开祖先关系。
- 正式 793/794：每组 5,120 worlds、384 train pairs、1,024 test pairs、三个 neural sites；两组均重新训练映射并独立重放通过。证据 validation/raw_numeric_acceptance.json、RESULTS_RAW_NUMERIC.md。
- seed 794 的 variance ratio 发生极大 NMSE，完整八运算恢复失败；dependence 和 regression 内部仍未展开。
''';p.write_text(t,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';t=p.read_text(encoding='utf-8');t=t.replace('未恢复完整数值算法（RESULTS_JOINT.md）；293/294 数值审计已重放','未恢复完整数值算法（RESULTS_JOINT.md）；793/794 已把 mean/variance/std/correlation/variance-ratio 展开到原始样本并干预，但 variance-ratio 失败且 dependence/regression 内部仍不透明（RESULTS_RAW_NUMERIC.md）；293/294 数值审计已重放');p.write_text(t,encoding='utf-8')
p=root/'README.md';t=p.read_text(encoding='utf-8');t+='\nRaw statistical operation intervention results: [RESULTS_RAW_NUMERIC.md](RESULTS_RAW_NUMERIC.md).\n';p.write_text(t,encoding='utf-8')
print('raw numeric report and acceptance written')
