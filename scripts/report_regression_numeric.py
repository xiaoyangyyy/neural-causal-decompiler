from pathlib import Path
import json
root=Path('.')
lines=['# Cross-fit regression internal mechanism audit: seeds 993 / 994','',
'Protocol: `docs/REGRESSION_NUMERIC_PROTOCOL.md`. Both runs were independently replayed from world generation through regression tracing, pair construction, mapping retraining, controls and metrics.','',
'Each run has 5,120 worlds, 28 synchronized regression groups, 384 training pairs, 1,024 test pairs and 338 held-out compatible pair masks.','',
'## Aggregate results','',
'| Seed | Site | Method | Natural NMSE | Target NMSE | No-intervention NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
for seed in (993,994):
 s=json.loads((root/f'runs/regression_numeric_seed{seed}/summary.json').read_text())
 for site,methods in s['sites'].items():
  for method,row in methods.items():
   ns=row['numeric_nodes'];nat=sum(x['natural_n']*x['natural_nmse'] for x in ns)/sum(x['natural_n'] for x in ns)
   def agg(part,key):
    count=sum(x[part]['n'] for x in ns);return sum(x[part]['n']*x[part][key] for x in ns if x[part]['n'])/count
   b=row['behavioral'];acc='N/A' if b['informative_accuracy'] is None else f"{b['informative_accuracy']:.1%}"
   lines.append(f"| {seed} | {site} | {method} | {nat:.3f} | {agg('targeted','nmse'):.3f} | {agg('targeted','no_intervention_nmse'):.3f} | {agg('collateral','nmse'):.3f} | {b['informative_pairs']} | {acc} |")
lines+=['','## Numeric mapping by regression step','',
'| Seed | Site | Direction | Fold | Step | Natural NMSE | Target NMSE | No-intervention NMSE |','|---:|---|---|---:|---|---:|---:|---:|']
for seed in (993,994):
 s=json.loads((root/f'runs/regression_numeric_seed{seed}/summary.json').read_text());labels=[]
 for g in s['groups']:
  index=int(g['occurrences'][0].split('/')[1]);direction='Y_given_X' if index in (8,10) else 'X_given_Y';labels.append((direction,g['ast']['fold'],g['ast']['role']))
 for site in s['sites']:
  for (direction,fold,role),row in zip(labels,s['sites'][site]['numeric']['numeric_nodes']):
   lines.append(f"| {seed} | {site} | {direction} | {fold} | {role} | {row['natural_nmse']:.3f} | {row['targeted']['nmse']:.3f} | {row['targeted']['no_intervention_nmse']:.3f} |")
lines+=['','## Interpretation','',
'- Natural probes recover substantial coefficient and normalization information (pooled NMSE 0.56–0.71), but this does not imply that the readout directions implement the computation.',
'- Numeric interchange target NMSE is 1.89–3.00 and is worse than the no-intervention baseline in every aggregate comparison. It is nevertheless far better than behavior-only and random controls.',
'- Behavior-only mappings can attain higher categorical accuracy while increasing numerical error by two or three orders of magnitude. Categorical effects alone are inadequate evidence for algorithm recovery.',
'- Informative-pair behavior accuracy remains low: at most 13.6% for the numeric method. The full cross-fit regression mechanism has not been causally aligned.',
'- All declared raw statistical operations are now executable and internally addressable, but the frozen teachers do not show high-fidelity interchange under the tested linear orthogonal mappings.',
'- Seeds 993/994 are observed and cannot be reused for confirmation of a revised mapping model.']
(root/'RESULTS_REGRESSION_NUMERIC.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
accept={'state':'verified','runs':{str(seed):{'worlds':5120,'groups':28,'sites':['representation','head_linear','head_tanh'],'train_pairs':384,'test_pairs':1024} for seed in (993,994)},
 'verification':'independent replay of worlds, expanded regression ASTs, pairs, probes, mapping retraining, controls and all metrics; both processes exited 0','science_not_certified':True}
(root/'validation/regression_numeric_acceptance.json').write_text(json.dumps(accept,indent=2),encoding='utf-8')
p=root/'WORK_STATUS.md';s=p.read_text(encoding='utf-8');s+='''
## 交叉拟合回归内部机制
- `trace_regression=True` 展开 X→Y/Y→X、两折、mean/std 与五个固定基函数系数；相同回归在 residual dependence/error 中同步，合计 28 组。
- mean/std→beta 的依赖被写入组合检查；默认及 dependence 程序 ID 均保持历史一致。
- quick 992：512 worlds、28 groups、48/96 pairs，完整重放通过。
- 正式 993/994：每组 5,120 worlds、384/1,024 pairs、338 个未见兼容双组 mask；两组独立重放通过（validation/regression_numeric_acceptance.json）。
- 自然读出 NMSE 0.56–0.71；数值干预 NMSE 1.89–3.00 且均差于无干预基线，行为最高 13.6%。回归内部信息可读，但未恢复可交换机制。详见 RESULTS_REGRESSION_NUMERIC.md。
''';p.write_text(s,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';s=p.read_text(encoding='utf-8');s=s.replace('893/894 已展开 dependence 核但干预恢复弱，regression 内部仍不透明','893/894 已展开 dependence 核但干预恢复弱；993/994 已展开 cross-fit regression 且自然可读但干预恢复失败');p.write_text(s,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8');s+='\nCross-fit regression internal intervention results: [RESULTS_REGRESSION_NUMERIC.md](RESULTS_REGRESSION_NUMERIC.md).\n';p.write_text(s,encoding='utf-8')
print('regression report and acceptance written')
