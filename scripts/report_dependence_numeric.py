from pathlib import Path
import json
root=Path('.')
lines=['# Dependence-kernel internal mechanism audit: seeds 893 / 894','',
'Protocol: `docs/DEPENDENCE_NUMERIC_PROTOCOL.md`. Both runs were independently replayed through world generation, expanded CDIR execution, causal pair construction, mapping retraining, controls and metrics.','',
'Each run contains 5,120 worlds, 18 dependence-step groups, 384 training pairs and 1,024 test pairs. There are 129 held-out compatible pair masks after excluding causal ancestor/descendant combinations.','',
'## Aggregate results','',
'| Seed | Site | Method | Natural NMSE | Target NMSE | No-intervention NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
for seed in (893,894):
 s=json.loads((root/f'runs/dependence_numeric_seed{seed}/summary.json').read_text())
 for site,methods in s['sites'].items():
  for method,row in methods.items():
   ns=row['numeric_nodes'];nat=sum(x['natural_n']*x['natural_nmse'] for x in ns)/sum(x['natural_n'] for x in ns)
   def agg(part,key):
    count=sum(x[part]['n'] for x in ns);return sum(x[part]['n']*x[part][key] for x in ns if x[part]['n'])/count
   b=row['behavioral'];acc='N/A' if b['informative_accuracy'] is None else f"{b['informative_accuracy']:.1%}"
   lines.append(f"| {seed} | {site} | {method} | {nat:.3f} | {agg('targeted','nmse'):.3f} | {agg('targeted','no_intervention_nmse'):.3f} | {agg('collateral','nmse'):.3f} | {b['informative_pairs']} | {acc} |")
lines+=['','## Numeric mapping by dependence step','',
'| Seed | Site | Feature | Step | Natural NMSE | Target NMSE | No-intervention NMSE |','|---:|---|---|---|---:|---:|---:|']
features={'7':'dep_xy','8':'resdep_xy','9':'resdep_yx'}
for seed in (893,894):
 s=json.loads((root/f'runs/dependence_numeric_seed{seed}/summary.json').read_text());labels=[]
 for g in s['groups']:
  member=g['occurrences'][0].split('/');labels.append((features[member[1]],g['ast']['role']))
 for site in s['sites']:
  for (feature,role),row in zip(labels,s['sites'][site]['numeric']['numeric_nodes']):
   lines.append(f"| {seed} | {site} | {feature} | {role} | {row['natural_nmse']:.3f} | {row['targeted']['nmse']:.3f} | {row['targeted']['no_intervention_nmse']:.3f} |")
lines+=['','## Interpretation','',
'- Natural readout is strong: pooled NMSE is 0.20–0.27 across cuts and seeds. The hidden state contains substantial information about kernel bandwidths, energies, numerators and denominators.',
'- Interchange is much weaker. Numeric mappings have pooled target NMSE 1.54–2.31. Only head-tanh in seed 893 and head-linear/head-tanh in seed 894 modestly improve over the no-intervention baseline.',
'- Behavior-only mappings destroy numerical semantics by one to three orders of magnitude. This confirms that matching class changes does not identify the underlying numerical computation.',
'- Informative categorical samples are scarce (20 and 24), and intervention accuracy ranges from 0% to 35%. No claim of causal abstraction equivalence is supported.',
'- The experiment exposes dependence internals but cross-fit regression remains opaque. The complete raw causal-discovery algorithm is still not recovered.',
'- Seeds 893/894 have been observed and cannot be reused as untouched confirmation sets.']
(root/'RESULTS_DEPENDENCE_NUMERIC.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
accept={'state':'verified','runs':{str(seed):{'worlds':5120,'groups':18,'sites':['representation','head_linear','head_tanh'],'train_pairs':384,'test_pairs':1024} for seed in (893,894)},
 'verification':'independent regeneration of worlds, dependence traces, causal pairs, probes, learned bases, controls and all metrics; both processes exited 0','science_not_certified':True}
(root/'validation/dependence_numeric_acceptance.json').write_text(json.dumps(accept,indent=2),encoding='utf-8')
p=root/'WORK_STATUS.md';s=p.read_text(encoding='utf-8');s+='''
## 依赖核内部机制
- RawDiscoveryExecutor 的 `trace_dependence=True` 版本化开关展开三个 dependence occurrence，每个包含两带宽、两核能量、numerator、denominator，共 18 组；默认关闭，793/794 的程序 ID 保持逐字一致。
- 配对引擎识别 bandwidth→energy/numerator/denominator 与 energy→denominator 的因果关系，排除祖先/后代联合干预。
- quick 892：512 worlds、18 groups、48/96 pairs，完整重放通过。
- 正式 893/894：每组 5,120 worlds、384/1,024 pairs、129 个未见兼容双组 mask；全部映射重新训练并独立重放通过（validation/dependence_numeric_acceptance.json）。
- 自然读出 NMSE 约 0.20–0.27，但干预 target NMSE 1.54–2.31，行为有效样本仅 20/24；可读出信息没有形成稳定因果交换。详见 RESULTS_DEPENDENCE_NUMERIC.md。
- regression 仍是黑箱；当前源码已超过 0.4.0 wheel，后续需新版本交付。
''';p.write_text(s,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';s=p.read_text(encoding='utf-8');s=s.replace('variance-ratio 失败且 dependence/regression 内部仍不透明（RESULTS_RAW_NUMERIC.md）','variance-ratio 失败；893/894 已展开 dependence 核但干预恢复弱，regression 内部仍不透明（RESULTS_RAW_NUMERIC.md、RESULTS_DEPENDENCE_NUMERIC.md）');p.write_text(s,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8');s+='\nDependence-kernel internal intervention results: [RESULTS_DEPENDENCE_NUMERIC.md](RESULTS_DEPENDENCE_NUMERIC.md).\n';p.write_text(s,encoding='utf-8')
print('dependence report and acceptance written')
