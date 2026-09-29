from pathlib import Path
import json
root=Path('.');s=json.loads((root/'runs/relational_program_693_694/summary.json').read_text(encoding='utf-8'))
lines=['# Relational context in extracted graph programs','',
'The protocol was frozen in `docs/RELATIONAL_PROGRAM_PROTOCOL.md` before seeds 693 and 694 were generated. Six programs and all reported metrics were independently replayed; see `validation/relational_program_replay.json`.','',
'## Primary paired exact-graph fidelity','',
'| Seed | Frozen teacher | Local composed | Relational primitive | Difference | Conservative 95% interval |','|---:|---|---:|---:|---:|---|']
for x in s['primary']:
 lines.append(f"| {x['seed']} | {x['mode']} | {x['local_composed']:.1%} | {x['relational_primitive']:.1%} | {x['paired_difference']:+.1%} | [{x['hoeffding95_interval'][0]:+.1%}, {x['hoeffding95_interval'][1]:+.1%}] |")
lines+=['','Each comparison contains 960 independently generated world units in the fixed mixture of three node sizes and five environments. Intervals treat the paired per-world difference as bounded in [-1, 1]; they do not correct across the four reported comparisons.','',
'## Pooled behavioral metrics','',
'| Seed | Teacher | Program | All-pair fidelity | Active-pair fidelity | Exact-graph fidelity | Program truth accuracy | Complexity |','|---:|---|---|---:|---:|---:|---:|---:|']
for seed in s['config']['seeds']:
 for mode in ('without_relations','with_relations'):
  for variant in s['config']['variants']:
   rows=[]
   for key,value in s['evaluation'][str(seed)].items():rows.append(value[mode][variant])
   total=sum(r['worlds'] for r in rows)
   agg=lambda k:sum(r['worlds']*r[k] for r in rows)/total
   complexity=json.loads((root/f'runs/relational_program_693_694/programs/{mode}/{variant}.json').read_text())['complexity']
   lines.append(f"| {seed} | {mode} | {variant} | {agg('all_pair_fidelity'):.1%} | {agg('active_pair_fidelity'):.1%} | {agg('exact_graph_fidelity'):.1%} | {agg('program_exact_graph_accuracy'):.1%} | {complexity} |")
lines+=['','## Interpretation','',
'- The relational program for the relation-enabled teacher selected contextual primitives, so the added language was available to synthesis. Its exact-graph fidelity increased only 0.5 and 0.9 percentage points across the two seeds; both conservative intervals include zero.',
'- For the teacher without relation biases, relational primitives reduced exact-graph fidelity by 2.8 and 3.3 percentage points. This is a useful negative control: a larger language can hurt finite-budget MDL search.',
'- These results do not explain the attention computation fully. Simple incidence means are too weak, or the six-split tree and candidate pruning cannot exploit them.',
'- Program truth accuracy remains a separate metric. Improving agreement with a frozen teacher does not imply better causal graph recovery.',
'- Seeds 693/694 have now been examined and cannot be reused as untouched confirmation data for a revised method.']
(root/'RESULTS_RELATIONAL_PROGRAM.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=root/'WORK_STATUS.md';t=p.read_text(encoding='utf-8');t+='''
## 加权搜索与关系程序实验
- `fit_rule` 支持正的逐样本权重；默认行为与冻结 0.3.0 实现逐项一致。67 项测试在加入关系原语前通过。
- runs/weighted_rules_seed593 比较均匀、世界等权、世界与教师类别等权。六个程序和 180 个分层已独立重放（validation/weighted_rules_replay.json）；类别再平衡使两组整图保真下降，不作为改进。
- ncd/graph_program_features.py 加入同源、同目标、后继、前驱的显式均值原语，保持节点置换等变。
- 按 docs/RELATIONAL_PROGRAM_PROTOCOL.md 预先冻结的 693/694 实验已完成并独立重放（validation/relational_program_replay.json）。关系教师整图保真仅提高 +0.5%/+0.9%，区间跨零；无关系教师下降 -2.8%/-3.3%。详见 RESULTS_RELATIONAL_PROGRAM.md。
- 简单关系均值没有解决程序保真；693/694 已用于观察结果，后续修改不得把它们作为未触碰确认集。
- 当前源码已超过冻结的 0.3.0 wheel；其安装验收仍是历史版本证据，下一次完整交付需构建新版本。
''';p.write_text(t,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';t=p.read_text(encoding='utf-8');t=t.replace('行为 fidelity 明显不足，未完整反编译 |','行为 fidelity 明显不足；加权搜索和显式关系均值 593/693/694 已独立重放但无稳定改善（RESULTS_RELATIONAL_PROGRAM.md），未完整反编译 |');t=t.replace('整图恢复有明显失败；显式关系注意力 493/494 已独立重放，整图改善不稳定（RESULTS_RELATIONAL.md） |','整图恢复有明显失败；显式关系注意力 493/494 改善不稳定，关系程序 693/694 也仅有微小且不确定的保真变化（RESULTS_RELATIONAL.md、RESULTS_RELATIONAL_PROGRAM.md） |');p.write_text(t,encoding='utf-8')
p=root/'README.md';t=p.read_text(encoding='utf-8');t+='\nRelational program extraction results: [RESULTS_RELATIONAL_PROGRAM.md](RESULTS_RELATIONAL_PROGRAM.md).\n';p.write_text(t,encoding='utf-8')
print('report and requirement records updated')
