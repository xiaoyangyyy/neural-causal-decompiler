# 原始目标完整验收清单

原始范围以 ORIGINAL_PROPOSAL.txt 为准。实现模块、可重放证据与科学结论分开判定；下列“已实现”不表示全部目标已经达到。

正式证据：runs/all_seed91_final 与 runs/all_seed92_final。二者均已独立 verify-all 通过，各包含 6,928 个双变量世界、936 个多变量世界、640 次反例查询重放及 12 次机制恢复。33 项测试通过。最终打包及汇总验收由 scripts/acceptance_v2.py 执行，结果写入 validation/acceptance_v2.json。

| 编号 | 原始目标 | 当前直接证据 | 尚未达到或限制 |
|---|---|---|---|
| R0 | 对象、可识别性、fidelity/equivalence 定义 | docs/THEORY.md；独立单位保守界 | 无唯一反编译或通用因果识别定理 |
| R1 | SCM、方程 AST、噪声、观察与干预 | 六族双变量及 3/5/8 节点世界；数据逐项重放 | 仅定义的无混杂无环基准 |
| R2 | 样本不变、变量等变神经发现 | 实际训练 checkpoint；置换测试与独立评估 | 多变量有显式统计前端；准确率有限 |
| R3 | 组合式 CDIR | cdir.py；typed AST、回归/残差/依赖、算术与布尔 | 搜索使用有限算子及深度，非任意程序 |
| R4 | 神经到短程序、MDL 与 CEGIS | rules.py、guided_synthesis.py；自动组合、复杂度项、主动修正；393/394 多位置内部评分重排和五环境独立验收（RESULTS_GUIDED.md） | 行为 fidelity 明显不足；加权搜索和显式关系均值 593/693/694 已独立重放但无稳定改善（RESULTS_RELATIONAL_PROGRAM.md），未完整反编译 |
| R5 | circuit–rule 内部对齐 | distributed_alignment.py、joint_alignment.py、joint_experiment.py；数据隔离、随机/置换/打乱/独立单变量控制与多层追踪 | V2 为单变量；新 191/192 实验联合六个位置并重放通过，但有效覆盖不足，表达式/判断干预目标等价，未恢复完整数值算法（RESULTS_JOINT.md）；793/794 已把 mean/variance/std/correlation/variance-ratio 展开到原始样本并干预，但 variance-ratio 失败；893/894 已展开 dependence 核但干预恢复弱；993/994 已展开 cross-fit regression 且自然可读但干预恢复失败（RESULTS_RAW_NUMERIC.md、RESULTS_DEPENDENCE_NUMERIC.md）；293/294 数值审计已重放，393/394 内部位置选择已接回候选重排，但尚未带来选择变化（RESULTS_NUMERIC.md、RESULTS_GUIDED.md） |
| R6 | 真值/网络/程序比较 | 五环境指标、置信界、图与方程指标、JSON/HTML | 正确率与保真均为有限基准结果 |
| R7 | 主动 SCM 参数优化 | 保持结构的参数变异；640 次查询重放/种子；重复采样复验 | 没有全局最强反例保证 |
| R8 | 函数、噪声、尺度、干预 OOD | 两个固定种子各五类独立测试 | 不支持普遍不变因果原则 |
| R9 | 多变量 3/5/8、条件变量、局部程序、无环 | graph_model.py、graphs.py；CPDAG 穷举至四节点；PC 与分离集；真实图提取 | sepset/collider 是显式实现的语义/基线，不是已从网络自动逆出的算法；整图恢复有明显失败；显式关系注意力 493/494 改善不稳定，关系程序 693/694 也仅有微小且不确定的保真变化（RESULTS_RELATIONAL.md、RESULTS_RELATIONAL_PROGRAM.md） |
| R10 | 神经机制到显式 SCM | 冻结 MLP 输出监督的符号拟合；两种图来源分列；24 次正式恢复 | 父集/算子/干预仍有误差，真实图诊断不能替代端到端恢复 |
| R11 | 完整消融 | full、无对齐/MDL/CEGIS/环境扩增、tree、真实 PySR；neural 单独评估 | 不预设 full 优于所有方法，以实际结果为准 |
| R12 | 等价类与 canonical AST | 受限代数精确化简；有限探针经验等价独立标识 | 非一般程序语义等价证明，未证明最短程序唯一 |
| R13 | 扩展研究工程 | 统一 CLI、报告、源快照、哈希/语义重放、wheel 实际隔离运行 | 历史聚合验收见 acceptance_v2.json；0.3.0 已接入 joint/numeric/guided/relational 及独立重放入口，四类新增实验均已在隔离安装包实际运行并重放（validation/wheel_v3_run/status.json），原有 run-all/verify-all 也已从 0.3.0 包实际运行通过（validation/wheel_v3_full/status.json）；0.4.0 原始统计干预及 0.5.0 dependence/regression 内部干预均从隔离安装包运行/重放（validation/wheel_v4_run/status.json、validation/wheel_v5_run/status.json）；CNN/robot 仅通用追踪接口，未定义外部任务 |

## 完成审计结论

工程链路已经可以运行，但原始完整目标尚未完成。特别是 R4、R5、R8、R9、R10 的强科学目标不能用单元测试或工件存在代替。不能把局部干预一致、低机制数值误差或人工内置统计规则当作完整内部算法恢复。

下一阶段应先补齐多步骤、多个中间变量的内部机制对齐和独立验证，再改善短程序保真与端到端图/机制恢复。新方法必须另设预先固定的训练/选择/测试种子，保留 91/92 作为历史结果，不根据这些测试反复挑选最佳方法。

CNN/robot policy 在原提案中是架构扩展愿景，当前通用 PyTorch 调用追踪与干预接口提供接口基础，不声称完成未定义的机器人实机任务。


## Version 0.6 evidence update

R5 now includes a full-trace biorthogonal intervention comparison over 54
raw-statistic, dependence-kernel, and cross-fit regression groups. Replacement
confirmation seeds 1193/1194 each contain 9,216 disjoint worlds, 384 fit pairs,
2,048 test pairs, all 1,411 compatible masks, three neural cuts, an orthogonal
comparator, and behavior-only/shuffled/random controls. Both runs independently
replayed and passed the preregistered relative-improvement rule at every cut;
see RESULTS_OBLIQUE_NUMERIC.md and
validation/oblique_numeric_acceptance.json.

This strengthens R5 by showing that separate dual read/write directions improve
held-out numerical interchange. It does not close R5: target NMSE is 2.40-4.30
and collateral NMSE is 12.89-17.80. Exact numerical mechanism recovery,
uniqueness, and universal equivalence remain unproved. Requirements R4, R8, R9,
and R10 also remain incomplete.


## Version 0.7 evidence update

R4 now has a candidate-independent path from 14 feature-root interventions into
beam-search costs and final candidate selection. Formal seeds 1393/1394 each
used 10,240 isolated worlds and independently replayed. Both changed the
selected program, but only 1393 improved final fidelity; 1394 degraded all five
environments. The preregistered replication rule failed, so R4 remains
incomplete. Evidence: RESULTS_CAUSAL_GUIDED.md and
validation/causal_guided_acceptance.json.


## Version 0.8 evidence update

R4 now includes an explicit validation-noninferiority gate for causal program
reranking. Seeds 1493/1494 independently replayed, and every selected program
satisfied the gate. Final OOD safety did not replicate: pooled fidelity changed
by -0.21 percentage points and environment-level decreases remained.
RESULTS_CONSERVATIVE_GUIDED.md and
validation/conservative_guided_acceptance.json record the negative result.
R4 remains incomplete; selection-time synthetic OOD strata are the next
required step.


## Version 0.9 evidence update

R4 now includes five selection-only OOD guard strata and requires candidate
noninferiority on ordinary validation plus every guard. Formal seeds 1593/1594
each used 12,800 disjoint worlds and passed full regeneration replay. The
historical program was the only eligible candidate in both runs; the selector
therefore abstained and every final-environment fidelity delta was zero.

This removes the observed OOD regressions from the 0.8 confirmation runs, but
it does not demonstrate improved decompilation: neither run changed program,
and the preregistered replicated-safe-improvement rule failed. R4 remains
incomplete. Evidence: `RESULTS_OOD_GUIDED.md` and
`validation/ood_guided_acceptance.json`.
R13 release evidence for 0.9.0 is
`validation/wheel_v9_run/status.json`: source, wheel, and isolated-installed
modules matched, and a fresh quick OOD-guarded run plus regeneration replay
completed. The full regression suite contains 92 passing tests.

## Version 0.10 evidence update

R10 now includes a hierarchy-constrained symbolic mechanism search evaluated on
60 preregistered multivariate worlds across two seeds, three graph sizes, and
five environments. Both formal runs replayed. Pooled frozen-neural NMSE fell by
94.6%, equations became shorter, and the frozen rule's truth/intervention
safeguards passed. Evidence: `RESULTS_STRUCTURED_MECHANISM.md` and
`validation/structured_mechanism_acceptance.json`.

The experiment supplies the oracle DAG to isolate equation extraction. It does
not close R9, and R10 remains incomplete end to end because inferred parent
sets and graph errors are excluded from the primary comparison. General
identifiability and exact recovery also remain unproved.
R13 release evidence for 0.10.0 is
`validation/wheel_v10_run/status.json`: source, wheel, and isolated-installed
modules matched, and a fresh quick structured-mechanism run plus replay passed.
The full regression suite contains 94 passing tests.

## Version 0.11 evidence update

R9 now includes a development-calibrated probability decoder and deterministic
replay over 960 new worlds. Pair SHD and skeleton metrics improved, but pooled
exact-graph accuracy gained only 1.30 percentage points and direction accuracy
decreased. The frozen confirmation rule failed, so R9 remains incomplete.
Evidence: `RESULTS_CALIBRATED_GRAPH_DECODER.md` and
`validation/calibrated_graph_decoder_acceptance.json`.

R13 release evidence for 0.11.0 is alidation/wheel_v11_run/status.json; source, wheel, and isolated-installed modules matched, and 95 regression tests passed.


## Version 0.12 evidence update

R9 now has an executable PC-style program with persisted CI, sepset, skeleton,
collider, and Meek traces. Seeds 2193/2194 each used 480 new worlds and passed
complete replay. The program was slightly more accurate against truth but much
less faithful to frozen teachers than the existing local tree; the fixed rule
failed. This rejects the tested Fisher-PC program as a behavioral explanation
of these networks and leaves R9 incomplete. Evidence:
`RESULTS_DISTILLED_PC.md` and `validation/distilled_pc_acceptance.json`.

R13 release evidence for 0.12.0 is alidation/wheel_v12_run/status.json; source, wheel, and isolated-installed modules matched, and 96 regression tests passed.

## Version 0.13 evidence update

R9 now includes an executable factorized graph program with separate skeleton and orientation rules, bidirectional consistency, persisted synthesis traces, and deterministic acyclic projection. Seeds 2393/2394 each used 480 new worlds and passed complete replay. Pooled exact teacher fidelity improved by only 0.52 percentage points, below the frozen +2 point requirement, and one mode declined in each seed. The confirmation rule failed, so R9 remains incomplete. Evidence: `RESULTS_FACTORIZED_GRAPH_PROGRAM.md` and `validation/factorized_graph_acceptance.json`.
R13 release evidence for 0.13.0 is `validation/wheel_v13_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
factorized-graph run and complete replay passed; 98 regression tests passed.

## Version 0.14 evidence update

R9 now includes source-selected positive-class costs for the factorized skeleton program. Seeds 2593/2594 each used 480 new worlds and passed candidate-refitting replay. Pooled exact teacher fidelity improved by 0.57 percentage points, below the frozen +2 point requirement; seed 2594 selected the unweighted baseline in both modes. The replication rule failed and pooled truth accuracy declined, so R9 remains incomplete. Evidence: `RESULTS_COST_SENSITIVE_GRAPH.md` and `validation/cost_sensitive_graph_acceptance.json`.
R13 release evidence for 0.14.0 is `validation/wheel_v14_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
cost-sensitive run and complete replay passed; 98 regression tests passed.

## Version 0.15 evidence update

R9 now includes unordered, swap-invariant skeleton supervision taken from the frozen teacher's decoded graph. Seeds 2793/2794 each used 480 new worlds and passed complete program-refitting replay. Pooled active-pair fidelity improved by 0.72 percentage points, while exact teacher fidelity declined by 0.05 points and truth accuracy declined. The rule failed; local pairwise tree changes have not produced stable exact-graph gains. R9 remains incomplete. Evidence: `RESULTS_SYMMETRIC_SKELETON.md` and `validation/symmetric_skeleton_acceptance.json`.
R13 release evidence for 0.15.0 is `validation/wheel_v15_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
symmetric-skeleton run and complete replay passed; 100 regression tests passed.

## Version 0.16 evidence update

R9 now includes an explicit graph-global sparse ranking program with a predicted world-specific edge budget. Seeds 2993/2994 each used 480 new worlds and passed complete candidate-refitting replay. Pooled exact teacher fidelity declined by 2.29 percentage points, truth accuracy declined by 3.54 points, and the complexity gate failed. Joint top-k selection amplified ranking/count errors and is rejected under the frozen rule. R9 remains incomplete. Evidence: `RESULTS_GRAPH_GLOBAL_RANKING.md` and `validation/graph_global_acceptance.json`.
R13 release evidence for 0.16.0 is `validation/wheel_v16_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick graph-global
run and complete replay passed; 102 regression tests passed.

## Version 0.17 evidence update

R5 now includes paired linear/diagonal-quadratic readout interventions over all 54 traced variables. Seeds 3193/3194 used 9,216 worlds each, covered all 1,411 compatible masks, and passed complete retraining replay. Quadratic probes improved natural-state NMSE at five of six seed/site cases but worsened targeted and collateral intervention NMSE at every case; one shuffled-control gate also failed. Flexible off-manifold probe behavior is rejected as a remedy for the intervention error floor. R5 remains incomplete. Evidence: `RESULTS_QUADRATIC_READOUT.md` and `validation/quadratic_readout_acceptance.json`.
R13 release evidence for 0.17.0 is `validation/wheel_v17_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick quadratic-readout
run and complete replay passed; 103 regression tests passed.


## Version 0.18 evidence update

R5 now includes local natural-manifold constrained writes over all 54 traced
variables. Seeds 3393/3394 each used 9,216 worlds, covered every compatible
single/two-variable mask, and passed complete regeneration and retraining
replay. Weighted targeted NMSE improved at all six seed/site cases and strongly
beat equal-rank random subspaces. The frozen rule still failed because seed
3393 at `head_tanh` increased collateral NMSE from 12.5107 to 12.5148;
behavioral interchange was also mixed. This supports the off-manifold failure
hypothesis but does not complete R5. Evidence:
`RESULTS_MANIFOLD_INTERVENTION.md` and
`validation/manifold_intervention_acceptance.json`.
R13 release evidence for 0.18.0 is `validation/wheel_v18_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
manifold-intervention run and complete replay passed; 104 regression tests
passed.

## Version 0.19 evidence update

R5 now tests whether inactive linear-readout coordinate preservation repairs
the remaining collateral error of PCA-tangent writes. Seeds 3593/3594 each
covered all 54 groups, all 1,411 compatible masks, and passed complete
regeneration and refitting replay. Inactive coordinate RMS fell at all six
seed/site cases and targeted NMSE remained better than ordinary biorthogonal
writes, but collateral NMSE improved only twice and worsened at all sites in
seed 3594. Linear probe-coordinate leakage is rejected as a sufficient proxy
for downstream collateral fidelity; R5 remains incomplete. Evidence:
`RESULTS_COLLATERAL_INTERVENTION.md` and
`validation/collateral_intervention_acceptance.json`.
R13 release evidence for 0.19.0 is `validation/wheel_v19_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
collateral-intervention run and complete replay passed; 105 regression tests
passed.

## Version 0.20 evidence update

R10 now has a paired end-to-end path from frozen neural graph predictions
through deterministic DAG completion, neural conditional mechanisms, and
structured explicit SCM equations. Seeds 3793/3794 each covered 15 new worlds
across 3/5/8 nodes and five environments and passed graph re-inference, neural
retraining, symbolic refitting, and metric replay. Structured equations were
shorter, improved pooled inferred-graph neural NMSE by 18.7%, and reduced
intervention error, but seed 3794 did not replicate neural-fidelity improvement.
Inferred structured truth NMSE and intervention MAE were 2.32 and 3.11 times
the oracle-DAG branch. The frozen rule failed; R10 remains incomplete and the
paired evidence identifies R9 graph recovery as the dominant end-to-end gap.
Evidence: `RESULTS_END_TO_END_MECHANISM.md` and
`validation/end_to_end_mechanism_acceptance.json`.
R13 release evidence for 0.20.0 is `validation/wheel_v20_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
end-to-end-mechanism run and complete replay passed; 105 regression tests
passed.

## Version 0.21 evidence update

R9 now includes a stronger permutation-equivariant graph teacher that explicitly
aggregates incoming/outgoing node context and global graph context before
shared edge attention. Seeds 3993/3994 each used 2,400 worlds with paired
initialization and batch order and passed full checkpoint retraining replay.
Exact-graph accuracy improved in both seeds and by 1.15 pooled percentage
points, with larger gains in ID and noise environments, but missed the frozen
+3-point requirement. SHD worsened in one seed and pooled skeleton/direction
metrics declined. The architecture is retained as a limited component, not a
solution; R9 remains incomplete. Evidence: `RESULTS_NODE_CONTEXT_GRAPH.md`
and `validation/node_context_graph_acceptance.json`.
R13 release evidence for 0.21.0 is `validation/wheel_v21_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
node-context-graph run and complete replay passed; 106 regression tests passed.

## Version 0.22 evidence update

R9 now includes equal-parameter flat and factorized heads on the node-context
teacher. Seeds 4193/4194 each used 2,400 shared worlds and passed full
retraining and zero-tolerance checkpoint replay. Factorization increased exact
graph accuracy in both seeds, lowered SHD in both, improved pooled skeleton F1,
and produced positive exact-graph deltas in all five environments. The pooled
exact gain was 1.35 points, below the frozen 2-point threshold; seed 4194
skeleton F1 slightly declined and pooled directed accuracy fell. R9 remains
incomplete, with orientation now isolated as the main residual graph-head
failure. Evidence: `RESULTS_FACTORIZED_NODE_CONTEXT.md` and
`validation/factorized_node_context_acceptance.json`.

R13 release evidence for 0.22.0 is `validation/wheel_v22_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
factorized-node-context run and complete replay passed; 107 regression tests passed.


## Version 0.23 evidence update

R9 now includes an exact node-swap orientation constraint on the factorized
node-context teacher. Seeds 4393/4394 each used 2,400 worlds with shared
compatible initialization and passed full retraining and zero-tolerance
checkpoint replay. The constraint improved exact accuracy, direction accuracy,
skeleton F1, and SHD in seed 4393, but all four regressed in seed 4394. Pooled
exact accuracy increased only 0.38 percentage points and directed accuracy was
effectively unchanged. Hard antisymmetry is rejected as a stable standalone
orientation remedy; R9 remains incomplete. Evidence:
`RESULTS_ANTISYMMETRIC_DIRECTION.md` and
`validation/antisymmetric_direction_acceptance.json`.

R13 release evidence for 0.23.0 is `validation/wheel_v23_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
antisymmetric-direction run and complete replay passed; 108 regression tests passed.

## Version 0.24 evidence update

R9 now tests whether aligning the factorized component loss with pair-symmetric
inference repairs orientation. Seeds 4593/4594 each used 2,400 worlds; both
branches had identical architecture, parameter count, initialization, and
batch order, and both passed full retraining and zero-tolerance checkpoint
replay. Pair-consistent training improved skeleton F1 and SHD only in seed 4594
and reduced directed-target accuracy in both seeds. Pooled exact accuracy fell
0.66 points, directed accuracy fell 2.11 points, and function-shift exact
accuracy fell 2.43 points. Training/inference symmetry mismatch is rejected as
the main direction bottleneck; R9 remains incomplete. Evidence:
`RESULTS_PAIR_CONSISTENT_TRAINING.md` and
`validation/pair_consistent_acceptance.json`.

R13 release evidence for 0.24.0 is `validation/wheel_v24_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick
pair-consistent run and complete replay passed; 109 regression tests passed.

## Version 0.24 evidence update

R9 now tests whether aligning the factorized component loss with inference-time
pair symmetrization repairs direction errors. Seeds 4593/4594 used identical
31,636-parameter models, initial weights, batch orders, and 2,400 worlds each,
and passed complete retraining and zero-tolerance checkpoint replay. The
candidate improved exact accuracy, skeleton F1, and SHD only in seed 4594;
directed accuracy declined in both seeds. Pooled exact and directed accuracy
fell by 0.66 and 2.11 percentage points, and function-shift exact accuracy fell
by 2.43 points. Pair-consistent loss replacement is rejected; R9 remains
incomplete. Evidence: `RESULTS_PAIR_CONSISTENT_TRAINING.md` and
`validation/pair_consistent_acceptance.json`.

The current rebuilt 0.24.0 distribution is additionally verified by
`validation/wheel_v24_rebuild_run/status.json` (SHA-256
`98f4696a4477b6e26b773b72065098d4c87691a83c3192b55d4e3b2afdf0a3fe`).


## Version 0.25 evidence update

R9 now has a strong active-intervention branch. For every world and source node,
two interventions at the observational mean plus/minus one standard deviation
supply ordered downstream-response summaries without reading graph or equation
metadata. Seeds 4793/4794 each used 2,400 worlds and 25,600 intervention
datasets; both passed complete data/feature regeneration, retraining, and
zero-tolerance checkpoint replay. Pooled exact-graph accuracy improved from
17.33% to 50.66%, directed accuracy from 48.69% to 87.27%, skeleton F1 from
0.7854 to 0.9196, and SHD from 2.8451 to 1.2128. All frozen criteria passed.
This establishes a strong interventional result but does not complete purely
observational graph recovery or historical-teacher decompilation, so broad R9
remains incomplete. Evidence: `RESULTS_ACTIVE_INTERVENTION_GRAPH.md` and
`validation/active_intervention_acceptance.json`.

R13 release evidence for 0.25.0 is `validation/wheel_v25_run/status.json`:
source, wheel, and isolated-installed modules matched; a fresh quick active
intervention graph run and complete replay passed; 110 regression tests passed.

## Version 0.26 evidence update

R10 now connects the version 0.25 paired observational and active-intervention
graph teachers to conditional neural mechanisms and hierarchy-constrained
symbolic SCMs, with a paired oracle-DAG diagnostic. Seeds 4993/4994 covered 15
new worlds each across 3/5/8 nodes and five environments and passed complete
world/feature regeneration, graph re-inference, neural retraining, symbolic
refitting, and metric replay. Active graph SHD improved in both seeds and fell
from 4.2333 to 2.6333 pooled. Structured truth NMSE and intervention-effect MAE
improved in both seeds by 15.29% and 17.80% pooled, but exact graph accuracy was
unchanged pooled and declined in seed 4994; the improvements missed their
frozen 20% margins. Active structured truth and intervention errors remained
2.53 and 3.00 times the oracle-DAG diagnostic, and neural fidelity did not
improve consistently. The frozen rule failed. This is partial active
intervention evidence; broad R9 and R10 remain incomplete. Evidence:
`RESULTS_ACTIVE_END_TO_END.md` and
`validation/active_end_to_end_acceptance.json`.
R13 release evidence for 0.26.0 is
`validation/wheel_v26_run/status.json`: source, wheel, and
isolated-installed modules matched; a fresh quick active-end-to-end run and
complete graph, mechanism, and symbolic replay passed; 111 regression tests
passed. Wheel SHA-256:
`59e66d6f1acd760e009d951484f595f6c19606ac00c79611f52d84a652c47e65`.
## Version 0.27 certified finite realization

The project now has a separate primary path for exact finite interventional
realization. It defines a finite deterministic Moore system over a complete
action-closed internal-state list, computes its exact behavioral quotient,
recovers the same quotient using response queries only, and emits independently
checkable upper and lower minimality certificates. Budget-limited runs preserve
unresolved pairs. Twelve formal cases (90 concrete states, 54 minimal states)
all recovered exactly. Three finite L-infinity tolerances per case also closed
their certified lower/upper intervals, giving aggregate complexity
90 -> 54 -> 12. All 60 exact and approximate certificates passed complete
regeneration and semantic replay. Evidence: 'RESULTS_CERTIFIED_FINITE.md' and
'runs/certified_finite_seed2701'.

R13 release evidence is 'validation/wheel_v27_run/status.json': every packaged
module matched source and isolated installation; fresh finite, quantized-traffic,
and continuous-separation runs and semantic replays passed; all 124 regression
tests passed. Wheel SHA-256:
'33fd202c630b686181d4a5386e4bfe964fa400229797d6cf737f67b663a3b548'.

This closes exact and finite L-infinity approximate phases under their declared
assumptions. Pairwise compatibility is not used as an upper certificate. A
sound finite-horizon continuous ReLU separation oracle now returns certified
lower/upper bounds or unresolved, and verified pair certificates produce an
exact chromatic minimality lower bound for a declared finite state collection.
At epsilon 0.3, a piecewise-action two-state candidate passes a uniform horizon-
one upper proof and matches the chromatic lower bound, closing `K=2` for the
frozen finite initial-state collection and continuous intervention interval.
Evidence is in 'RESULTS_CERTIFIED_CONTINUOUS.md' and
'runs/certified_continuous'. A uniform candidate-realization upper certificate
over an uncountable initial-state domain remains open. The previous
R4/R5/R8/R9/R10 claims remain unchanged and incomplete.

## Version 0.28 continuous certification scaling

The continuous verifier now preserves shared intervention dependencies whenever
both neural executions have certifiably matching stable ReLU phases. Four formal
profiles cover latent dimensions 8/32/64/128, action dimensions 2/3/4/5, and
horizons 3/5/10/20. The largest fitted network has 52,002 parameters. Relational
bounds certified all four near pairs within tolerance and all four far pairs as
separated; independent IBP left every near pair unresolved under its declared
eight-leaf budget. All 16 certificates and all model fitting were regenerated
successfully. Evidence: `RESULTS_CERTIFIED_CONTINUOUS_SCALE.md` and
`validation/certified_continuous_scale_acceptance.json`.

Release evidence is `validation/wheel_v28_run/status.json`: source, wheel, and
isolated installation matched byte-for-byte; all eight finite, traffic,
continuous, and scaling generation/replay commands passed; 127 regression tests
passed. Wheel SHA-256:
`18343a2ead65601d11a7c2286a8083e4734e843cf0a95321c3008dc702580143`.

This result is restricted to learned stable affine dynamics represented by ReLU
networks. Scaling across widespread activation-boundary crossings, continuous
initial-state regions, and nonlinear learned traffic dynamics remains open.

## Post-0.26 calibrated decoder evidence

R9 now includes a frozen decoder-only diagnostic on the version 0.25 active
teacher probabilities. A development-selected 0.55 edge-presence threshold
improved exact accuracy, SHD, and skeleton F1 in both seeds and reduced pooled
SHD from 1.2128 to 1.1774. Pooled exact accuracy improved only 0.382 percentage
points, below the frozen 0.5-point gate, and exact accuracy declined under
function and noise shifts. The rule failed. Global calibration is a secondary
bottleneck; broad R9 remains incomplete. Evidence:
`RESULTS_CALIBRATED_ACTIVE_DECODER.md` and
`validation/calibrated_active_decoder_acceptance.json`.

## Version 0.29 phase-crossing continuous evidence

The continuous branch-and-bound verifier now supports best-bound coordinate
selection and sound per-leaf hybrid propagation. Relational difference bounds
are used only when both executions have the same certified ReLU phase; every
other leaf falls back to outward-rounded independent IBP. The formal seed-5701
run covers 16/32/64 state dimensions, 2/2/3 continuous controls, and horizons
1/2/3. The declared domains cross three activation boundaries, all model fits
have maximum error below 2.11e-12, and independent replay verified all 24
certificates.

All far pairs are separated. The 16D one-step near pair closes under every
ablation; hybrid/widest needs 46 leaves versus 70 for independent/widest. The
32D two-step near pair closes under widest splitting but not the greedy
best-bound strategy at 64 leaves. The 64D three-step near pair remains
unresolved at 24 leaves, with best upper bound 0.243001 against threshold 0.24.
This establishes sound certification through genuine control-dependent phase
changes for selected finite-horizon fixed-state claims. It does not close a
continuous initial-region realization, global minimality, or natural traffic
dynamics. Evidence: RESULTS_CERTIFIED_CONTINUOUS_NONLINEAR.md,
docs/CONTINUOUS_NONLINEAR_METHOD.md, and
validation/certified_continuous_nonlinear_acceptance.json.
R13 release evidence for 0.29.0 is validation/wheel_v29_run/status.json:
source, wheel, and isolated-installed modules matched; all ten certified
generation/replay commands passed; 130 regression tests passed. Wheel SHA-256:
6a097cafe7cdb96734a9c2e67f2c2dd5cf92847de669e96bff3b872afa1c721a.
## Post-0.29 nonlinear budget-closure evidence

A targeted follow-up changed only the 64D/H=3 leaf budget from 24 to 28. The
best-bound partition closes a verified near-pair upper certificate at
0.239735521 against threshold 0.24. Widest splitting remains unresolved at
0.261212231 under the same budget. All 28 near leaves use independent IBP, so
the result establishes a partition-and-budget closure rather than a relational
bound gain. It closes the fixed-state, finite-horizon proof gap but does not
address continuous initial-state regions or global minimality. Evidence:
RESULTS_CONTINUOUS_NONLINEAR_BUDGET_CLOSURE.md and
validation/continuous_nonlinear_budget28_acceptance.json.
## Version 0.30 continuous initial-region evidence

The verifier now certifies Cartesian products of two continuous initial-state
boxes and a complete finite-horizon action-word box. It proves either a uniform
within-tolerance upper statement for every state pair and action word, or a
robust lower statement using one stored action word that separates every
cross-region state pair. Branch trees may split left-state, right-state, or
action coordinates; the independent verifier checks exact product coverage and
recomputes every leaf and global witness.

Formal profiles span 8/32/64 state dimensions, 2/3/4 controls, and horizons
3/5/10 with per-coordinate radius 0.001. Relational bounds close all three near
regions at upper bound 0.0165 against threshold 0.02 and all three far regions
at robust lower bound 0.2385 against threshold 0.20. Independent IBP leaves
every near region unresolved at eight leaves. Three models and 12 certificates
replay exactly. This establishes local continuous-region certification in the
stable ReLU regime, but not a global quotient or minimal realization over the
entire continuous state space. Evidence: RESULTS_CERTIFIED_CONTINUOUS_REGIONS.md,
docs/CONTINUOUS_REGION_METHOD.md, and
validation/certified_continuous_regions_acceptance.json.
R13 release evidence for 0.30.0 is validation/wheel_v30_run/status.json:
source, wheel, and isolated-installed modules matched; all 12 certified
generation/replay commands passed; 134 regression tests passed. Wheel SHA-256:
92c2a3b66e590cd224f3bb453740dba20878744b7edf1a0fce91684c803cb6b2.
## Version 0.31 global continuous behavioral-cover evidence

The project now covers complete uncountable unit state domains with certified
finite response codebooks. Every grid cell has a universal region-to-center
upper proof over all three-step action words. A complete packing grid supplies
pairwise separation certificates greater than twice epsilon, giving a lower
bound for any epsilon response codebook.

At epsilon 0.101, the exact minimum cover number is K=5 on the unit interval
and K=25 on the unit square. The formal run verifies 30 cell upper certificates
and 310 packing lower certificates, and all 340 replay deterministically. This
closes minimum finite-horizon behavioral covering complexity for the declared
contractive ReLU systems. It does not require representatives to be closed
under transitions and therefore does not yet establish a finite-state causal
quotient or infinite-horizon realization. Evidence:
RESULTS_CERTIFIED_CONTINUOUS_COVER.md,
docs/CONTINUOUS_COVER_METHOD.md, and
validation/certified_continuous_cover_acceptance.json.

R13 release evidence for 0.31.0 is validation/wheel_v31_rebuild_run/status.json: source, wheel, and isolated-installed modules matched; all 14 certified generation/replay commands passed; 138 regression tests passed. Wheel SHA-256: 4e7664edff404e47149ba94f049cd05cedb85025bd8a468491dce6646cff512f.

## Version 0.32 infinite-horizon finite realization

A deterministic finite-state realization now closes under all continuous
controls for the frozen coordinate-separable ReLU benchmark on [0,1]^d.
Its inductive interval certificate proves L-infinity observation error at most
0.101 for every run length, including infinite action sequences. Ten grid
states per axis give 10 states in 1D and 100 in 2D. Initial-output packing
proves lower bounds of 5 and 25; the bounds do not close, so minimality remains
open. The proof requires exact network matching and does not extend to
arbitrary coupled or phase-crossing systems. Evidence:
RESULTS_CERTIFIED_CONTINUOUS_REALIZATION.md,
docs/CONTINUOUS_CLOSED_REALIZATION_METHOD.md, and
validation/certified_continuous_realization_acceptance.json.
Version 0.32.0 release verification: 142 regression tests and 16
isolated-wheel certified generation/replay commands passed. Source, wheel,
and isolated-installed Python modules matched byte for byte. Wheel SHA-256:
0909a44c4ffffe3314403ae681c9a891633190e2d7859fb046e19552d3275168.
Evidence: validation/wheel_v32_run/status.json.
## Version 0.33 generic coupled-ReLU realization

The infinite-horizon interval checker now accepts a supplied serialized ReLU
network and independently checks its full state-action transition table.
A 2D system with cross-coordinate coupling and an internal ReLU phase boundary
is certified with 100 states at output tolerance 0.12 and relation radius
0.115. The verifier checks 100 initial cells, 100 relation-observation boxes,
10,000 transition boxes, and 300 packing pairs; the size interval remains
25–100. A changed-weight positive control also certifies. Failed grids return
unresolved rather than an upper bound. Evidence:
RESULTS_GENERIC_GRID_REALIZATION.md,
docs/GENERIC_GRID_REALIZATION_METHOD.md, and
validation/certified_generic_grid_acceptance.json.
Version 0.33.0 passed 147 regression tests and 18 installed-wheel
generation/replay commands. Source, wheel, and installed modules matched.
Wheel SHA-256: e4bc3ec740ff70fbfa6308a494f8430c036dd76cb69ed35a62b933c7c0ad5d01.
Evidence: validation/wheel_v33_run/status.json.
## Version 0.34 transition-aware lower certificate

For the frozen one-dimensional continuous ReLU system, a new exact-rational
overlap proof excludes every deterministic 1–5-state realization under all
continuous action words and all time horizons. Together with the independently
replayed ten-state executable upper model, the minimum-state interval is now
6–10. This is strictly stronger than pairwise initial-output packing, which
only gave five. The exact minimum and a nonlinear coupled-system lower theorem
remain open. Evidence: RESULTS_TRANSITION_AWARE_LOWER.md,
docs/TRANSITION_AWARE_LOWER_BOUND_METHOD.md, and
validation/certified_transition_lower_acceptance.json.
## Version 0.35 multi-switch lower and shifted upper

An exact-rational multi-switch theorem excludes every deterministic
1–6-state realization of the frozen scalar continuous ReLU system.
A separate nine-center executable model certifies uniform output error
at most 0.101 for every continuous action sequence and every time.
The independently replayed one-dimensional minimum-state interval is
therefore 7–9. The coordinate-product upper bound in two dimensions
drops from 100 to 81, while the complete packing lower bound is 25.
These are certified intervals, not exact minimum numbers. The theorem
currently requires scalar affine dynamics and identity output; the upper
product proof requires the frozen separable network. Evidence:
RESULTS_MULTISWITCH_LOWER.md, RESULTS_SHIFTED_REALIZATION.md,
docs/MULTISWITCH_LOWER_BOUND_METHOD.md,
docs/SHIFTED_REALIZATION_METHOD.md,
validation/certified_multiswitch_lower_acceptance.json, and
validation/certified_shifted_realization_acceptance.json.
Version 0.35.0 release verification: 159 regression tests passed.
The isolated wheel matched all source modules and completed all 24 certified
generation/replay CLI commands. Wheel SHA-256:
533f19e99f879a197d61080ecae0579c1ce1b3a56e1341f684fb9c2e3e7cfaaa.
Evidence: validation/wheel_v35_run/status.json.

## Version 0.36 trained compositional realization evidence

The frozen 8/32/64/128-dimensional trained affine ReLU systems now have
exact-rational, full-unit-domain, infinite-horizon intervention simulation
certificates. A coordinate-weighted abstraction collapses weakly influential
interior coordinates to one bin and certifies the same interval [81,31,104]
at epsilon 0.17 for all four models. It improves over the uniform 14^d
upper bound without enumerating a transition table. All certificates replay
from the frozen serialized weights; 163 tests and five isolated-wheel
generation/replay commands passed. Evidence:
RESULTS_COMPOSITIONAL_TRAINED_REALIZATION.md,
validation/compositional_trained_acceptance.json, and
validation/wheel_v36_run/status.json.

This does not establish exact minimality, naturally trained nonlinear traffic
dynamics, or completion of R4/R5/R8/R9/R10.



Release 0.37.0 is recorded in 'validation/wheel_v37_run/status.json'.
All 165 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte. Four installed-package commands passed:
fresh nonlinear generation/replay, formal 128D nonlinear replay, and the
earlier weighted-affine replay. Wheel SHA-256:
563d7d34dfcce68210c8ee2d0fedfff08a24082b0e0476e4eea10f277b9bb805.



## Version 0.37 trained nonlinear global evidence

Two independent seeds and 8/32/64/128 state dimensions give eight frozen
ReLU recurrent models with genuine state- and control-dependent phase
crossings. Disjoint train/selection/test splits, held-out one-step tests,
and ten-step rollouts pass the fixed gates. Exact-rational certificates
cover the full continuous unit state/action domains for every horizon and
give [81,2250] as the state-complexity interval at epsilon 0.17 in all
eight cases. Retraining and certificate replay pass. A matched-data,
post-hoc affine diagnostic has at least 47.3 times higher held-out RMSE.
Evidence: RESULTS_TRAINED_NONLINEAR_GLOBAL.md and
validation/trained_nonlinear_global_acceptance.json.

The simulator is synthetic and the hidden feature layer is structured.
Exact minimality, external traffic validity, unconstrained causal
representation learning, and broad R4/R5/R8/R9/R10 remain incomplete.



## Version 0.38 learned local-phase evidence

The fixed hinge dictionary in version 0.37 has been replaced with jointly
trained local hidden ReLU directions, biases, and readouts. Two fresh
seeds at 8/32/64/128 dimensions give eight frozen networks. Every case
passes the prewritten disjoint-split quality, actual hidden-parameter
movement, dual phase-witness, and full-unit-domain infinite-horizon
certificate gates at epsilon 0.17. Each minimum-state interval remains
[81,2250]. Complete retraining and exact certificate replay pass.
A post-hoc same-data comparison improves held-out RMSE by a factor of
1.32-1.71 over the previous fixed-dictionary model. Evidence:
RESULTS_LEARNED_LOCAL_GLOBAL.md,
validation/learned_local_global_acceptance.json, and
validation/learned_local_dictionary_comparator.json.

Known sparse locality, a synthetic teacher, unmatched minimality bounds,
external validity, and R4/R5/R8/R9/R10 remain incomplete.



Release 0.38.0 is recorded in 'validation/wheel_v38_run/status.json'.
All 167 regression tests passed. Source, wheel, and isolated-installed
Python modules matched byte for byte. Four installed-package commands
passed, including fresh local-phase generation/replay and formal 128D
full retraining/replay. Wheel SHA-256:
07db257e1306c4e7cb9ec9e704fb27ed016fb0618f753f26e88ca178b4a6a67d.



## Version 0.39 automatic global-grid synthesis

A weight-derived proposal algorithm now chooses coordinate state-bin counts
and relation radii without a handwritten grid. It uses a contractive
absolute-influence bound and a greedy log-state-cost rule. The search itself
is not trusted: the exact-rational verifier checks every proposed finite
machine against the complete continuous state/action cubes.
Twenty-one trained and coupled cases regenerate and replay. Eight learned
local-phase networks improve from 2250 supplied states to 1000-1125
automatic states; the coupled phase-crossing 2D network improves from 100
to 64, retaining a separate exact 25-state packing lower bound.
The automatic result on trained affine networks is worse than the existing
manual 31,104-state model, so optimality is not claimed.
Evidence: RESULTS_AUTOMATIC_GRID_GLOBAL.md and
validation/automatic_grid_global_acceptance.json.

This removes manual grid design but does not recover the neural causal
support from interventions, prove a globally minimal quotient, validate
an external physical system, or complete R4/R5/R8/R9/R10.



Release 0.39.0 is recorded in 'validation/wheel_v39_run/status.json'.
All 170 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte; five installed-package commands passed,
including fresh coupled-grid synthesis/replay and formal learned and affine
replays. Wheel SHA-256:
91a32e50d87303cb683d05f44cc576d6ed3c04128f7a5892727a5388eb958a9e.

## Version 0.40 verifier-guided interventional support

The transition support candidate now comes from behavior-only coordinate
interventions. A separate exact-rational checker proves each proposed edge
with an input-pair witness and proves each nonedge with the absence of a
nonzero-weight path. Possible paths without witnesses remain unresolved.
The 24-case study covers 20 frozen trained networks at 8/32/64/128D and
four controls. Twenty-two cases have complete support certificates:
5,191 certified edges and 105,525 certified nonedges across all cases;
the narrow-tent and path-cancellation controls each retain one unresolved
pair. See RESULTS_INTERVENTIONAL_SUPPORT.md,
docs/INTERVENTIONAL_SUPPORT_METHOD.md, and
validation/interventional_support_acceptance.json.

This establishes only coordinate-specific one-step support for frozen
networks with declared state/action coordinates. The verifier reads the
network weights; latent variables, the minimum causal quotient, external
validity, and original R4/R5/R8/R9/R10 remain open. Release 0.40.0 is
recorded in validation/wheel_v40_run/status.json: 174 tests and five
isolated-installed commands passed, with byte-identical source, wheel,
and installed modules.

## Version 0.41 exact function-level support

A bounded exact ReLU activation-region checker now addresses the two
unresolved 0.40 controls. Strict rational Fourier-Motzkin elimination
identifies feasible full-dimensional regions. A nonzero regional slope
produces an exact intervention witness; zero slopes across all regions
prove coordinate independence by continuity. The 27-case formal study
replays with 5,197 certified edges, 105,531 certified nonedges, and no
unresolved pairs. Six edge witnesses and two absence proofs require the
new region checker. Seven controls include narrow, oblique and three-input
coupled tents plus duplicate and distinct-hinge cancellations.
Evidence: RESULTS_FUNCTIONAL_SUPPORT.md,
docs/FUNCTIONAL_SUPPORT_METHOD.md, and
validation/functional_support_acceptance.json.

The behavior proposal still reads only the callable network, but the new
exact-region search reads frozen weights. This is coordinate-specific
one-step support, not general latent-variable discovery or proof of a
minimum causal quotient. Region reasoning has explicit three-input,
twelve-hidden-unit and search caps. Larger networks can still be
certified when query witnesses and structural zero paths suffice.
General large-network functional equivalence, external validity, and
original R4/R5/R8/R9/R10 remain open. Release 0.41.0 in
validation/wheel_v41_run/status.json passed 180 regression tests and
eight isolated-installed commands with byte-identical modules.

## Version 0.42 exact all-horizon behavioral quotient

For frozen ReLU networks proven globally affine and invariant on the full
state/action cube, the project now computes the exact future-response
equivalence classes. The quotient map is the rational observability row
space of (C,A), with exact induced dynamics and output checks. A cyclic
triangular proof certifies that the previously trained 8/32/64/128D
affine models have full exact quotient dimension, despite four immediate
observations. In 128D an initially invisible coordinate first changes
an output after 124 steps; the exact difference is nonzero but about
2.82e-161. A controlled counterfactual changes only effective
A[0,127] and lowers the exact quotient dimension from 128 to 4.
A separate mixed-state control has a non-coordinate one-dimensional
quotient; a constant-output control has dimension zero. A nonlinear
phase-changing control correctly remains unresolved.

The same frozen 128D model independently replays its epsilon=0.17
finite-state complexity interval [81, 40824]. Exact quotient dimension
and finite epsilon state count are different quantities. Evidence:
RESULTS_AFFINE_BEHAVIORAL_QUOTIENT.md,
docs/AFFINE_BEHAVIORAL_QUOTIENT_METHOD.md, and
validation/affine_quotient_acceptance.json. Release 0.42.0 in
validation/wheel_v42_run/status.json passed 189 regression tests and
eight isolated-installed commands with byte-identical modules.

This does not settle nonlinear all-horizon equivalence, a globally
minimum finite epsilon realization, arbitrary latent coordinate
discovery, external validity, or original R4/R5/R8/R9/R10.

## Version 0.43 invariant-slice lower bound

The frozen separable 2D ReLU benchmark now has a 28-state lower bound
at epsilon 0.101, improving the previous 25-state initial-output
packing while retaining the 81-state executable upper certificate.
The proof checks exact coordinate separability, regenerates the scalar
seven-state transition lower theorem, and constructs four disjoint
fixed-point slices with admissible constant actions. Each slice
requires seven abstract states; output separation prevents states
from being shared across slices. The exclusion applies to arbitrary
encoders, abstract outputs and deterministic action transitions for
all initial states and all continuous action words. Evidence:
RESULTS_INVARIANT_SLICE_LOWER.md,
docs/INVARIANT_SLICE_LOWER_METHOD.md, and
validation/invariant_slice_lower_acceptance.json.

This does not close the scalar 7?9 or 2D 28?81 minimum-state intervals,
nor does it automatically apply to coupled nonlinear or trained
high-dimensional systems. Release 0.43.0 in
validation/wheel_v43_run/status.json passed 192 regression tests and
five isolated-installed commands with byte-identical modules.
Original R4/R5/R8/R9/R10 remain open.


## Version 0.44 exact scalar minimum

The frozen scalar affine-in-ReLU benchmark at exact binary64 tolerance
0.101 has a nine-state minimum over arbitrary deterministic
intervention-labelled finite realizations. A 40,545-token branch proof
enumerates every possible increasing target subchain for each of eight
interval hulls. All 40,387 closed branches have exact rational Farkas
identities, independently replayed without a numerical solver. Interval
duplication rules out all smaller sizes. A separate exact rational
nine-state construction closes the upper bound for all continuous action
sequences. Four disjoint invariant slices raise the separable 2D lower
bound from 28 to 36; its 81-state upper bound remains, so its minimum
is not identified. Evidence: RESULTS_EXACT_INTERVAL_CHAIN.md,
docs/EXACT_INTERVAL_CHAIN_METHOD.md,
validation/exact_interval_lower_8.json.gz, and
validation/exact_interval_lower_acceptance.json. This theorem does not
settle coupled nonlinear or trained high-dimensional minimum-state
realization; original R4/R5/R8/R9/R10 remain open.

Release 0.44.0 passed 195 regression tests and isolated-wheel replay.
All 104 installed modules matched source bytes; the wheel SHA-256 is
`bed88cf2d7f2c4b69478842e3542b923ee81f7022f830a8d24ecaa8c73440519`
(`validation/wheel_v44_run/status.json`).


## Version 0.45 exact separable-product amplification

The exact scalar nine-state minimum now yields symbolic lower and upper
bounds `9*4^(d-1) <= K_d <= 9^d` for any dimension whose serialized
ReLU network is proved, by exact affine extraction and full-cube phase
checking, to implement the coordinatewise benchmark. Frozen 2/8/32/128D
models replay from JSON, including all 131,712 parameters at 128D. A
hidden-layer permutation and positive rescaling leaves the 8D result
unchanged, showing that the claim is about the realized affine function
within the globally fixed-phase class rather than one neuron ordering.
Coupled 2D and trained 128D networks fail closed. Evidence:
RESULTS_SEPARABLE_PRODUCT_BOUNDS.md,
docs/SEPARABLE_PRODUCT_BOUNDS_METHOD.md, and
validation/separable_product_bounds_acceptance.json. The exponentially
large numbers apply only to the exactly separable synthetic family; they
do not resolve trained or nonlinear high-dimensional minimum realization
or the original open R4/R5/R8/R9/R10 objectives.

Release 0.45.0 passed all 199 regression tests and isolated-wheel replay.
All 105 installed modules matched source bytes. The wheel SHA-256 is
`b0c8fc613c396fc3ea03074526feb1deeae83a7024ff1b8858c6ace72e7633af`
(`validation/wheel_v45_run/status.json`).


## Version 0.46 exact dynamic packing on trained affine rings

A generic exact finite-horizon intervention packing verifier now accepts
rational initial states and a shared action word for arbitrary frozen
ReLU systems. It either proves full-domain fixed ReLU phases and uses a
sparse exact affine evaluator, or evaluates the nonlinear network exactly
at the witness points. A 135-point one-step witness raises the trained
affine 128D ring's finite-state lower bound from 81 to 135 at epsilon
0.17. All 9,045 pairs are checked; 54 first separate after one zero
action. The old same-model 40,824-state upper certificate replays, so the
new interval is [135,40824]. The witness also passes on affine 8/32/64D
profiles. Removing the effective feedback edge or switching to two
trained nonlinear 128D networks causes a specific critical pair to fail;
this says only that the proposed witness does not transfer. Evidence:
RESULTS_DYNAMIC_PACKING.md, docs/DYNAMIC_PACKING_METHOD.md, and
validation/dynamic_packing_acceptance.json. Finite-state minimality of
these high-dimensional models and original R4/R5/R8/R9/R10 remain open.

Release 0.46.0 passed all 202 regression tests and isolated-wheel replay.
All 106 installed Python modules matched source bytes. The wheel SHA-256
is `72af6d415bcc9875161a934a1ab45757b41fa042b637ff05ecac39cff863a80b`
(`validation/wheel_v46_run/status.json`).


## Version 0.47 exact integer-grid refinement

For the frozen trained affine 128D ring at binary64 epsilon 0.17, an
automatic bounded integer search proposes two smaller weighted grids.
The original 40,824-state certificate and the new 135-state dynamic
packing lower are replayed on the same serialized model. Independent
exact-rational verification certifies a 27,648-state state-only grid and
a 27,216-state state/action grid; the latter uses 512 implicit action
bins per coordinate and reduces the upper bound by exactly one third.
The minimum-state interval becomes [135,27216] for all unit-domain
initial states and continuous action words over unbounded time.
Evidence: RESULTS_INTEGER_GRID_REFINEMENT.md,
docs/INTEGER_GRID_REFINEMENT_METHOD.md,
validation/integer_grid_refinement_acceptance.json, and
runs/integer_grid_refinement_v1/state_action/certificate.json.
The search is not globally exhaustive; trained nonlinear finite-state
minimality and original R4/R5/R8/R9/R10 remain open.


Release 0.47.0 passed 203 regression tests and isolated-wheel replay;
all 107 installed modules matched source bytes. Wheel SHA-256:
`858d2f9f74651c03de3c5643a79bc09519cdc7073480d7a1e92ad0017855369b` (validation/wheel_v47_run/status.json).


## Version 0.48 exact weighted-grid class optimality

A ten-step exact-rational finite sensitivity sum proves that the
27,216-state construction is minimal among *all* uniform
coordinate-grid certificates satisfying the weighted verifier for the
same frozen trained affine 128D network and exact tolerance 17/100.
The proof quantifies over arbitrary positive integer bins on all 128
state coordinates, arbitrary action-bin counts, and positive relation
radii. It requires at least six bins on each of observed axes 1, 2,
and 3; an AM-GM branch handles any refinement of other axes; 125
exact integer exclusions handle the remaining two-axis case.
The existing 27,216-state upper matches the class lower. Evidence:
RESULTS_WEIGHTED_GRID_OPTIMALITY.md,
docs/WEIGHTED_GRID_OPTIMALITY_METHOD.md,
runs/weighted_grid_optimality_v1/certificate.json, and
validation/weighted_grid_optimality_acceptance.json.

This is a structural limit of the *proof/abstraction class*, not the
minimum finite-state count over arbitrary realizations. The latter
remains in [135,27216], and original R4/R5/R8/R9/R10 remain open.


Release 0.48.0 passed 205 regression tests and isolated-wheel replay.
All 108 installed Python modules matched source bytes. Wheel SHA-256:
`f7ae7adc341ae82b16e3893f7e5d33b7bdfcb01f65ec7a1a69daacd165431101` (validation/wheel_v48_run/status.json).


## Version 0.49 exact all-horizon trace-packing capacity

A 162-point rational witness improves the same trained affine 128D
network's arbitrary-realization lower bound from 135 to 162 at
binary64 epsilon 0.17. Exact replay checks all 13,041 pairs.
An independent analytic certificate proves no pairwise common-action
output-trace packing can exceed 162 at any horizon: 81 observed
coordinate cells hold at most two one-step-distinguishable points each,
and a full-cube affine difference invariant proves that pairs
unseparated at times 0 and 1 stay unseparated forever. The result
also replays on frozen affine 8/32/64D rings. The certified general
minimum-state interval is [162,27216]. Evidence:
RESULTS_ALL_HORIZON_PACKING_CAPACITY.md,
docs/ALL_HORIZON_PACKING_CAPACITY_METHOD.md,
validation/one_step_capacity_acceptance.json, and
runs/one_step_capacity_v1/affine_d128/capacity_certificate.json.

The all-horizon packing maximum is not the globally minimal
deterministic realization; transition-consistency lower bounds and
non-grid upper models remain open. Original R4/R5/R8/R9/R10 also
remain open.


Release 0.49.0 passed 206 regression tests and isolated-wheel replay;
all 109 installed Python modules matched source bytes. Wheel SHA-256:
`99c82c2321981f25fd0408a03b5f3278d8bc984d8541196e3600f00eadd56829` (validation/wheel_v49_run/status.json).


## Version 0.50 exact all-horizon behavioral cover

For each frozen affine 8/32/64/128D ReLU ring, 162 explicit concrete
initial states give a uniform epsilon=0.17 output-trajectory cover
for every initial state, every common continuous action word and all
times. Exact rational affine and invariant checks prove the upper;
the v0.49 exact 162-point packing gives a matching lower. Hence the
all-horizon behavioral cover number is exactly 162. Evidence:
RESULTS_ALL_HORIZON_BEHAVIORAL_COVER.md,
docs/ALL_HORIZON_BEHAVIORAL_COVER_METHOD.md,
validation/behavioral_cover_acceptance.json, and
runs/behavioral_cover_v1/affine_d128/certificate.json.

A trajectory cover is not a transition-closed deterministic finite
realization. The 128D finite-machine minimum remains in [162,27216];
general causal quotient recovery and original R4/R5/R8/R9/R10 remain
open.


Release 0.50.0 passed 207 regression tests and isolated-wheel replay;
all 110 installed Python modules matched source bytes. Wheel SHA-256:
`3c23d6a327d8b16c76da14a5a1b4a0c5bfc5302f7cb35c586c0c1be592699506` (validation/wheel_v50_run/status.json).


## Version 0.51 reachable two-stage finite realization

A tagged deterministic finite machine for the frozen trained affine
128D ReLU ring now has 243 full-cube initial states and 4,480
reachable recurrent states, for an exact certified upper of 4,723
at rational tolerance 17/100. The initial state/output/transition
functions are executable. Exact rational checks prove the
initial-output error, one-step handoff into the existing weighted
recurrent relation, and closure of the retained recurrent index
set under all continuous unit actions. The independent binary64
0.17 packing lower remains 162, so the same-model minimum-state
interval is [162,4723]. Evidence:
RESULTS_REACHABLE_TWO_STAGE.md,
docs/REACHABLE_TWO_STAGE_METHOD.md,
validation/reachable_two_stage_acceptance.json, and
runs/reachable_two_stage_v1/affine_d128/certificate.json.

This reaches beyond the v0.48 full-cube uniform-grid class by
separating initial and recurrent states; it does not close the
general finite-machine minimum or original R4/R5/R8/R9/R10.


Release 0.51.0 passed 208 regression tests and isolated-wheel replay;
all 111 installed Python modules matched source bytes. Wheel SHA-256:
`ef32785ee9238486f55ac2ec6d75d4cd07f0167d7992fe8d73bd1177c8bad461` (validation/wheel_v51_run/status.json).


## Version 0.52 exact abstract action-box closure

The frozen trained affine 128D ReLU ring now has an executable 2,303-state finite realization: 243 full-cube initial states and 2,060 recurrent states in the least closed conservative action-box successor graph. Exact rational affine extraction and continuous-action coordinate intervals produce 1,952 initial successor cells and 51,304 enumerated recurrent edges. The verifier recomputes the full sorted graph and proves closure inside the previous certified two-stage recurrent rectangle; previous initial-output, handoff and recurrent simulation inequalities remain valid. With the independent 162-state packing lower, the same-model general minimum interval is [162,2303]. Evidence: RESULTS_ABSTRACT_REACHABILITY.md, docs/ABSTRACT_REACHABILITY_METHOD.md, validation/abstract_reachability_acceptance.json, and runs/abstract_reachability_v1/affine_d128/certificate.json.

The graph is a conservative abstraction, not the exact concrete reachable set. It does not settle global minimality, nonlinear 128D certificates, or original R4/R5/R8/R9/R10.

Release 0.52.0 passed 209 regression tests and isolated-wheel replay; all 112 installed Python modules matched source bytes. Wheel SHA-256: `2caca954eb1b4a690dc4d13af08339d0d3151bea0f061b2eeec89afe163a9262` (validation/wheel_v52_run/status.json).


## Version 0.53 exact shared-action pairwise support closure

For the frozen trained affine 128D ReLU ring, an executable finite realization now has 243 initial and 899 recurrent states, an exact upper of 1,142 at tolerance 17/100. The checker extracts exact affine action coefficients, constructs 60 pairwise shared-action support intervals, and rejects target cells only when rational intervals are strictly disjoint. A least fixed-point closure of the resulting conservative graph examines 6,768 initial candidate edges and 20,672 recurrent candidate edges, retaining 859 distinct initial successor cells and 6,368 recurrent edges. The verifier recomputes all 899 ordered recurrent states, checks they remain within the v0.52 certified graph, and reuses its exact initial handoff and recurrent simulation proof. With the independent 162-state packing lower, the same-model general minimum interval is [162,1142]. Evidence: RESULTS_CORRELATED_REACHABILITY.md, docs/CORRELATED_REACHABILITY_METHOD.md, validation/correlated_reachability_acceptance.json, and runs/correlated_reachability_v1/affine_d128/certificate.json.

The pairwise filter is necessary but not a complete action-polytope feasibility test; this graph remains conservative. Global minimality, nonlinear 128D certificates and original R4/R5/R8/R9/R10 remain open.

Release 0.53.0 passed 210 regression tests and isolated-wheel replay; all 113 installed Python modules matched source bytes. Wheel SHA-256: `173d35dd57c027d3f93e8d318c79921d70cab2cbcc699e822365b473e1ea8436` (validation/wheel_v53_run/status.json).


## Version 0.54 jointly chosen state/action grid with reachable closure

For the frozen trained affine 128D ReLU ring, the independently certified state/action grid with active bins (14,6,6,6,9) and 512 implicit action bins supports a new exact two-stage handoff from 243 full-cube initial states. Exact shared-action pairwise support closure retains 798 recurrent states, giving an executable 1,041-state machine at rational tolerance 17/100. The checker recomputes 6,480 initial candidate edges and 14,696 recurrent candidate edges, retains 780 distinct initial successors and 4,901 recurrent edges, and checks the full ordered graph lies inside the certified transition-image rectangle. This rectangle has 4,992 cells, larger than the previous grid's 4,480, yet the closed graph is smaller than its 899 cells. With the independent 162-state lower certificate, the same-model general minimum interval is [162,1041]. Evidence: RESULTS_DUAL_GRID_REACHABILITY.md, docs/DUAL_GRID_REACHABILITY_METHOD.md, validation/dual_grid_reachability_acceptance.json, and runs/dual_grid_reachability_v1/affine_d128/closure_certificate.json.

The graph is a conservative upper construction, not a globally minimal realization. Nonlinear 128D certificates and original R4/R5/R8/R9/R10 remain open.

Release 0.54.0 passed 211 regression tests and isolated-wheel replay; all 114 installed Python modules matched source bytes. Wheel SHA-256: `33c7e3d99f45b62e7a386443bc9689d73a971617dfd96835c73624f24e5ce11e` (validation/wheel_v54_run/status.json).


## Version 0.55 trained nonlinear two-stage realization

For two frozen seeds each at 8/32/64/128 state dimensions, the actual phase-crossing trained ReLU transition network now has an executable 459-state finite realization at rational epsilon 17/100. Exact state/control phase second-difference witnesses are replayed. A 243-state full-cube initial grid hands off, under every continuous action, to the prior exact weighted recurrent relation. Only the 216 recurrent cells in the certified full-cube transition-image enclosure are retained. The independent 81-state lower remains, so all eight same-model minimum intervals narrow from [81,2250] to [81,459]. Evidence: RESULTS_NONLINEAR_TWO_STAGE.md, docs/NONLINEAR_TWO_STAGE_METHOD.md, validation/nonlinear_two_stage_acceptance.json, and eight certificates under runs/nonlinear_two_stage_v1.

The model family has structured fixed hidden features and learned output coefficients on synthetic traffic-inspired data. The recurrent enclosure is conservative and the global minimum is unknown. This does not close end-to-end causal decompilation or original R4/R5/R8/R9/R10.

Release 0.55.0 passed 212 regression tests and isolated-wheel replay over all eight cases; all 114 installed Python modules matched source bytes. Wheel SHA-256: `a48ed39caae8a6a8d04e1f3685ec824cf375289d75c2678d27b0aadfd4e0b978` (validation/wheel_v55_run/status.json).


## Version 0.56 exact minimum initial grid in a fixed proof class

The same eight frozen phase-crossing trained ReLU rings now have exact
finite-realization uppers of 297 at 8D and 324 at 32/64/128D. Initial grid
synthesis uses 81 or 108 states and retains the previous 216 recurrent
cells. Four direct observed axes force at least 81 initial coordinate
cells; below 108 the only grid is the 81-cell baseline. Exact sensitivity
handoff accepts that baseline at 8D, excludes it at larger dimensions,
and accepts a 108-cell refinement. This proves the exact minimum initial
grid in the fixed recurrent sensitivity proof class. All hidden initial
axes have one bin. The general machine intervals remain [81,297] and
[81,324], with the original 81-state lower unchanged. Evidence:
RESULTS_OPTIMAL_INITIAL_GRID.md, docs/OPTIMAL_INITIAL_GRID_METHOD.md,
validation/optimal_initial_grid_acceptance.json, and eight certificates
under runs/optimal_initial_grid_v1.

A failed sensitivity upper inequality excludes a grid from this proof
class; it is not a concrete unsafe trajectory or a global impossibility
proof. Structured synthetic models, arbitrary partitions, alternative
relations, global minimum and original R4/R5/R8/R9/R10 remain open.


Release 0.56.0 passed 213 regression tests and isolated installed-wheel
replay over all eight cases. All 115 installed Python modules matched
source bytes. The checker binds the exact acceptance and JUnit file hashes
and replays original lower/two-stage proofs, new optimal-initial-grid
proofs, phase witnesses and executable traces. Wheel SHA-256:
`c21b1e0931699b099a315930eef988dfea1e93926188d85273c58368e1e0bc01`. Evidence: validation/wheel_v56_run/status.json and
validation/pytest_v56.xml.


## Version 0.57 nonlinear shared-action closure and standalone programs

All eight frozen phase-crossing trained networks now have certified
finite-realization uppers of 132/166/162/163 at dimensions 8/32/64/128,
respectively. Their 81/108 initial states enter complete closed recurrent
graphs of 51/58/54/55 cells. Exact two-dimensional action phase polygons
and joint target-cell intersection retain the actual shared-action
correlations. Every initial center and every retained recurrent center is
checked, including closed-cell point/segment boundary contacts.

A verified exporter produces portable finite programs. Every model uses
one shared normalized action template plus source-dependent offsets and
state/output/successor rows. Execution takes only program JSON and needs
no source network; tests disable neural execution/compilation helpers.
The general minimum intervals are [81,132], [81,166], [81,162], [81,163].
Evidence: RESULTS_NONLINEAR_ACTION_CLOSURE.md,
docs/NONLINEAR_ACTION_CLOSURE_METHOD.md,
validation/nonlinear_action_closure_acceptance.json, and eight graph and
program artifacts under runs/nonlinear_action_closure_v1.

The conservative graph and standalone program are certified uppers at
17/100 tolerance. Structured shallow synthetic models, global state/code
minimum, uniqueness, broader network/intervention families and original
R4/R5/R8/R9/R10 remain open. This strengthens executable certified neural
realization without replacing the original end-to-end requirements.


Release 0.57.0 passed 216 regression tests and isolated installed-wheel
replay of all eight complete graph certificates and standalone programs.
All 116 installed Python modules matched source bytes. Exact phase
witnesses, original lower/initial proofs, joint action graphs, factored
program templates and installed standalone traces passed. The status
binds the acceptance, program and JUnit hashes. Evidence:
validation/wheel_v57_run/status.json and validation/pytest_v57.xml.
Wheel SHA-256: `cdf993c5775dd5058aa0a29aebb85b6baa4a28cd9b4de8ee3cfb6b6a48eadacb`.


## Version 0.58 minimum initial labels and transition-consistency lower

All eight frozen phase-crossing networks now use exactly 81 initialization
labels at epsilon 17/100. Exact signed state-phase extrema and action
phase gradient bounds replay every initial handoff inequality under the
unchanged recurrent relation. Packing proves 81 minimum initial labels
among arbitrary selectors. Complete joint-action closure and standalone
export give uppers 132/139/135/136 at dimensions 8/32/64/128.

A distinct continuum-coverage/common-successor theorem excludes any
81-state deterministic machine with a fixed state-only output map. Exact
one-step neural witnesses are replayed for all eight cases; arbitrary
partitions and simulation relations are allowed. The joint general
minimum intervals are [82,132], [82,139], [82,135], [82,136]. Initial label
minimum 81 is distinct from the whole-machine minimum. All persistent
memory needed by the decoder must be counted as state.

Evidence: RESULTS_PHASE_INITIAL_HANDOFF.md,
docs/PHASE_INITIAL_HANDOFF_METHOD.md,
docs/TRANSITION_CONSISTENCY_LOWER_METHOD.md,
validation/phase_initial_handoff_acceptance.json,
validation/transition_consistency_lower_acceptance.json, and eight new
program/lower pairs under runs/phase_initial_handoff_v1 and
runs/transition_consistency_lower_v1. Exact old graphs and programs remain
replayable. The family remains shallow and structured; global state/code
minimum, general deep networks and original R4/R5/R8/R9/R10 remain open.


Release 0.58.0 passed **221 regression tests**, with zero failures, errors
or skips. Isolated installed-wheel replay passed all eight cases,
including original lower/initial proofs, legacy graphs/programs, exact
phase-aware handoffs, minimum initial labels, the new transition-consistency
lower certificates, complete new graphs and standalone execution.
All **118 installed Python modules matched source bytes**. The status
binds both acceptance files, the JUnit report, program/lower certificate
hashes and the wheel. Evidence:
[wheel replay](../validation/wheel_v58_run/status.json),
[complete tests](../validation/pytest_v58.xml), and
[checker](../validation/check_wheel_v58.py).
Wheel SHA-256: `c4f4118b7e8fed1e9e7d4f7c7ca6433e03efe54113093d5dc6b3050d2b0f1908`.

The next implementation target is the
[joint mixed-state/control phase proof](NEXT_MIXED_RELU_PLAN.md).
The original full objective remains incomplete.
