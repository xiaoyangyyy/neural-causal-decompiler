# 当前进度

精确有限认证阶段已经完成；连续/近似神经系统和历史 R4/R5/R8/R9/R10
强目标仍未完成。范围见 docs/FULL_REQUIREMENTS.md。历史工件保持原样。

## 0.27 certified finite realization

- 新主线以有限确定性 Moore 系统为对象，动作标签与完整、闭合、可重置状态集显式声明。
- oracle 精确最小商与 response-only closed/consistent table 两条独立构造路径均已实现。
- upper certificate 逐项验证输出保持及全部 state/action 转移同态；lower certificate 重放每对抽象代表的区分词。
- seed 2701 的 12 个正式案例含 90 个具体状态和 54 个最小状态；主动恢复 12/12 精确并认证，共 678 次响应查询。
- 有限 L-infinity 近似认证在 epsilon=0/0.25/10 三点全部闭合，上下界总计为 90/54/12；未用两两兼容代替上界。
- 同查询预算下 random 为 7/12、passive 为 11/12、activation clustering 为 0/12；三者均没有完整性证书。
- 45% 查询预算实验保留 76 个 unresolved pair，没有把“未找到反例”报告成等价。
- 正式工件在 runs/certified_finite_seed2701；60 个证书独立完整重放通过；结果见 RESULTS_CERTIFIED_FINITE.md。
- 当前全套回归测试为 115 passed。

## 已验收
- V2 91/92：RESULTS_V2.md。
- 联合内部干预 191/192：RESULTS_JOINT.md。
- 多位置数值审计 293/294：RESULTS_NUMERIC.md。
- 内部评分参与选择 393/394：RESULTS_GUIDED.md；最终程序选择未改变。
- 显式边关系注意力 493/494：RESULTS_RELATIONAL.md；validation/relational_acceptance.json。
  每组 2,976 个世界、18 次机制恢复；独立重放进程 29345 已退出 0。
  原实验进程 50979/99525 句柄已不存在；不应重启这些实验。
  整图表现变化有升有降，不能宣称稳定改善。
- 历史该阶段全套测试：85 passed，validation/pytest_regression_numeric.xml。

## 本轮交付整合
- ncd CLI 新增 joint/numeric/guided/relational 和四个 verify 命令。
- README 记录命令及依赖关系；run-all 保留原双变量与多变量范围。
- 0.3.0 wheel 已构建并安装到 validation/wheel_v3_env。
- scripts/acceptance_release_v3.py 核对源码、wheel、安装目录字节一致，确认导入安装包，实际运行四类 quick 实验及重放。
- 隔离安装验证完成：四类新增实验各实际运行并独立重放，8 条命令全部退出 0。证据 validation/wheel_v3_run/status.json；会话 92071 已正常结束。
- 新版安装包原有 run-all 与 verify-all 也已实际运行通过，包含真实 PySR；证据 validation/wheel_v3_full/status.json，会话 70302 正常退出 0。所有安装包运行均为 quick 集成，正式科学实验仍以各历史冻结结果为准。

## 后续必须推进
数值统计计算的完整恢复、干预对齐覆盖、程序保真、图/父集/算子恢复仍不足。
关系偏置实验没有解决这些目标，需要依据失败分析改进计算与程序表达能力。
安装验证只证明交付可运行；不能替代科学结论。

## 图失败分解
- scripts/audit_graph_failures.py 校验输入 manifest 哈希后，将两个种子、三种节点数、五个环境、两种模型、两个家族分成 120 个分层。
- RESULTS_GRAPH_FAILURES.md 与 validation/relational_failure_audit.json：漏边、多边、反向、定向/无向混淆、类混淆矩阵、全边/活跃边/整图保真。
- 493 的 8 节点非线性关系模型：全边保真 93.1%，活跃边保真 69.3%，整图保真 14.5%。不能用全边一致率替代图级恢复。
- 493 的 3 节点高斯关系模型：程序全边保真仅 33.3%；应分别研究稀有方向未定类别和家族混合，而非只提升总体精度。
- 这些是冻结测试集上的探索性诊断，不能再把它们用作后续方法的独立确认集。

## 加权搜索与关系程序实验
- `fit_rule` 支持正的逐样本权重；默认行为与冻结 0.3.0 实现逐项一致。67 项测试在加入关系原语前通过。
- runs/weighted_rules_seed593 比较均匀、世界等权、世界与教师类别等权。六个程序和 180 个分层已独立重放（validation/weighted_rules_replay.json）；类别再平衡使两组整图保真下降，不作为改进。
- ncd/graph_program_features.py 加入同源、同目标、后继、前驱的显式均值原语，保持节点置换等变。
- 按 docs/RELATIONAL_PROGRAM_PROTOCOL.md 预先冻结的 693/694 实验已完成并独立重放（validation/relational_program_replay.json）。关系教师整图保真仅提高 +0.5%/+0.9%，区间跨零；无关系教师下降 -2.8%/-3.3%。详见 RESULTS_RELATIONAL_PROGRAM.md。
- 简单关系均值没有解决程序保真；693/694 已用于观察结果，后续修改不得把它们作为未触碰确认集。
- 当前源码已超过冻结的 0.3.0 wheel；其安装验收仍是历史版本证据，下一次完整交付需构建新版本。

## 原始统计运算干预
- ncd/raw_program_trace.py 从原始样本执行 14 个 CDIR 特征 AST，再执行冻结规则；内部标量可按 occurrence 或结构等价组干预。
- 八个组覆盖 mean/variance/std/correlation/variance ratio；祖先/后代组合在展开真实 AST 路径后被排除。
- ncd/fixed_numeric_mapping.py 使用一次预计算的非复用配对池训练映射，避免每个步骤重复计算原始统计量。
- quick 792：512 worlds、48/96 pairs，独立重放通过。失败的 quick 791 保留，原因是旧组兼容检查未展开祖先关系。
- 正式 793/794：每组 5,120 worlds、384 train pairs、1,024 test pairs、三个 neural sites；两组均重新训练映射并独立重放通过。证据 validation/raw_numeric_acceptance.json、RESULTS_RAW_NUMERIC.md。
- seed 794 的 variance ratio 发生极大 NMSE，完整八运算恢复失败；dependence 和 regression 内部仍未展开。

## 0.4.0 交付
- 统一 CLI 新增 raw-numeric / verify-raw-numeric。
- dist/neural_causal_decompiler-0.4.0-py3-none-any.whl 已构建；SHA-256 `c2514ae0081fb1c05d4a9717c50ae81ad2cc811625a51b01fb2a1ee7e3e6e394`。
- validation/wheel_v4_env 为隔离安装；validation/wheel_v4_run/status.json 证明从安装包实际运行 quick raw-numeric 和完整重放，两条命令退出 0，源码/wheel/安装模块字节一致。
- 下一阶段：展开 dependence kernel 与 cross-fit regression 的内部运算；为 variance ratio 的重尾误差设计预先固定的稳健目标和新确认种子。

## 依赖核内部机制
- RawDiscoveryExecutor 的 `trace_dependence=True` 版本化开关展开三个 dependence occurrence，每个包含两带宽、两核能量、numerator、denominator，共 18 组；默认关闭，793/794 的程序 ID 保持逐字一致。
- 配对引擎识别 bandwidth→energy/numerator/denominator 与 energy→denominator 的因果关系，排除祖先/后代联合干预。
- quick 892：512 worlds、18 groups、48/96 pairs，完整重放通过。
- 正式 893/894：每组 5,120 worlds、384/1,024 pairs、129 个未见兼容双组 mask；全部映射重新训练并独立重放通过（validation/dependence_numeric_acceptance.json）。
- 自然读出 NMSE 约 0.20–0.27，但干预 target NMSE 1.54–2.31，行为有效样本仅 20/24；可读出信息没有形成稳定因果交换。详见 RESULTS_DEPENDENCE_NUMERIC.md。
- regression 仍是黑箱；当前源码已超过 0.4.0 wheel，后续需新版本交付。

## 交叉拟合回归内部机制
- `trace_regression=True` 展开 X→Y/Y→X、两折、mean/std 与五个固定基函数系数；相同回归在 residual dependence/error 中同步，合计 28 组。
- mean/std→beta 的依赖被写入组合检查；默认及 dependence 程序 ID 均保持历史一致。
- quick 992：512 worlds、28 groups、48/96 pairs，完整重放通过。
- 正式 993/994：每组 5,120 worlds、384/1,024 pairs、338 个未见兼容双组 mask；两组独立重放通过（validation/regression_numeric_acceptance.json）。
- 自然读出 NMSE 0.56–0.71；数值干预 NMSE 1.89–3.00 且均差于无干预基线，行为最高 13.6%。回归内部信息可读，但未恢复可交换机制。详见 RESULTS_REGRESSION_NUMERIC.md。

## 0.5.0 交付
- 统一 CLI 新增 dependence-numeric / verify-dependence-numeric 与 regression-numeric / verify-regression-numeric。
- dist/neural_causal_decompiler-0.5.0-py3-none-any.whl，SHA-256 `80f157da8ac23b4edfdf5151334ff47a1c6def83a451818b0320592f0c9a32af`。
- validation/wheel_v5_run/status.json：隔离安装包分别实际运行 dependence/regression quick 实验并完整重放，4 条命令退出 0；源码、wheel、安装模块哈希一致。
- 当前完整 raw CDIR 计算已可执行和分层干预，但冻结教师的数值交换与行为保真仍低；下一步应改进非线性、多位置或非正交映射并使用新预注册种子验证。


## 0.6.0 biorthogonal mapping milestone

- Added exact dual read/write intervention bases, training-only orthogonal warm
  starts, full-pool checkpoint selection, stability diagnostics, and behavioral
  plus numerical audits.
- The all scope contains 8 raw, 18 dependence, and 28 regression groups.
- Initial confirmation seeds 1093/1094 were invalidated before metric
  inspection because 4,096 test worlds yielded only 1,383 pairs and failed
  all-mask coverage.
- Amended seeds 1193/1194 each accepted 2,048 pairs and covered all 1,411 masks.
  Both independently replayed and passed the fixed relative-improvement rule at
  all three cuts. Evidence: RESULTS_OBLIQUE_NUMERIC.md and
  validation/oblique_numeric_acceptance.json.
- Absolute errors remain too high for exact mechanism recovery. The full
  original objective remains active; next work should use this improved mapping
  to strengthen executable-program fidelity and graph/SCM recovery rather than
  treating relative intervention gains as completion.
- Current regression suite: 87 passed
  (validation/pytest_oblique.xml).


## 0.7.0 causal feature guidance milestone

- Added numerical-only interventions for all 14 fixed feature roots, with
  orthogonal, biorthogonal, shuffled, and random controls.
- Added candidate-independent feature support, aligned beam searches, common
  candidate provenance, untouched five-environment evaluation, and
  regeneration-based replay.
- Formal seeds 1393/1394 both selected different programs. Seed 1393 improved
  all environments (+2.34 pp mean); seed 1394 degraded all environments
  (-1.45 pp mean). Replication failed and the method is not accepted as an R4
  solution.
- The next method must estimate support robustness using selection-only
  subenvironments and new confirmation seeds; 1393/1394 are historical audit
  data.
- Current regression suite: 90 passed
  (validation/pytest_causal_guided.xml).


## 0.8.0 conservative guidance milestone

- Added a backward-compatible conservative selection policy: causal support
  cannot compensate for a lower validation fidelity-minus-complexity score.
- Added explicit abstention, selected-candidate gate provenance, CLI commands,
  full regeneration replay, and a fixed safe-improvement protocol.
- Formal 1493 changed programs and gained +0.27 pp on average but decreased
  scale/intervention fidelity. Formal 1494 changed programs and lost -0.68 pp;
  pooled change was -0.21 pp. Both replayed.
- Aggregate ID validation noninferiority is insufficient for OOD safety. The
  next selection stage must contain synthetic function/noise/scale/intervention
  validation strata and use new confirmation seeds.
- Current regression suite: 91 passed
  (validation/pytest_conservative_guided.xml).


## 0.9.0 OOD-guarded guidance milestone

- Added five selection-only frozen-teacher guard partitions for ID, function,
  noise, scale, and intervention shifts.
- A guided candidate must match or exceed the historical candidate on the
  ordinary validation objective and in every guard; otherwise selection
  explicitly abstains.
- Formal seeds 1593/1594 each used and replayed 12,800 worlds. The historical
  candidate was the sole eligible candidate in both runs, so selection did not
  change and every final-environment delta was zero.
- The safety gate behaved as designed on these runs, but the fixed improvement
  rule failed because no supported new program was accepted. Robust program
  synthesis, exact numerical mechanism recovery, graph recovery, and explicit
  SCM recovery remain open.
- Current regression suite: 92 passed
  (`validation/pytest_ood_guided.xml`).
- Release artifact: `dist/neural_causal_decompiler-0.9.0-py3-none-any.whl`,
  SHA-256 `123a35a2de56d95ae6fac4d4a83554e66f66de8bd0f6d409b283a72c15356516`.
  `validation/wheel_v9_run/status.json` records source/wheel/install module
  parity plus a fresh isolated quick run and complete regeneration replay.

## 0.10.0 structured mechanism extraction milestone

- Added a teacher-only structured symbolic search with one unary operator per
  parent, hierarchical interactions, deterministic four-fold validation, and
  bounded beam search.
- Added paired oracle-graph mechanism experiments and complete replay across
  ID, function, noise, scale, and intervention environments for 3/5/8 nodes.
- Formal seeds 1793/1794 each covered 30 worlds. Pooled neural-teacher NMSE fell
  from 0.111672 to 0.006023, operator exactness rose from 0.504 to 0.668, and
  nonconstant atoms fell from 2.261 to 0.849. The preregistered rule passed.
- The result isolates equation extraction with an oracle DAG. End-to-end graph
  and SCM recovery, exact internal numeric recovery, and general causal
  identification remain open.
- Current regression suite: 94 passed
  (`validation/pytest_structured_mechanism.xml`).
- Release artifact: `dist/neural_causal_decompiler-0.10.0-py3-none-any.whl`,
  SHA-256 `1c86d6624e99309e0ba23f765f33b6162ba17b63b1fff6c97c3a53e9e6b929cb`.
  `validation/wheel_v10_run/status.json` records source/wheel/install parity,
  a fresh isolated quick mechanism-search run, and complete replay.

## 0.11.0 calibrated graph decoder milestone

- Added swap-symmetric edge-presence thresholding with deterministic acyclic
  projection while preserving the historical decoder.
- Added development-only per-size calibration, source hash validation, two
  frozen teacher modes, five-environment evaluation, CLI, and full replay.
- Seeds 1993/1994 covered 480 unique worlds each and replayed. Pooled exact
  graph accuracy rose from 13.91% to 15.21%, below the fixed +2 point gate;
  pair SHD improved from 3.405 to 2.664 while orientation accuracy declined.
- This is a recorded negative result. R9 and end-to-end neural-to-SCM recovery
  remain open; the next graph method must model orientation separately from
  skeleton sparsity rather than relying on a single edge threshold.
- Current regression suite: 95 passed
  (`validation/pytest_calibrated_graph_decoder.xml`).
- Release artifact: `dist/neural_causal_decompiler-0.11.0-py3-none-any.whl`,
  SHA-256 `c9bd3d1a81ee8e73e830ccc279041b78b6abfca69cbc6ccb5b906babd2b6a149`.
  `validation/wheel_v11_run/status.json` records source/wheel/install parity,
  a fresh isolated quick graph-decoder run, and complete replay.

## 0.12.0 teacher-distilled PC program milestone

- Added teacher-only selection of Fisher CI alpha and conditioning depth.
- Added executable PC programs that persist every CI query, sepset, skeleton,
  collider-oriented intermediate graph, and final Meek-closed PDAG.
- Added frozen local-tree comparison, two teacher modes, family strata, CLI,
  source hashing, and complete deterministic replay.
- Seeds 2193/2194 covered 480 new worlds each. Pooled exact teacher fidelity was
  10.21% for PC versus 19.48% for the local tree, so the fixed rule failed.
  PC truth exactness was 2.60 points higher, confirming that fidelity and causal
  correctness are separate.
- R9 remains open. The next approach must distill skeleton and orientation from
  teacher behavior separately rather than imposing a complete Fisher-PC prior.
- Current regression suite: 96 passed (`validation/pytest_distilled_pc.xml`).
- Release artifact: `dist/neural_causal_decompiler-0.12.0-py3-none-any.whl`,
  SHA-256 `f07ff5153d9dcb42dbcce443de3c1c7738143bf5c25c9ac9c4e3d59081571bf3`.
  `validation/wheel_v12_run/status.json` records source/wheel/install parity,
  a fresh isolated distilled-PC quick run, and complete replay.
## 0.13 factorized graph-program status

Implemented separate teacher-only skeleton and orientation trees, ordered-view reconciliation, acyclic execution, source-integrity checks, all-strata reporting, and full regeneration replay. Formal runs 2393/2394 replayed. The fixed acceptance rule failed: pooled exact teacher fidelity gained 0.52 percentage points and each seed contained a mode-level decline. R9 and the full project goal remain open. See `RESULTS_FACTORIZED_GRAPH_PROGRAM.md`.
The 0.13.0 wheel matched every source module and completed an isolated quick
factorized-graph run plus full replay. Evidence:
`validation/wheel_v13_run/status.json`. The wheel SHA-256 is recorded there.

## Next frozen experiment after 0.13

The saved failure audit shows that teacher-edge omissions dominate false
presences and that orientation gains do not compose into exact graphs. The next
protocol therefore changes only skeleton positive-class cost and uses fresh
confirmation seeds 2593/2594. It was frozen in
`docs/COST_SENSITIVE_FACTORIZED_GRAPH_PROTOCOL.md`; 0.13 confirmation outputs
will not be reused for fitting or selection.

## 0.14 cost-sensitive graph-program status

Formal seeds 2593/2594 and both complete replays passed. Positive skeleton costs were safe under the frozen nondecline checks, but improvement did not replicate: pooled exact teacher fidelity gained only 0.57 percentage points and source 494 retained weight 1.0. Truth accuracy declined. R9 and the full objective remain open. See `RESULTS_COST_SENSITIVE_GRAPH.md`.
The 0.14.0 wheel matched every source module and completed an isolated quick
cost-sensitive run plus full replay. Evidence:
`validation/wheel_v14_run/status.json`.

## 0.15 symmetric skeleton status

Formal seeds 2793/2794 passed complete replay. Swap-invariant decoded-skeleton supervision improved active-pair fidelity but reduced pooled exact teacher fidelity by 0.05 percentage points. R9 and the full objective remain open. Further pairwise tree refinements are not supported by the accumulated 0.13-0.15 evidence; the next graph approach must be graph-global. See `RESULTS_SYMMETRIC_SKELETON.md`.
The 0.15.0 wheel matched every source module and completed an isolated quick
symmetric-skeleton run plus full replay. Evidence:
`validation/wheel_v15_run/status.json`.

## Next frozen graph-global experiment

Source-only audit found stable mean degree but variable per-world edge counts. The
next program uses a sparse explicit pair scorer, an affine world-specific edge
count predictor, and deterministic joint top-k selection. Its protocol and new
confirmation seeds 2993/2994 are frozen in
`docs/GRAPH_GLOBAL_RANKING_PROTOCOL.md`. No 0.13-0.15 confirmation output will
be used for fitting or selection.

## 0.16 graph-global ranking status

Formal seeds 2993/2994 passed complete replay. The explicit sparse scorer plus world-level edge-count program reduced pooled exact teacher fidelity by 2.29 percentage points and truth accuracy by 3.54 points; selected sparsity also exceeded the gate. R9 remains open. After four graph-program negative results, the next work returns to R5 internal circuit-to-program alignment rather than adding another output decoder. See `RESULTS_GRAPH_GLOBAL_RANKING.md`.
The 0.16.0 wheel matched every source module and completed an isolated quick
graph-global run plus full replay. Evidence: `validation/wheel_v16_run/status.json`.

## Next frozen R5 experiment

The verified 0.6 artifacts show natural linear-probe NMSE of 0.54-0.89, so
readout error materially contaminates intervention error. The next experiment
compares linear and explicit diagonal-quadratic probes under identical rank-one
biorthogonal interventions, pair pools, teachers, and symbolic targets. Fresh
confirmation seeds 3193/3194 and the full decision rule are frozen in
`docs/QUADRATIC_READOUT_INTERVENTION_PROTOCOL.md`.

## 0.17 quadratic readout status

Formal seeds 3193/3194 passed full mask coverage and complete retraining replay. Diagonal-quadratic probes improved natural-state measurement in five of six cases but made every targeted and collateral intervention metric worse. This identifies off-manifold readout instability and leaves R5 open. See `RESULTS_QUADRATIC_READOUT.md`.
The 0.17.0 wheel matched every source module and completed an isolated quick
quadratic-readout run plus full replay. Evidence: `validation/wheel_v17_run/status.json`.
# Certified realization status (0.28 scaling)

- Exact finite realization: 12/12 active recoveries certified; 60 exact and
  finite-approximate certificates replayed; aggregate complexity curve
  90 -> 54 -> 12.
- Quantized traffic stress test: two learned networks, five natural controls,
  20 concrete states -> 8 certified minimal states; deterministic retraining
  replay passes and limited budgets retain 30 unresolved pairs.
- Continuous ReLU separation: outward-rounded branch-and-bound certificates
  cover the full bounded intervention word; all three outcomes are exercised.
  Four declared states yield five certified incompatibility edges and exact
  chromatic lower bound 3 at epsilon 0.03. At epsilon 0.3, an independently
  checked two-state candidate closes the realization interval at LB=UB=2.
- The global continuous realization upper bound over an uncountable initial
  state domain remains an explicit open problem.
- Stable relational bounds now scale through 8/32/64/128 latent dimensions,
  2/3/4/5 continuous controls, and horizons 3/5/10/20. All eight formal pair
  queries close; independent IBP leaves the four near pairs unresolved.
- Version 0.28 release: 127 tests passed; eight isolated wheel commands passed.
  SHA-256: `18343a2ead65601d11a7c2286a8083e4734e843cf0a95321c3008dc702580143`.
- Full regression: 124 passed. Isolated 0.27.0 wheel replayed finite, traffic,
  and continuous workflows. SHA-256:
  `33fd202c630b686181d4a5386e4bfe964fa400229797d6cf737f67b663a3b548`.

---

# Certified realization status (0.29 phase crossings)

- Three learned ReLU systems certify that all declared training domains cross
  three control-dependent activation boundaries.
- The sound hybrid uses relational propagation only on matching stable phases
  and records an independent-IBP fallback on every other leaf.
- All 12 far-pair ablations are separated. Near pairs close for 16D/H=1 under
  all methods and for 32D/H=2 under widest splitting; 64D/H=3 remains unresolved
  under the fixed 24-leaf budget.
- On 16D/H=1, hybrid/widest uses 46 leaves versus 70 for independent/widest.
  Greedy best-bound closes in 27 leaves there but misses the two-step threshold,
  showing that the heuristic does not dominate balanced splitting.
- Formal replay verified 3 profiles and 24 certificates. Evidence:
  RESULTS_CERTIFIED_CONTINUOUS_NONLINEAR.md and
  validation/certified_continuous_nonlinear_acceptance.json.
- Version 0.29 release: 130 tests and 10 isolated wheel commands passed.
  SHA-256: `6a097cafe7cdb96734a9c2e67f2c2dd5cf92847de669e96bff3b872afa1c721a`.

## Post-0.29 three-step budget closure

- Increasing only the 64D/H=3 leaf budget from 24 to 28 closes the best-bound
  upper certificate at 0.239735521 against threshold 0.24.
- Both widest-coordinate baselines remain unresolved at 0.261212231 under the
  same 28-leaf budget.
- Every near leaf uses independent IBP; the closure is due to partitioning and
  budget, not relational propagation.
- Evidence: RESULTS_CONTINUOUS_NONLINEAR_BUDGET_CLOSURE.md and
  validation/continuous_nonlinear_budget28_acceptance.json.

# Certified realization status (0.30 continuous regions)

- Region certificates cover left initial box x right initial box x complete
  bounded action-word box, with splits over state or action coordinates.
- A within certificate is universal over both regions and every action word.
  A robust-separation certificate stores one action word that separates every
  cross-region state pair.
- Formal profiles span 8/32/64 dimensions and horizons 3/5/10 at coordinate
  radius 0.001. Relational bounds close all three near and all three far cases.
- Independent IBP leaves all three near regions unresolved at eight leaves;
  all far regions remain robustly separated.
- Independent replay verified 3 learned models and 12 region certificates.
  Evidence: RESULTS_CERTIFIED_CONTINUOUS_REGIONS.md and
  validation/certified_continuous_regions_acceptance.json.
- Version 0.30 release: 134 tests and 12 isolated wheel commands passed.
  SHA-256: `92c2a3b66e590cd224f3bb453740dba20878744b7edf1a0fce91684c803cb6b2`.

# Certified realization status (0.31 global behavioral cover)

- The full unit state interval and square are tiled by certified response-cover
  cells; every point and every three-step continuous action word is covered.
- Complete packing certificates provide matching lower bounds.
- Exact finite-horizon cover numbers close at K=5 in 1D and K=25 in 2D.
- 30 cell upper certificates and 310 packing-pair lower certificates replay,
  for 340 verified constituent certificates.
- This is a minimal behavioral codebook, not yet a transition-closed quotient.
- Evidence: RESULTS_CERTIFIED_CONTINUOUS_COVER.md and
  validation/certified_continuous_cover_acceptance.json.
- Version 0.31 release: 138 tests and 14 isolated wheel commands passed.
  SHA-256: `4e7664edff404e47149ba94f049cd05cedb85025bd8a468491dce6646cff512f`.

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
## 原始因果逐项证明实施进度：2026-09-28

现已转回原始 R0–R13 主线。最新合并进度请见 `RESULTS_ORIGINAL_PLAN_PROGRESS.md`，不以历史有限状态最小性替代原始因果反编译目标。原始台账为 38 项，其中 2 项严格反证、36 项未解决。当前核心 0.60 的 263 项隔离回归已经通过；0.61 候选九任务生成完成，当前公共重放与 270 项回归使用工作区临时存储继续运行，随后排队续跑非平凡机制的保留区域。候选未在回归通过前合入。新增有限样本 MEC 理论及联合噪声反例均已完成独立安装包核查。
## 2026-09-28 原始证明计划 V3 阶段记录

最新进展见 [RESULTS_ORIGINAL_PLAN_PROGRESS_V3.md](RESULTS_ORIGINAL_PLAN_PROGRESS_V3.md)，机器身份记录为 `validation/original_plan_progress_20260928_v3.json`。新增两个固定高斯估计器的严格置信度反例，七项限定统计合同已得到五项证明、两项反证；原始 R0–R13 仍为 2 项反证、36 项未解决。正式核心保持 0.60.0.dev1。修复后的 0.62 完整验收、0.63 完整验收与历史非平凡机制续跑串行执行/排队，阶段文档不预先宣告后台任务通过。
## 2026-09-28 原始证明计划 V4

最新阶段记录见 [RESULTS_ORIGINAL_PLAN_PROGRESS_V4.md](RESULTS_ORIGINAL_PLAN_PROGRESS_V4.md)。四个实际图网络、三种节点规模的数学程序保真与14处固定读写映射已经完成隔离重放，源码/安装包各77项核查通过。设备标签近并列失败保留；原始台账仍为2项严格反证、36项未解决。0.62的284项完整验收通过，0.63继续验收，机制v8按冻结队列等待续跑，正式核心仍为0.60。统一图证明登记和原始因果恢复缺口继续推进。
## 2026-09-28 原始证明计划 V5

最新阶段记录见 [RESULTS_ORIGINAL_PLAN_PROGRESS_V5.md](RESULTS_ORIGINAL_PLAN_PROGRESS_V5.md)。正式核心 0.63 已通过十三任务公开重放与 292 项完整隔离回归。0.64.dev2 统一图台账预检源码/wheel 各85项通过，完整十四任务及323项回归等待机制v8结束。新忠实性边界包源码/wheel各68项检查及六证书独立重放通过；观察 CPDAG 的总体识别与有限样本分离条件明确区分。原始台账仍为2项严格反证、36项未解决；新增搜索检查点尚待独立验收。

## 2026-09-28 原始证明计划 V6

最新阶段记录见 [RESULTS_ORIGINAL_PLAN_PROGRESS_V6.md](RESULTS_ORIGINAL_PLAN_PROGRESS_V6.md)。六份 CPDAG/忠实性边界反例已接入统一 V7，源码/wheel各105项预检和公开五任务独立重放通过。正式核心仍为已完成292项回归的0.63；机制v8继续同域证明，随后串行验收0.64的323项与0.65的343项。原始38项台账仍为2项严格反证、36项未解决，未宣布整体完成。
