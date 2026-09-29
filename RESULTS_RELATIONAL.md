# 显式边关系注意力：实际对照结果

协议见 docs/RELATIONAL_PROTOCOL.md。两个模型共享世界、训练预算、参数分配和解码。关系开关会改变有效自由度，不把它表述为完全相同容量。

全套 59 项测试通过；两个正式实验独立重放通过。

## ID 神经图（无环投影后）

| Seed | 节点数 | 模型 | 宏 F1 | 骨架 F1 | 方向准确率 | 整图准确率 | pair SHD |
|---|---:|---|---:|---:|---:|---:|---:|
| 493 | 3 | without_relations | 0.819 | 0.911 | 78.9% | 56.2% | 0.562 |
| 493 | 3 | with_relations | 0.800 | 0.901 | 78.3% | 53.1% | 0.625 |
| 493 | 5 | without_relations | 0.729 | 0.818 | 70.7% | 26.0% | 1.604 |
| 493 | 5 | with_relations | 0.733 | 0.827 | 74.7% | 26.0% | 1.542 |
| 493 | 8 | without_relations | 0.737 | 0.809 | 69.4% | 6.2% | 2.823 |
| 493 | 8 | with_relations | 0.743 | 0.792 | 71.8% | 3.1% | 3.010 |
| 494 | 3 | without_relations | 0.751 | 0.891 | 80.3% | 44.8% | 0.729 |
| 494 | 3 | with_relations | 0.791 | 0.912 | 84.3% | 51.0% | 0.635 |
| 494 | 5 | without_relations | 0.748 | 0.816 | 71.0% | 11.5% | 1.938 |
| 494 | 5 | with_relations | 0.748 | 0.825 | 71.6% | 8.3% | 1.844 |
| 494 | 8 | without_relations | 0.739 | 0.775 | 70.0% | 7.3% | 3.292 |
| 494 | 8 | with_relations | 0.758 | 0.802 | 72.6% | 6.2% | 2.958 |

## 同一测试世界的整图胜负

| Seed | 节点数 | 仅关系模型正确 | 仅基线正确 | 整图准确率差 | 保守 95% 区间 |
|---|---:|---:|---:|---:|---|
| 493 | 3 | 9 | 12 | -3.1% | [-30.8%, +24.6%] |
| 493 | 5 | 6 | 6 | +0.0% | [-27.7%, +27.7%] |
| 493 | 8 | 1 | 4 | -3.1% | [-30.8%, +24.6%] |
| 494 | 3 | 14 | 8 | +6.2% | [-21.5%, +34.0%] |
| 494 | 5 | 3 | 6 | -3.1% | [-30.8%, +24.6%] |
| 494 | 8 | 1 | 2 | -1.0% | [-28.8%, +26.7%] |

区间以世界为单位，适用于该固定基准混合分布；没有把同图内边当作独立样本。区间未对多个节点数/种子的同时比较作多重校正，不能用它们挑选唯一获益子组。

## 神经图、符号图与投影分开（ID）

| Seed | 节点数 | 模型 | 神经原始整图 | 神经投影整图 | 符号投影整图 | 程序保真 | 神经删环边数 | 符号删环边数 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 493 | 3 | without_relations | 56.2% | 56.2% | 25.0% | 77.4% | 0 | 0 |
| 493 | 3 | with_relations | 53.1% | 53.1% | 25.0% | 67.4% | 0 | 0 |
| 493 | 5 | without_relations | 26.0% | 26.0% | 22.9% | 88.3% | 0 | 0 |
| 493 | 5 | with_relations | 26.0% | 26.0% | 22.9% | 89.4% | 0 | 0 |
| 493 | 8 | without_relations | 6.2% | 6.2% | 6.2% | 92.0% | 0 | 0 |
| 493 | 8 | with_relations | 3.1% | 3.1% | 6.2% | 92.1% | 0 | 0 |
| 494 | 3 | without_relations | 44.8% | 44.8% | 36.5% | 78.1% | 0 | 0 |
| 494 | 3 | with_relations | 51.0% | 51.0% | 27.1% | 68.1% | 0 | 0 |
| 494 | 5 | without_relations | 11.5% | 11.5% | 11.5% | 82.9% | 1 | 0 |
| 494 | 5 | with_relations | 8.3% | 8.3% | 11.5% | 82.8% | 2 | 0 |
| 494 | 8 | without_relations | 7.3% | 7.3% | 0.0% | 89.7% | 1 | 0 |
| 494 | 8 | with_relations | 6.2% | 6.2% | 3.1% | 90.2% | 0 | 0 |

## OOD 神经图

| Seed | 环境 | 节点数 | 模型 | 宏 F1 | 方向准确率 | 整图准确率 |
|---|---|---:|---|---:|---:|---:|
| 493 | test_function | 3 | without_relations | 0.584 | 40.4% | 20.8% |
| 493 | test_function | 3 | with_relations | 0.582 | 43.2% | 20.8% |
| 493 | test_function | 5 | without_relations | 0.614 | 36.2% | 10.4% |
| 493 | test_function | 5 | with_relations | 0.636 | 42.0% | 8.3% |
| 493 | test_function | 8 | without_relations | 0.589 | 30.5% | 1.0% |
| 493 | test_function | 8 | with_relations | 0.627 | 37.8% | 2.1% |
| 493 | test_noise | 3 | without_relations | 0.685 | 65.6% | 30.2% |
| 493 | test_noise | 3 | with_relations | 0.688 | 66.9% | 32.3% |
| 493 | test_noise | 5 | without_relations | 0.741 | 63.9% | 21.9% |
| 493 | test_noise | 5 | with_relations | 0.745 | 65.9% | 16.7% |
| 493 | test_noise | 8 | without_relations | 0.750 | 63.6% | 4.2% |
| 493 | test_noise | 8 | with_relations | 0.785 | 69.8% | 5.2% |
| 493 | test_scale | 3 | without_relations | 0.434 | 60.0% | 11.5% |
| 493 | test_scale | 3 | with_relations | 0.410 | 60.0% | 8.3% |
| 493 | test_scale | 5 | without_relations | 0.420 | 50.3% | 1.0% |
| 493 | test_scale | 5 | with_relations | 0.310 | 51.3% | 0.0% |
| 493 | test_scale | 8 | without_relations | 0.334 | 53.2% | 0.0% |
| 493 | test_scale | 8 | with_relations | 0.258 | 55.5% | 0.0% |
| 493 | test_intervention | 3 | without_relations | 0.393 | 21.5% | 11.5% |
| 493 | test_intervention | 3 | with_relations | 0.378 | 19.8% | 10.4% |
| 493 | test_intervention | 5 | without_relations | 0.393 | 19.3% | 4.2% |
| 493 | test_intervention | 5 | with_relations | 0.373 | 18.7% | 3.1% |
| 493 | test_intervention | 8 | without_relations | 0.421 | 19.9% | 0.0% |
| 493 | test_intervention | 8 | with_relations | 0.461 | 25.4% | 2.1% |
| 494 | test_function | 3 | without_relations | 0.613 | 43.9% | 25.0% |
| 494 | test_function | 3 | with_relations | 0.651 | 48.6% | 28.1% |
| 494 | test_function | 5 | without_relations | 0.637 | 39.4% | 11.5% |
| 494 | test_function | 5 | with_relations | 0.655 | 47.4% | 9.4% |
| 494 | test_function | 8 | without_relations | 0.561 | 33.8% | 1.0% |
| 494 | test_function | 8 | with_relations | 0.595 | 40.7% | 4.2% |
| 494 | test_noise | 3 | without_relations | 0.730 | 65.9% | 40.6% |
| 494 | test_noise | 3 | with_relations | 0.711 | 63.4% | 41.7% |
| 494 | test_noise | 5 | without_relations | 0.701 | 64.9% | 8.3% |
| 494 | test_noise | 5 | with_relations | 0.712 | 68.8% | 8.3% |
| 494 | test_noise | 8 | without_relations | 0.711 | 64.6% | 6.2% |
| 494 | test_noise | 8 | with_relations | 0.734 | 66.2% | 5.2% |
| 494 | test_scale | 3 | without_relations | 0.559 | 51.7% | 17.7% |
| 494 | test_scale | 3 | with_relations | 0.542 | 45.6% | 21.9% |
| 494 | test_scale | 5 | without_relations | 0.473 | 50.0% | 1.0% |
| 494 | test_scale | 5 | with_relations | 0.486 | 46.8% | 1.0% |
| 494 | test_scale | 8 | without_relations | 0.461 | 49.2% | 0.0% |
| 494 | test_scale | 8 | with_relations | 0.440 | 44.9% | 0.0% |
| 494 | test_intervention | 3 | without_relations | 0.392 | 16.8% | 14.6% |
| 494 | test_intervention | 3 | with_relations | 0.394 | 16.1% | 15.6% |
| 494 | test_intervention | 5 | without_relations | 0.356 | 11.6% | 3.1% |
| 494 | test_intervention | 5 | with_relations | 0.376 | 14.5% | 4.2% |
| 494 | test_intervention | 8 | without_relations | 0.365 | 12.8% | 1.0% |
| 494 | test_intervention | 8 | with_relations | 0.401 | 19.4% | 1.0% |

## 预先选定世界的机制恢复（每格两个世界的均值）

| Seed | 节点数 | 图来源 | 父集准确比例 | 符号/神经 NMSE | 符号/真值 NMSE | 干预效应 MAE |
|---|---:|---|---:|---:|---:|---:|
| 493 | 3 | without_relations | 83.3% | 0.0107 | 0.0329 | 0.0301 |
| 493 | 3 | with_relations | 83.3% | 0.0107 | 0.0329 | 0.0301 |
| 493 | 3 | oracle_graph_diagnostic | 100.0% | 0.0117 | 0.0341 | 0.0292 |
| 493 | 5 | without_relations | 60.0% | 0.0059 | 0.1115 | 0.0695 |
| 493 | 5 | with_relations | 60.0% | 0.0059 | 0.1115 | 0.0695 |
| 493 | 5 | oracle_graph_diagnostic | 100.0% | 0.0175 | 0.0361 | 0.0374 |
| 493 | 8 | without_relations | 75.0% | 0.0301 | 0.0508 | 0.0141 |
| 493 | 8 | with_relations | 43.8% | 0.0328 | 0.1239 | 0.0338 |
| 493 | 8 | oracle_graph_diagnostic | 100.0% | 0.0129 | 0.0222 | 0.0138 |
| 494 | 3 | without_relations | 33.3% | 0.0033 | 0.1088 | 0.1211 |
| 494 | 3 | with_relations | 66.7% | 0.0053 | 0.0996 | 0.0939 |
| 494 | 3 | oracle_graph_diagnostic | 100.0% | 0.0044 | 0.0152 | 0.0457 |
| 494 | 5 | without_relations | 60.0% | 0.0633 | 0.1022 | 0.0364 |
| 494 | 5 | with_relations | 60.0% | 0.0510 | 0.0773 | 0.0384 |
| 494 | 5 | oracle_graph_diagnostic | 100.0% | 0.0757 | 0.1054 | 0.0313 |
| 494 | 8 | without_relations | 62.5% | 0.0104 | 0.0435 | 0.0244 |
| 494 | 8 | with_relations | 87.5% | 0.0117 | 0.0212 | 0.0118 |
| 494 | 8 | oracle_graph_diagnostic | 93.8% | 0.0116 | 0.0208 | 0.0122 |

## 边界

- 真图机制诊断与推断图端到端恢复分列。低方程拟合误差不能替代正确父集或干预效果。
- 高斯世界以 CPDAG 为目标；机制恢复中的 DAG completion 有记录，不声称识别了不可辨识方向。
- 程序仍使用局部统计特征；更强的上下文网络不保证能由当前局部语言高保真表示。
- 新关系模型不是对历史冻结教师的反编译成功证明；旧模型和工件保持不变。
- 任何局部改善都不能替代全原始目标：完整内部算法、原始统计运算逆解以及普遍因果原则仍需独立证据。
