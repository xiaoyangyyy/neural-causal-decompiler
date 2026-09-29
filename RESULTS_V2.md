# V2 实际结果与验收

以下数字由 scripts/acceptance_v2.py 从已独立重放的工件生成。软件验收不证明完整内部算法恢复。

## 验收

- 33 项测试通过。
- 两个固定种子 91/92，均完成双变量、多变量、真实 PySR、内部干预和显式 SCM 实验。
- wheel 在隔离安装目录运行完整小规模实验，并独立重放；源码与安装包逐字节一致。
- 历史快照保留；仅验收函数有 2 处文件差异，实验代码 AST 一致。详情见 validation/acceptance_v2.json。

## 双变量独立测试

| Seed | 环境 | 网络准确率 | 程序准确率 | 程序网络一致率 |
|---|---|---:|---:|---:|
| 91 | test_id | 69.3% | 65.4% | 69.3% |
| 91 | test_function | 64.6% | 57.3% | 57.6% |
| 91 | test_noise | 68.8% | 64.8% | 71.6% |
| 91 | test_scale | 58.1% | 63.0% | 51.8% |
| 91 | test_intervention | 66.7% | 57.8% | 72.1% |
| 92 | test_id | 72.9% | 61.5% | 67.4% |
| 92 | test_function | 71.1% | 50.3% | 56.8% |
| 92 | test_noise | 72.4% | 64.1% | 64.6% |
| 92 | test_scale | 65.4% | 63.5% | 54.7% |
| 92 | test_intervention | 68.5% | 55.5% | 70.6% |

## 完整消融：ID

| Seed | 方法 | 程序准确率 | 一致率 | 复杂度 |
|---|---|---:|---:|---:|
| 91 | full | 65.4% | 69.3% | 29 |
| 91 | without_cegis | 63.3% | 71.4% | 21 |
| 91 | without_alignment | 67.4% | 72.1% | 25 |
| 91 | without_mdl | 70.3% | 69.5% | 29 |
| 91 | without_environment_augmentation | 69.8% | 77.1% | 29 |
| 91 | decision_tree | 60.2% | 70.8% | 25 |
| 91 | pysr | 66.1% | 73.2% | 22 |
| 92 | full | 61.5% | 67.4% | 27 |
| 92 | without_cegis | 62.0% | 71.9% | 29 |
| 92 | without_alignment | 67.4% | 71.9% | 36 |
| 92 | without_mdl | 61.5% | 67.4% | 27 |
| 92 | without_environment_augmentation | 59.1% | 72.7% | 27 |
| 92 | decision_tree | 56.2% | 66.1% | 25 |
| 92 | pysr | 59.4% | 69.3% | 37 |

## 内部干预

| Seed | 特征/秩 | 有效配对数 | 有效子集一致率 | 随机映射均值 |
|---|---|---:|---:|---:|
| 91 | var_log_ratio/1 | 37 | 73.0% | 0.5% |
| 92 | reserr_xy/4 | 347 | 68.0% | 0.8% |

配对中世界会重复使用；本表不是独立样本显著性检验。原始报告另含打乱目标控制、不复用世界配对及保守界。局部映射不能证明全部规则对应网络内部算法。

## 多变量 ID

| Seed | 节点数 | 边程序保真 | 宏 F1 | 整图准确率 |
|---|---:|---:|---:|---:|
| 91 | 3 | 72.2% | 0.566 | 33.3% |
| 91 | 5 | 77.9% | 0.633 | 0.0% |
| 91 | 8 | 86.6% | 0.538 | 0.0% |
| 92 | 3 | 62.5% | 0.491 | 20.8% |
| 92 | 5 | 77.5% | 0.518 | 8.3% |
| 92 | 8 | 92.4% | 0.629 | 8.3% |

## 神经机制到 SCM

| Seed | 工件 | 父集准确比例 | 符号/神经 NMSE | 干预效应 MAE |
|---|---|---:|---:|---:|
| 91 | mechanisms/n3_1_inferred | 0.0% | 0.0096 | 0.0813 |
| 91 | mechanisms/n3_1_oracle_graph_diagnostic | 100.0% | 0.0115 | 0.0552 |
| 91 | mechanisms/n3_2_inferred | 66.7% | 0.0535 | 0.0529 |
| 91 | mechanisms/n3_2_oracle_graph_diagnostic | 100.0% | 0.0272 | 0.0559 |
| 91 | mechanisms/n5_1_inferred | 60.0% | 0.0024 | 0.0331 |
| 91 | mechanisms/n5_1_oracle_graph_diagnostic | 100.0% | 0.0058 | 0.0289 |
| 91 | mechanisms/n5_2_inferred | 40.0% | 0.0132 | 0.0805 |
| 91 | mechanisms/n5_2_oracle_graph_diagnostic | 100.0% | 0.0038 | 0.0300 |
| 91 | mechanisms/n8_1_inferred | 50.0% | 0.0073 | 0.0266 |
| 91 | mechanisms/n8_1_oracle_graph_diagnostic | 100.0% | 0.0074 | 0.0126 |
| 91 | mechanisms/n8_2_inferred | 87.5% | 0.0027 | 0.0073 |
| 91 | mechanisms/n8_2_oracle_graph_diagnostic | 100.0% | 0.0008 | 0.0089 |
| 92 | mechanisms/n3_1_inferred | 66.7% | 0.0285 | 0.0732 |
| 92 | mechanisms/n3_1_oracle_graph_diagnostic | 100.0% | 0.0012 | 0.0440 |
| 92 | mechanisms/n3_2_inferred | 100.0% | 0.1602 | 0.0743 |
| 92 | mechanisms/n3_2_oracle_graph_diagnostic | 100.0% | 0.1602 | 0.0743 |
| 92 | mechanisms/n5_1_inferred | 40.0% | 0.0074 | 0.0350 |
| 92 | mechanisms/n5_1_oracle_graph_diagnostic | 100.0% | 0.0468 | 0.0369 |
| 92 | mechanisms/n5_2_inferred | 80.0% | 0.0096 | 0.0193 |
| 92 | mechanisms/n5_2_oracle_graph_diagnostic | 100.0% | 0.0096 | 0.0183 |
| 92 | mechanisms/n8_1_inferred | 87.5% | 0.0137 | 0.0214 |
| 92 | mechanisms/n8_1_oracle_graph_diagnostic | 100.0% | 0.0142 | 0.0193 |
| 92 | mechanisms/n8_2_inferred | 87.5% | 0.0058 | 0.0145 |
| 92 | mechanisms/n8_2_oracle_graph_diagnostic | 100.0% | 0.0052 | 0.0137 |

真实图诊断与推断图端到端结果分别保留；前者不能替代发现图结构。低数值误差不等于父集或算子恢复正确。

## 尚未达到的科学目标

- 短程序仍存在明显网络保真误差，不能称为完整反编译。
- 当前映射是单一统计量的局部子空间对齐，尚未覆盖整套程序的所有中间步骤。
- 多变量整图、父集及机制算子尚不能稳定准确恢复。
- 两个合成基准种子的 OOD 结果不能证明普遍因果原则，也不足以确立创新性或通用优势。
- 原始强目标仍保持开放；后续改进必须另设实验协议，不能回用这些测试世界调参。
