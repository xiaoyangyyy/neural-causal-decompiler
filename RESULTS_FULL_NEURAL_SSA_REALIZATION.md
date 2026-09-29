# 完整冻结网络的显式程序与组合证明

已把两个历史 Discoverer 冻结网络的全部计算转换为可独立执行的显式 SSA：teacher1193 为 37,460 条指令，discoverer91 为 37,459 条；两者各覆盖 43 个真实 FX 张量和 1,116 个可干预坐标。全部线性层展开为有理权重乘加，Tanh、均值、总体方差、平方根、归一化、截断、拼接、变量翻转和分类决策具有明确数学语义，保留两个交换分支，不再调用神经编码器或神经输出头。

独立验证器从冻结 checkpoint 重建真实 FX 图与参数，逐项检查局部关系。线性计算使用精确有理系数正规形，不通过重新运行编译器来假定等价；每条基本指令必须落入一个核查关系，所有真实张量坐标必须覆盖。通过拓扑归纳得到任意兼容干预子集下的中间状态、附带变化、四个 logits 与最终标签一致性。

证明域是全部有限实数 N×2 输入（N≥16）。每个干预坐标可来自独立源；向量源要求样本数匹配，标量源可使用不同样本数。标准差下限及其日志输入别名的写入值必须 ≥实际 1e-5 常量，否则保护性除法或日志可能改变语义，验证器拒绝纳入保证。零方差、下限相等、截断边界和分类平局包含在数学保证内。这里证明的是加载权重与实数算子定义的网络，未证明设备浮点实现的全域舍入误差。

数值诊断采用四个不同计算步骤的控制变量（原始均值、编码器 Tanh、交换分支池化平方、输出头 Tanh），每个独立来源，覆盖 16 个掩码和 3 个输入，共每网络 48 次、两个网络 96 次。每次检查全部 1,116 个中间坐标、未干预坐标的附带变化和输出；最大最终输出差分别为 5.33e-15、3.55e-15。诊断覆盖两种标准差下限分支、三种截断情况，但有限诊断不是全域数学证明的替代，也不是独立世界统计保证。

区间解释器复用有严格余项界的有理 Tanh/log 包围与有理平方根包围；条件分支按守卫细化分母下界，不能把分支边界删去。两个实际网络的同一有理输入分别被严格分类为 0 和 2。测试也覆盖未决守卫、分类平局、被干预条件节点未访问分支、证书及权重篡改、映射缺失、源形状不兼容、越过保护域与无需 Torch 的程序执行。

这是一项完整计算实现的上界，仍保留大量已学习权重作为显式常量。它没有证明短程序、MDL 最小性、唯一最小因果计算商、可读的因果发现算法、真实图或真实 SCM 正确性。固定坐标映射也不构成对所有抽象映射族的保证。原始强目标仍需独立完成，不因这个范围明确的证书自动关闭。

工件在 `runs/full_neural_ssa_realization_v1`，冻结协议为 `validation/neural_ssa_protocol_v1.json`。命令：

```powershell
python -m neural_ssa_proof prove --config validation/neural_ssa_protocol_v1.json
python -m neural_ssa_proof verify-bundle runs/full_neural_ssa_realization_v1/manifest.json
python -m neural_ssa_proof verify-proof runs/full_neural_ssa_realization_v1/teacher1193/certificate.json --program runs/full_neural_ssa_realization_v1/teacher1193/program.json
python -m neural_ssa_proof execute --program runs/full_neural_ssa_realization_v1/teacher1193/program.json --input input.json --output output.json
```

`execute --interval` 接受有理点或有理闭区间，并保留未决分类。浮点执行超出数值后端能力时明确拒绝输出全域设备保证。安装包重放与逐项源字节身份检查结果见 `validation/neural_ssa_installed_v1/status.json`；最终验收见 `validation/neural_ssa_acceptance_v1.json`。核心 0.59 与既有证明包不改动，新命名空间及协议有独立源快照清单。
