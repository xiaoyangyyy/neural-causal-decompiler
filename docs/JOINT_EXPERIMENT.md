# 联合机制实验运行

V2 工程验收保持冻结。新增联合实验从新训练的发现网络与提取规则出发，使用预先指定的 191/192 种子。

```powershell
.\.venv-research\Scripts\python.exe -m ncd.joint_experiment run runs/my_joint --seed 191
.\.venv-research\Scripts\python.exe -m ncd.joint_experiment verify runs/my_joint
.\.venv-research\Scripts\python.exe -m ncd.joint_experiment run runs/my_joint_quick --quick
.\.venv-research\Scripts\python.exe scripts/joint_acceptance.py
```

每次运行使用空目录。最后一条命令重放固定的 runs/joint_seed191、runs/joint_seed192 并生成 RESULTS_JOINT.md。

正式配置每种子含 5,128 个世界；训练 1,800 个世界、40 epochs；规则提取 512 个世界；映射拟合 384、验证 256，每类最终测试 384。验证使用固定 seed+10000 的 alignment_fit 生成语义，身份与其他划分独立。

程序中的六个分支表达式/判断位置各对应联合正交子空间块，秩 1/2 只在验证集上选择。单变量组合与部分双变量组合参与拟合，其余双变量组合作为留出组合。训练与评估显式记录节点是否实际执行；不同分支可能不同时可达，不能将原始配对数当作有效样本数。

控制包括同秩随机正交映射（10 组）、变量块置换、打乱目标训练、未干预教师，以及相同测试配对的单变量独立拟合。保存不复用世界的配对评估、全部中间程序轨迹、模型、映射、候选验证结果、源文件、数据和哈希。

当前实验检查多个决策位置的联合干预，但不证明完整数值统计计算被恢复。对于固定阈值树，整表达式值替换与判断结果替换会产生相同输出目标；两种模式不构成两份独立证据。尚未集成多层位置选择或将联合内部评分接回程序合成。

下一步应针对表达式内部算子及中间数值设计可区分的干预任务，用新世界评估。不能在已观察的 191/192 测试数据上反复挑选策略并继续称独立确认。
