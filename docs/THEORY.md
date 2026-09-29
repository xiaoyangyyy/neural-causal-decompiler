# 对象、可识别性和反编译理论边界

## 四个对象与三个不同任务

真实 SCM 为 M=(G,F,U)，D 由 M 采样。因果发现网络 Nθ 把整个数据集 D 映射到图或图的等价类。发现程序 P 近似 Nθ(D)，而每个神经机制模型 Qj 把父变量值映射到子变量条件均值。机制表达式 Ej 近似冻结 Qj 的输出。

因此 P 不是 M。由 P 得到图，再从机制网络蒸馏 E，才构成显式估计 SCM。实现中图、神经机制与符号机制分别保存，所有符号监督来源标记。真实父集诊断与推断图端到端结果严格分列。

## 行为 fidelity

给定数据集分布 Ω，FΩ(P,N)=Pr[D~Ω](P(D)=N(D))。当 FΩ>=1-ε，称其在 Ω 下 ε-faithful。这是相对于指定分布的性质，不是所有可能输入上的语义等价。

对于冻结且未使用测试集选取的 P，在 m 个独立、取值位于 [0,1] 的世界成功率单位上，Hoeffding 不等式给出置信度 1-δ 的下界：

F >= empirical_mean - sqrt(log(1/δ)/(2m))。

双变量一个世界贡献一个成功指标；多变量一个世界贡献该世界局部边的平均成功率，不能把同图内每条边当作独立样本。固定分层族的独立世界对其对应混合权重的期望平均值适用该界；不能任意把它外推到别的混合权重。

## 干预 fidelity

指定低层映射 τ、允许的神经干预 IN、高层程序干预 IP 和源/目标世界分布。比较 τ(N^IN(D)) 与 P^IP(D)，或比较两者相对基线的输出变化。

“存在一次干预一致”不足以证明 causal abstraction。映射、允许干预族和选择协议必须明确；应在独立源/目标上检查，加入同维随机子空间与标签置换控制，并报告有意义的输出变化子集。重复使用世界形成的多组 pair 并不独立，不能直接套用上述独立样本界。

分布式对齐仅优化映射，保持原神经网络参数不变；修改网络以符合预定程序属于另一种训练任务，不能冒充恢复已有机制。结果若只支持部分变量，按部分对齐报告。

## 反编译等价类

[P]Ω={Q: 对 Ω 支持集中的每个 D，Q(D)=P(D)}。可在类中选择描述长度最小者；最短表示不保证唯一，长度还依赖指定语言和编码。

本项目对数学表达式的受限代数子集使用 SymPy 的精确有理常量进行展开/因式分解等价判断；对有限探针上的输出一致只称经验等价，不称证明。保护性除法与普通除法语义不同，因此不参与普通代数等价证明。通用程序最短化、可识别唯一反编译均没有在本项目中获得一般性定理。

## 因果识别与图

在有独立噪声、无隐藏混杂、无环、具体函数/噪声族假设时，某些方向可识别；“知道生成方向”本身不是识别证明。线性高斯观察任务输出 CPDAG，保留等价类中未定向边。双变量 Unknown 是该基准约定，不是有限数据自动证明不可识别。

PC 式 collider 定向需要：x-z-y 为无盾三元组，且 z 不在已知的 x,y 分离集中。没有分离集证据时不定向。多变量实现用独立的 d-separation oracle 测试图语义，oracle 不向神经预测或程序提取提供信息。Fisher partial-correlation CI 只作为带高斯假设的 baseline/统计特征，不能宣称是一般非线性 CI 检验。

## 什么才算候选因果原则

高行为 fidelity 只能表明模仿网络。候选因果原则还需要在适用假设内跨函数、噪声、尺度与干预环境保持正确性；在不满足假设的环境失败不能自动归因为捷径。联合错误和主动参数攻击能支持发现某些错误行为，不单独确定唯一错误原因。

本项目会保留不能支持强科学主张的结果，不以软件测试通过替代上述证据。

## 原始参考

- Geiger et al., [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)
- Geiger et al., [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://arxiv.org/abs/2303.02536)
- [Demystifying amortized causal discovery with transformers](https://openreview.net/pdf?id=CJg9Jyr4ZE)
- [causal-learn UCSepset implementation](https://github.com/py-why/causal-learn/blob/main/causallearn/utils/PCUtils/UCSepset.py)
