# 非平凡机制全域分段验证：首轮未解决

实际目标是历史 active_end_to_end_seed4993 的 observational_graph/baseline/mechanism_0.pt，父坐标为 [1,2]。checkpoint 哈希、冻结训练归一化尺度、源代码和闭域 [-1,1]^3 均由 `validation/piecewise_mechanism_protocol_v1.json` 绑定。阈值使用冻结训练输出尺度归一化后的绝对误差 ≤0.01。

首轮在 767 个树节点预算下生成并独立重新计算了每个已接受叶子的均值定理误差界。结果为 378 个已证明叶区域、6 个未解决叶区域。全部分割边界保留，未解决区域仍在整棵域树中，整体结论为 **未解决**。任务正常结束，耗时 1708.52 秒；预算耗尽不能被记录为不存在任何忠实程序的反证。

没有导出声称覆盖全域的可执行分段程序，也没有最短/MDL、正确父集、真实机制或设备舍入保证。现有证书及每个子域的检查点可用于后续续跑；改变搜索预算必须创建新的协议版本并保存本轮结果。整个树的安装包独立重放仍待安排在当前 263 项回归之后，不能同时启动另一个重任务。

证据：`runs/historical_mechanism0_piecewise_v1/certificate.json`、`verification.json`、`checkpoint.json`、`acceptance.json`，以及 `validation/piecewise_mechanism_launch_v1/launch.json`。原始完整目标仍未完成。

## 安装包重放完成补记

现有部分证书已在旧 0.59 安装包、冻结源快照及受控工作进程树中独立重算，结束于 2026-09-28 10:20:13 UTC。结论仍为 378 个闭合叶区域、6 个未解决叶区域，无全域完成声明。峰值 Job 内存 647,479,296 字节，无超时。最终交叉绑定记录为 `validation/piecewise_mechanism_acceptance_v1.json`，安装阶段为 `validation/partial_mechanism_installed_v1/launch.json`。前文“重放待安排”保留为首轮出具时的历史状态。
## 覆盖体积补记

叶区域数量不是全域进度比例。按声明盒的 Lebesgue 体积精确计算，已证明区域占 **63/512 = 12.3046875%**，未解决区域占 **449/512 = 87.6953125%**。其中三个未解决区域仍是较大的根分支，因此不能称“只有六个小区域”。这只是几何体积，不是数据分布下的成功概率，也不构成全域保证。原证书中的全部边界与未解决区域继续保留。记录：`validation/piecewise_mechanism_geometric_coverage_v1.json`；神经误差界仍由上述独立重放证据支持。