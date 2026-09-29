# 忠实性与观察 CPDAG 恢复边界：验收结果

新增六份限定反例证书已经从独立安装包重放：3、5、8 节点分别覆盖无忠实性时的总体不可识别，以及满足忠实性却没有分离条件时的有限样本一致恢复不可能性。数学推导见 [方法与定理](docs/FAITHFULNESS_BOUNDARY_METHOD.md)。

默认 96 个观察样本、目标错误不超过 1% 时，两份忠实 Gaussian SCM 构成的对照证明：任何只读观察数据的估计器，在至少一个模型上的错误概率 ≥751/1600，约 46.94%。结构边的绝对系数均超过 1/2。总体有不同分布，因此该结果没有把总体不可识别和有限样本恢复混为一谈。

源码与独立 wheel 各 68 项检查通过；公开生成、单独独立重放及复制后的可移植重放全部通过。验证过程不训练神经网络。256 MiB 的全进程资源限制下，记录峰值约 43.5 MiB，进程回收后活动子进程为零。

范围严格限制为理想实数 SCM 和 Gaussian 概率律、未知观察尺度、一个正确 CPDAG 的纯观察恢复。它不反证有充分干预信息的算法、候选集合或拒绝决策，不宣称某个 binary64/PRNG 实例精确产生这些参数，也不推翻随机世界平均或 99% 统计验收。

证据入口：

- `validation/faithfulness_boundary_acceptance_v1.json`：证据哈希与独立安装验收。
- `runs/faithfulness_boundary_v1/scoped_ledger.json`：六项限定结论、依赖及未覆盖部分。
- `validation/faithfulness_boundary_installed_v1/status.json`：68 项安装核查及三次公开命令结果。
- `validation/faithfulness_boundary_installed_v1/supervision.json`：全部后代进程的资源与回收记录。

已生成的工件可重放：

```powershell
validation/faithfulness_boundary_env_v1/Scripts/python.exe -I -m faithfulness_boundary_proof verify-proof runs/faithfulness_boundary_v1
```

首次生成使用 `prove --config validation/faithfulness_boundary_protocol_v1.json`；已存在的包拒绝覆盖，历史证据保留。下一项集成工作是将本入口接入统一 ncd 的显式新修订，随后完成完整隔离回归。当前原始目标状态仍为未完成，38 个原子主张仍有 36 个未解决。
