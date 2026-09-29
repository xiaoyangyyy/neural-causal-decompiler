# 统一识别边界登记 V7

0.65.0.dev1 为新的冻结候选，源码与独立安装包各通过 105 项预检。公开五任务生成、独立 `verify-proof` 与严格 `audit-requirements --require-closed` 均已实际执行。完整十五任务和 343 项回归已排队；当前正式核心仍是完成 292 项完整验收的 0.63.0.dev1。

## 新入口的范围

`observational_cpdag_boundary` 后端重放 [忠实性边界的六份证书](FAITHFULNESS_BOUNDARY_METHOD.md)，3、5、8 节点各包含无忠实性时的总体不可识别，以及有忠实性但没有分离条件时的有限样本一致恢复反例。信息范围仅有 IID 观察数据，估计器不知道真实尺度和结构方程，也未接收识别干预。

限定反例逐项进入 `scoped_identification_contracts`。总体不可识别、有限样本最坏恢复概率、独立随机世界的评估置信度、神经程序保真和真实干预识别均为不同对象，不能互换。原始 R9 包含允许候选集合的观察处理和真实干预分支，新的纯观察单 CPDAG 反例不关闭它。

## 独立核查

新后端在单独安装的 `faithfulness_boundary_proof` 环境中启动。核心首先检查批准的源快照、三份模块的完整字节、已有验收记录和证据哈希；工作进程再次核对实际安装路径与版本，并重新计算全部六项证书的结构输运、协方差、图等价类、条件独立、KL、干预均值和严格错误下界。

每条已验收限定合同的完整规范 JSON 指纹固定，改写样本数、下界、信息条件、证书身份、对象类别或原始闭合标记均不能通过。六项不能遗漏或重复。任务预算不足时保留六份未解决合同，严格审计公开列出它们，不能把未完成验证当成反证。

统一任务显示的 `verified` 是集合工件重放状态；其中六条科学结论分别为 `refuted`。原始 38 项结论仍为 0 证明、2 严格反证、36 未解决，严格审计退出 1。

旧 schema 1–6 仍派发到绑定版本的独立环境；新增 schema6 通过 0.64.0.dev2 的 140 模块源快照重放，目录寻找依赖批准清单的精确身份。源快照提供语义身份，不能充当尚未完成的完整回归验收。

## 已完成与待完成的验收

新预检覆盖上下文与域篡改、修改后的错误下界、伪造证书、原始主张提升、遗漏任务、六个超时域的保留、严格审计和外部工作目录下的旧 schema6 拒绝行为。预检资源包括全部后代进程；8 GiB 限制下峰值约 788 MiB，结束后活动子进程为零。

已完成的公开五任务预检入口：

```powershell
validation/wheel_v65_env/Scripts/python.exe -I -m ncd verify-proof runs/original_identification_registry_public_preflight_v65
validation/wheel_v65_env/Scripts/python.exe -I -m ncd audit-requirements runs/original_identification_registry_public_preflight_v65 --require-closed
```

后一个命令预期退出 1，因为原始主张尚未闭合。完整阶段由 `validation/run_registry_v65_stage_v1.py` 串行执行，先等待机制 v8 和 0.64 十四任务验收；预算和原第一阶段截止时间不重置。任何失败或超预算均保留为未解决。
