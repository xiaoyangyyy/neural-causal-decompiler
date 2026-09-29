# 统一证明入口 V6：实际图程序登记

这是 0.64.0.dev2 的冻结候选接口说明。当前正式核心仍为已通过 292 项完整隔离回归的 0.63.0.dev1。0.64 的源码与安装包各通过 85 项预检；十四任务和 323 项完整回归已排队，尚未验收完成。

V6 新增 `actual_graph_ssa` 后端，将四个确认网络的 3、5、8 节点证书接入 `ncd prove`、`ncd verify-proof` 和严格原始要求审计。验证进程从独立图证明安装环境加载批准的八个模块，逐字节匹配冻结源清单及已接受的图证明记录，然后重新验证实际权重、程序和固定内部映射。

每个程序记录精确的 checkpoint、网络、显式程序和证书哈希。十二个数学特征域进入独立的 `scoped_program_realization_contracts`，原始 38 项台账不删减。超时或验证未完成的图域必须保留为未解决；不能丢弃困难子域，也不能将数学等价提升为设备舍入或因果真值恢复。完整域与映射条件见 [实际图程序方法](ACTUAL_GRAPH_PROGRAM_METHOD.md)。

严格审计同时列出原始未解决项、限定统计合同、限定程序合同和未完成任务。当前原始结论仍为 0 证明、2 严格反证、36 未解决，因此 `--require-closed` 退出 1。历史失败搜索本身不会永远否决后来取得同范围有效证明的主张；实际闭合仍要求对应原始量词的证据。

旧 schema 1–5 的重放经过固定版本与批准源清单派发。V6 修订加入从工件路径、协议项目根和安装位置寻找原始项目的逻辑，候选位置必须匹配已批准的源清单哈希。工作目录名相同、伪造目录或缺失记录均不能通过。因此，从任意工作目录发起旧证书重放也不会偷偷采用当前源语义。

初版 0.64.0.dev1 的安装预检有两个外部工作目录失败，保留在 `validation/wheel_v64_preflight_v1`。修复通过新版本 0.64.0.dev2 和新冻结协议登记，旧版 wheel、源快照及失败日志保留。

已完成的四任务公开预检包含原始主线、两项精确多项式和实际图程序；该预检从外部工作目录运行，独立 verify 与严格 audit 均检查通过。它不替代完整十四任务验收。

```powershell
validation/wheel_v64r2_env/Scripts/python.exe -I -m ncd verify-proof runs/original_graph_registry_public_preflight_v64r2
validation/wheel_v64r2_env/Scripts/python.exe -I -m ncd audit-requirements runs/original_graph_registry_public_preflight_v64r2 --require-closed
```

完整阶段由 `validation/run_registry_v64r2_stage_v1.py` 自动串行执行。它先等待历史机制 v8 结束，并确认真实重任务进程已退出；原第一阶段截止时间、2 个训练线程、8 GiB 后代总内存与 8 GiB 新工件预算保留。生产源只有在完整隔离验收通过后才能更新。
