# 原始因果证明核心 0.60.0.dev1 本地发布

统一证明入口已从隔离候选合入项目核心。136 个模块与完成测试的候选、安装包字节一致；263 项完整安装包回归通过，无失败、跳过或超时。Windows Job Object 将工作进程树纳入 8 GiB 内存限额，峰值为 1,056,727,040 字节。核心修复精确二进制常量的多项式语义和实际统计前端验证，并统一旧证明包及新的有限有理多项式记录。

核心命令：

```powershell
python -m ncd prove --config validation/original_proof_registry_protocol_v2r1.json --resume
python -m ncd verify-proof runs/original_proof_registry_v2r1
python -m ncd audit-requirements runs/original_proof_registry_v2r1 --require-closed
```

最后一个命令仍须以退出码 1 返回：原始 38 项台账只有 2 项严格反证，36 项未解决。新完整网络 SSA 和实际标签平局反例已独立打包、重放；纳入统一核心记录及 R2 分数/标签契约拆分仍是下一项工作。本次发布不代表原始因果反编译目标完成。

历史证据依赖源快照 `validation/source_snapshot_059` 和原有 0.59 安装环境。旧协议继续从该快照重放，而不是在改变后的核心上冒充旧语义。300 世界确认验收也已从旧安装包与快照再次封存，输出与原验收逐项一致。旧候选验收文件保留其发布时“候选/排队”状态，新状态写入独立发布记录。可逆合入前的模块与 pyproject 元数据备份保存于 `validation/core_v60_promotion_v1_backup`，首次发布命令的解析失败在任何修改发生前退出并保留记录。

机制全域分段首轮仍是 378 个闭合叶区域、6 个未解决区域；当前安装包独立重放已排队后启动，不参与宣布完成。它只能验证现有部分证书，不能将预算耗尽转为反证或删除未解决区域。

发布证据：`validation/unified_proof_registry_release_v60.json`、`validation/core_v60_promotion_v1.json`、`validation/full_regression_v60_launch/launch.json`、`validation/original_confirmation_snapshot_reseal_v1.json`。本次仅更新本地工作区，没有向外部服务发布。
