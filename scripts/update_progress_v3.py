from pathlib import Path
p=Path('docs/FULL_REQUIREMENTS.md');s=p.read_text(encoding='utf-8');s=s.replace('整图恢复有明显失败 |','整图恢复有明显失败；显式关系注意力 493/494 已独立重放，整图改善不稳定（RESULTS_RELATIONAL.md） |');s=s.replace('聚合验收见 acceptance_v2.json；CNN/robot','历史聚合验收见 acceptance_v2.json；0.3.0 已接入 joint/numeric/guided/relational 及独立重放入口，隔离安装包验证进行中；CNN/robot');p.write_text(s,encoding='utf-8')
Path('WORK_STATUS.md').write_text('''# 当前进度

原始完整目标仍未完成；范围见 docs/FULL_REQUIREMENTS.md。历史工件保持原样。

## 已验收
- V2 91/92：RESULTS_V2.md。
- 联合内部干预 191/192：RESULTS_JOINT.md。
- 多位置数值审计 293/294：RESULTS_NUMERIC.md。
- 内部评分参与选择 393/394：RESULTS_GUIDED.md；最终程序选择未改变。
- 显式边关系注意力 493/494：RESULTS_RELATIONAL.md；validation/relational_acceptance.json。
  每组 2,976 个世界、18 次机制恢复；独立重放进程 29345 已退出 0。
  原实验进程 50979/99525 句柄已不存在；不应重启这些实验。
  整图表现变化有升有降，不能宣称稳定改善。
- 最新全套测试：59 passed，validation/pytest_cli_integration.xml。

## 本轮交付整合
- ncd CLI 新增 joint/numeric/guided/relational 和四个 verify 命令。
- README 记录命令及依赖关系；run-all 保留原双变量与多变量范围。
- 0.3.0 wheel 已构建并安装到 validation/wheel_v3_env。
- scripts/acceptance_release_v3.py 核对源码、wheel、安装目录字节一致，确认导入安装包，实际运行四类 quick 实验及重放。
- 验证正在执行，会话 92071。工件及日志位于 validation/wheel_v3_run；继续时先轮询同一会话，不重复启动。

## 后续必须推进
数值统计计算的完整恢复、干预对齐覆盖、程序保真、图/父集/算子恢复仍不足。
关系偏置实验没有解决这些目标，需要依据失败分析改进计算与程序表达能力。
安装验证只证明交付可运行；不能替代科学结论。
''',encoding='utf-8')
