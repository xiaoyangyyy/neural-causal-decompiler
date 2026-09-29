from pathlib import Path
import json
root=Path('.')
s=json.loads((root/'validation/wheel_v3_full/status.json').read_text(encoding='utf-8'))
assert s['state']=='verified' and len(s['commands'])==2 and all(c['returncode']==0 for c in s['commands'])
p=root/'WORK_STATUS.md';t=p.read_text(encoding='utf-8');t=t.replace('- 验收范围为新增四类实验 quick 集成；旧 run-all 的新版安装包重新运行尚未做，旧版证据仍见 acceptance_v2.json。','- 新版安装包原有 run-all 与 verify-all 也已实际运行通过，包含真实 PySR；证据 validation/wheel_v3_full/status.json，会话 70302 正常退出 0。所有安装包运行均为 quick 集成，正式科学实验仍以各历史冻结结果为准。');t+='''
## 图失败分解
- scripts/audit_graph_failures.py 校验输入 manifest 哈希后，将两个种子、三种节点数、五个环境、两种模型、两个家族分成 120 个分层。
- RESULTS_GRAPH_FAILURES.md 与 validation/relational_failure_audit.json：漏边、多边、反向、定向/无向混淆、类混淆矩阵、全边/活跃边/整图保真。
- 493 的 8 节点非线性关系模型：全边保真 93.1%，活跃边保真 69.3%，整图保真 14.5%。不能用全边一致率替代图级恢复。
- 493 的 3 节点高斯关系模型：程序全边保真仅 33.3%；应分别研究稀有方向未定类别和家族混合，而非只提升总体精度。
- 这些是冻结测试集上的探索性诊断，不能再把它们用作后续方法的独立确认集。
''';p.write_text(t,encoding='utf-8')
p=root/'README.md';t=p.read_text(encoding='utf-8').replace('The legacy `run-all` workflow retains its\n0.2.0 verification evidence and has not yet been rerun from this new wheel.','The legacy `run-all` workflow and `verify-all` also passed a fresh quick run\nfrom this wheel, including real PySR: `validation/wheel_v3_full/status.json`.');t+='\nFrozen graph error diagnostics: [RESULTS_GRAPH_FAILURES.md](RESULTS_GRAPH_FAILURES.md).\n';p.write_text(t,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';t=p.read_text(encoding='utf-8').replace('旧 run-all 尚未在新版包重新运行','原有 run-all/verify-all 也已从新版包实际运行通过（validation/wheel_v3_full/status.json）');p.write_text(t,encoding='utf-8')
print('Updated release verification and graph failure evidence.')
