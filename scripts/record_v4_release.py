from pathlib import Path
import json
root=Path('.')
status=json.loads((root/'validation/wheel_v4_run/status.json').read_text())
assert status['state']=='verified' and all(x['returncode']==0 for x in status['commands'])
p=root/'WORK_STATUS.md';s=p.read_text(encoding='utf-8').replace('- 最新全套测试：59 passed，validation/pytest_cli_integration.xml。','- 最新全套测试：83 passed，validation/pytest_raw_numeric.xml。')
s+='''
## 0.4.0 交付
- 统一 CLI 新增 raw-numeric / verify-raw-numeric。
- dist/neural_causal_decompiler-0.4.0-py3-none-any.whl 已构建；SHA-256 `c2514ae0081fb1c05d4a9717c50ae81ad2cc811625a51b01fb2a1ee7e3e6e394`。
- validation/wheel_v4_env 为隔离安装；validation/wheel_v4_run/status.json 证明从安装包实际运行 quick raw-numeric 和完整重放，两条命令退出 0，源码/wheel/安装模块字节一致。
- 下一阶段：展开 dependence kernel 与 cross-fit regression 的内部运算；为 variance ratio 的重尾误差设计预先固定的稳健目标和新确认种子。
''';p.write_text(s,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';s=p.read_text(encoding='utf-8').replace('原有 run-all/verify-all 也已从新版包实际运行通过（validation/wheel_v3_full/status.json）；CNN/robot','原有 run-all/verify-all 也已从 0.3.0 包实际运行通过（validation/wheel_v3_full/status.json）；0.4.0 新增原始统计干预并从隔离安装包运行/重放（validation/wheel_v4_run/status.json）；CNN/robot');p.write_text(s,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8');s+='\nVersion 0.4.0 isolated-wheel acceptance: `validation/wheel_v4_run/status.json` (source/wheel/install parity plus fresh raw-numeric run and replay).\n';p.write_text(s,encoding='utf-8')
print('recorded 0.4 delivery and remaining scientific targets')
