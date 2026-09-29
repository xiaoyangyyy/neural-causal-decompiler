from pathlib import Path
import json
root=Path('.');v=json.loads((root/'validation/wheel_v5_run/status.json').read_text());assert v['state']=='verified' and all(x['returncode']==0 for x in v['commands'])
p=root/'WORK_STATUS.md';s=p.read_text(encoding='utf-8').replace('- 最新全套测试：83 passed，validation/pytest_raw_numeric.xml。','- 最新全套测试：85 passed，validation/pytest_regression_numeric.xml。')
s+='''
## 0.5.0 交付
- 统一 CLI 新增 dependence-numeric / verify-dependence-numeric 与 regression-numeric / verify-regression-numeric。
- dist/neural_causal_decompiler-0.5.0-py3-none-any.whl，SHA-256 `80f157da8ac23b4edfdf5151334ff47a1c6def83a451818b0320592f0c9a32af`。
- validation/wheel_v5_run/status.json：隔离安装包分别实际运行 dependence/regression quick 实验并完整重放，4 条命令退出 0；源码、wheel、安装模块哈希一致。
- 当前完整 raw CDIR 计算已可执行和分层干预，但冻结教师的数值交换与行为保真仍低；下一步应改进非线性、多位置或非正交映射并使用新预注册种子验证。
''';p.write_text(s,encoding='utf-8')
p=root/'docs/FULL_REQUIREMENTS.md';s=p.read_text(encoding='utf-8').replace('0.4.0 新增原始统计干预并从隔离安装包运行/重放（validation/wheel_v4_run/status.json）；CNN/robot','0.4.0 原始统计干预及 0.5.0 dependence/regression 内部干预均从隔离安装包运行/重放（validation/wheel_v4_run/status.json、validation/wheel_v5_run/status.json）；CNN/robot');p.write_text(s,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8');s+='\nVersion 0.5.0 isolated-wheel acceptance: `validation/wheel_v5_run/status.json`.\n';p.write_text(s,encoding='utf-8')
print(json.dumps({'version':v['version'],'wheel_sha256':v['wheel_sha256'],'commands':len(v['commands'])}))
