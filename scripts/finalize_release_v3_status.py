from pathlib import Path
import json
r=json.loads(Path('validation/wheel_v3_run/status.json').read_text(encoding='utf-8'))
assert r['state']=='verified' and len(r['commands'])==8 and all(c['returncode']==0 for c in r['commands'])
p=Path('WORK_STATUS.md');s=p.read_text(encoding='utf-8').replace('- 验证正在执行，会话 92071。工件及日志位于 validation/wheel_v3_run；继续时先轮询同一会话，不重复启动。','- 隔离安装验证完成：四类新增实验各实际运行并独立重放，8 条命令全部退出 0。证据 validation/wheel_v3_run/status.json；会话 92071 已正常结束。\n- 验收范围为新增四类实验 quick 集成；旧 run-all 的新版安装包重新运行尚未做，旧版证据仍见 acceptance_v2.json。');p.write_text(s,encoding='utf-8')
p=Path('docs/FULL_REQUIREMENTS.md');s=p.read_text(encoding='utf-8').replace('隔离安装包验证进行中','四类新增实验均已在隔离安装包实际运行并重放（validation/wheel_v3_run/status.json），旧 run-all 尚未在新版包重新运行');p.write_text(s,encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');s+='\nRelease validation: 59 tests passed; all four added quick experiments and their\nindependent replays passed using the isolated 0.3.0 installation. Evidence:\n`validation/wheel_v3_run/status.json`. The legacy `run-all` workflow retains its\n0.2.0 verification evidence and has not yet been rerun from this new wheel.\n';p.write_text(s,encoding='utf-8')
print('Recorded eight successful installed CLI commands and explicit remaining scope.')
