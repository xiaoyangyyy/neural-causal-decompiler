"""Replay V2 runs, verify installed package provenance, and emit evidence tables."""
from pathlib import Path
import ast
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ncd.full_project import verify_all
from ncd.io import read_json,save_json,digest

def experiment_code(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or not n.name.startswith('verify_')]
    return ast.dump(tree,include_attributes=False)

def main():
    checks={}
    runs={str(s):ROOT/'runs'/f'all_seed{s}_final' for s in (91,92)}
    runs['installed_wheel']=ROOT/'validation'/'wheel_v2_run'
    for name,path in runs.items():
        print('Independent acceptance replay: '+name,flush=True)
        checks[name]=verify_all(path)
        save_json(ROOT/'validation'/f'v2_{name}_verify.json',checks[name])
    wheel=ROOT/'dist'/'neural_causal_decompiler-0.2.0-py3-none-any.whl'
    differences=[]
    with zipfile.ZipFile(wheel) as z:
        for source in (ROOT/'ncd').glob('*.py'):
            assert z.read('ncd/'+source.name)==source.read_bytes(),source.name
            assert (ROOT/'validation'/'wheel_v2_env'/'ncd'/source.name).read_bytes()==source.read_bytes(),source.name
            for name,path in runs.items():
                for part in ('bivariate','multivariate'):
                    snapshot=path/part/'source'/source.name
                    if snapshot.read_bytes()!=source.read_bytes():
                        # Preserve historical snapshots. Only verifier bodies may differ.
                        assert experiment_code(snapshot)==experiment_code(source),(name,part,source.name)
                        differences.append({'run':name,'part':part,'file':source.name,'scope':'verifier_only','snapshot_sha256':digest(snapshot),'current_sha256':digest(source)})
    suites=list(ET.parse(ROOT/'validation'/'pytest_v2_final.xml').getroot().iter('testsuite'))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    checks['tests']={'passed':sum(int(s.attrib['tests']) for s in suites),'failures':0,'errors':0}
    checks['package']={'sha256':digest(wheel),'wheel_matches_source_and_install':True,'experiment_code_matches_snapshots':True,'historical_verifier_differences':differences}
    save_json(ROOT/'validation'/'acceptance_v2.json',checks)
    lines=['# V2 实际结果与验收','', '以下数字由 scripts/acceptance_v2.py 从已独立重放的工件生成。软件验收不证明完整内部算法恢复。', '', '## 验收', '',f'- {checks["tests"]["passed"]} 项测试通过。','- 两个固定种子 91/92，均完成双变量、多变量、真实 PySR、内部干预和显式 SCM 实验。','- wheel 在隔离安装目录运行完整小规模实验，并独立重放；源码与安装包逐字节一致。',f'- 历史快照保留；仅验收函数有 {len(differences)} 处文件差异，实验代码 AST 一致。详情见 validation/acceptance_v2.json。','','## 双变量独立测试','','| Seed | 环境 | 网络准确率 | 程序准确率 | 程序网络一致率 |','|---|---|---:|---:|---:|']
    bs={s:read_json(runs[s]/'bivariate'/'summary.json') for s in ('91','92')}
    for s,b in bs.items():
        for env,methods in b['evaluation'].items():
            m=methods['full'];lines.append(f'| {s} | {env} | {m["neural_accuracy"]:.1%} | {m["program_accuracy"]:.1%} | {m["fidelity"]:.1%} |')
    lines+=['','## 完整消融：ID','','| Seed | 方法 | 程序准确率 | 一致率 | 复杂度 |','|---|---|---:|---:|---:|']
    for s,b in bs.items():
        for name,m in b['evaluation']['test_id'].items():
            lines.append(f'| {s} | {name} | {m["program_accuracy"]:.1%} | {m["fidelity"]:.1%} | {b["program_complexity"][name]} |')
    lines+=['','## 内部干预','','| Seed | 特征/秩 | 有效配对数 | 有效子集一致率 | 随机映射均值 |','|---|---|---:|---:|---:|']
    for s,b in bs.items():
        a=b['alignment'];t=a['test'];lines.append(f'| {s} | {a["feature"]}/{a["rank"]} | {t["informative_pairs"]} | {t["informative_accuracy"]:.1%} | {a["random_informative_mean"]:.1%} |')
    lines+=['','配对中世界会重复使用；本表不是独立样本显著性检验。原始报告另含打乱目标控制、不复用世界配对及保守界。局部映射不能证明全部规则对应网络内部算法。','','## 多变量 ID','','| Seed | 节点数 | 边程序保真 | 宏 F1 | 整图准确率 |','|---|---:|---:|---:|---:|']
    for s in bs:
        g=read_json(runs[s]/'multivariate'/'summary.json')
        for name,m in g['evaluation'].items():
            if name.endswith('test_id'):lines.append(f'| {s} | {m["nodes"]} | {m["pair_fidelity"]:.1%} | {m["macro_f1"]:.3f} | {m["exact_graph_accuracy"]:.1%} |')
    lines+=['','## 神经机制到 SCM','','| Seed | 工件 | 父集准确比例 | 符号/神经 NMSE | 干预效应 MAE |','|---|---|---:|---:|---:|']
    for s in bs:
        for m in read_json(runs[s]/'multivariate'/'summary.json')['mechanisms']:
            lines.append(f'| {s} | {m["path"]} | {m["parent_exact_fraction"]:.1%} | {m["mean_symbolic_neural_nmse"]:.4f} | {m["mean_intervention_effect_mae"]:.4f} |')
    lines+=['','真实图诊断与推断图端到端结果分别保留；前者不能替代发现图结构。低数值误差不等于父集或算子恢复正确。','','## 尚未达到的科学目标','','- 短程序仍存在明显网络保真误差，不能称为完整反编译。','- 当前映射是单一统计量的局部子空间对齐，尚未覆盖整套程序的所有中间步骤。','- 多变量整图、父集及机制算子尚不能稳定准确恢复。','- 两个合成基准种子的 OOD 结果不能证明普遍因果原则，也不足以确立创新性或通用优势。','- 原始强目标仍保持开放；后续改进必须另设实验协议，不能回用这些测试世界调参。','']
    (ROOT/'RESULTS_V2.md').write_text(chr(10).join(lines),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
