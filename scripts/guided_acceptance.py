"""Independent final-test acceptance for the precommitted guided selection runs."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ncd.io import read_json,save_json
from ncd.guided_experiment import verify_guided_experiment

def main():
    suites=list(ET.parse(ROOT/"validation"/"pytest_guided.xml").getroot().iter("testsuite"))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    checks={"tests":{"passed":sum(int(s.attrib["tests"]) for s in suites)}}
    summaries={};selections={}
    for seed in (393,394):
        print(f"independent guided replay {seed}",flush=True)
        root=ROOT/"runs"/f"guided_seed{seed}"
        checks[str(seed)]=verify_guided_experiment(root)
        save_json(ROOT/"validation"/f"guided_seed{seed}_verify.json",checks[str(seed)])
        summaries[seed]=read_json(root/"summary.json")
        selections[seed]=read_json(root/"selection"/"selection.json")
    lines=["# 内部评分参与程序选择：393 / 394","","固定协议见 docs/GUIDED_PROTOCOL.md。候选程序与隐藏位置均在提取/拟合/验证世界上决定，之后才生成最终测试世界。","","## 验收","","- 全套 "+str(checks["tests"]["passed"])+" 项测试通过。","- 两个正式实验独立重放通过，包括候选重新合成、读出重新拟合、验证分数重算和最终五环境指标复核。","- 方法是有限候选重排，不是无限程序空间的全局搜索。","","## 实际选择","","| Seed | 候选数 | 获得正内部支持的候选 | 内部选择是否改变程序 |","|---|---:|---:|---|"]
    for seed,s in summaries.items():
        lines.append(f"| {seed} | {s['candidate_count']} | {s['positive_internal_support']} | {s['selection_changed']} |")
    lines+=["","## 候选的验证证据","","| Seed | 候选 ID | 复杂度 | 验证保真 | 内部支持 | 受支持位置 | 行为/MDL 分数 | 加入内部支持后的分数 |","|---|---|---:|---:|---:|---|---:|---:|"]
    for seed,s in selections.items():
        for r in s["candidates"]:
            i=r["id"]
            lines.append(f"| {seed} | {i[:12]} | {r['complexity']} | {r['validation_fidelity']:.3f} | {r['mechanism_support']:.6f} | {r['selected_site'] or '无'} | {s['base_scores'][i]:.6f} | {s['guided_scores'][i]:.6f} |")
    lines+=["","## 最终独立测试","","| Seed | 环境 | 方法 | 网络准确率 | 程序准确率 | 程序保真 |","|---|---|---|---:|---:|---:|"]
    for seed,s in summaries.items():
        for split,methods in s["evaluation"].items():
            for name,m in methods.items():
                lines.append(f"| {seed} | {split} | {name} | {m['neural_accuracy']:.1%} | {m['program_accuracy']:.1%} | {m['fidelity']:.1%} |")
    lines+=["","## 解释边界","","- 内部和无内部方法共享候选、验证数据及复杂度项；内部方法额外使用映射计算，不宣称总计算预算相同。","- 纠正后的内部支持必须超过随机和打乱目标控制；证据不足时支持为零，不宣称选出了受支持的隐藏位置。","- 选择相同、支持为零或最终测试退化均保留，不据测试结果调大内部权重或删去失败候选。","- 即使选到了不同程序，也不能据此证明原始统计运算、完整内部算法或普遍因果原则已恢复。","- 图结构、机制父集/算子恢复，以及原始数据到统计算子的完整反编译仍须继续实现和检验。",""]
    (ROOT/"RESULTS_GUIDED.md").write_text(chr(10).join(lines),encoding="utf-8")
    save_json(ROOT/"validation"/"guided_acceptance.json",checks)
    print(checks)

if __name__=="__main__":main()
