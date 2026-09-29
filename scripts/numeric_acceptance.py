"""Verify the precommitted numerical audits and report limits alongside results."""
from pathlib import Path
import sys
import numpy as np
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ncd.io import read_json,save_json
from ncd.numeric_experiment import verify_numeric

def number(value):return "N/A" if value is None else f"{value:.4f}"
def weighted(rows,category,key):
    usable=[r[category] for r in rows if r[category]["n"] and r[category][key] is not None]
    n=sum(r["n"] for r in usable)
    return sum(r[key]*r["n"] for r in usable)/n if n else None

def main():
    checks={};reports={}
    suites=list(ET.parse(ROOT/"validation"/"pytest_numeric.xml").getroot().iter("testsuite"))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    checks["tests"]={"passed":sum(int(s.attrib["tests"]) for s in suites)}
    for seed in (293,294):
        print(f"independent numerical replay {seed}",flush=True)
        root=ROOT/"runs"/f"numeric_seed{seed}";checks[str(seed)]=verify_numeric(root)
        save_json(ROOT/"validation"/f"numeric_seed{seed}_verify.json",checks[str(seed)])
        reports[seed]=(root,read_json(root/"summary.json"),read_json(root/"test_pairs.json"))
    lines=["# 多隐藏位置数值机制实验","","所有位置与控制均报告；未使用测试结果选择最优位置。协议见 docs/NUMERIC_PROTOCOL.md。", "",
        f"全套 {checks['tests']['passed']} 项测试通过；两个正式实验均通过独立重放。","","## 条件化配对覆盖","","| Seed | 世界总数 | 请求配对 | 接受配对 | 尝试组 | 消耗测试世界 | 数值操作数 | 排除操作数 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed,(_,s,p) in reports.items():
        lines.append(f"| {seed} | {checks[str(seed)]['worlds']} | {p['requested_pairs']} | {p['accepted_pairs']} | {p['attempted_groups']} | {p['worlds_consumed']} | {len(s['addresses'])} | {len(s['unsupported_addresses'])} |")
    lines+=["","配对仅按符号可达性接受；不按教师预测或真实类别筛选。接受和拒绝的世界均不复用。该结果属于符号执行条件下的干预分布，不能与之前无条件配对的百分比直接比较。","","## 数值干预与附带误差","","| Seed | 位置 | 方法 | 被干预节点 NMSE | 未干预数值基线 NMSE | 其他执行节点 NMSE | 类别有效配对 | 类别干预一致率 |","|---|---|---|---:|---:|---:|---:|---:|"]
    for seed,(_,s,p) in reports.items():
        for site,methods in s["sites"].items():
            for method,m in methods.items():
                nodes=m["numeric_nodes"];b=m["behavioral"];rate=b.get("informative_accuracy")
                lines.append(f"| {seed} | {site} | {method} | {number(weighted(nodes,'targeted','nmse'))} | {number(weighted(nodes,'targeted','no_intervention_nmse'))} | {number(weighted(nodes,'collateral','nmse'))} | {b.get('informative_pairs',0)} | {'N/A' if rate is None else format(rate,'.1%')} |")
    lines+=["","NMSE 以拟合集各操作数方差归一化，表格按节点观察数加权。节点数值误差不是相互独立的统计样本；不据此计算显著性。读出仅在拟合集训练并冻结。","","## 未参与拟合的双变量组合","","| Seed | 位置 | 方法 | 双变量配对 | 被干预数值个数 | 数值 NMSE | 有效类别配对 | 类别一致率 |","|---|---|---|---:|---:|---:|---:|---:|"]
    for seed,(root,s,p) in reports.items():
        for site,methods in s["sites"].items():
            for method in methods:
                r=read_json(root/"sites"/site/(method+".json"));b=r["behavioral"];n=r["numeric"]
                if b.get("status")=="no_executed_pairs":continue
                masks=np.asarray(b["masks"],bool);held=masks.sum(1)>1
                visits=np.asarray(n["visited"],bool)
                target=np.asarray(n["symbolic_numeric_targets"]);pred=np.asarray(n["predicted"])
                variance=np.array([row["fit_variance"] for row in n["per_node"]])
                scored=visits&masks&held[:,None]
                err=float(np.mean(((target-pred)**2/variance)[scored])) if scored.any() else None
                target_class=np.asarray(b["symbolic_target"]);initial=np.asarray(b["neural_before"])
                # Source/target execution already satisfied by accepted pairs.
                # Recover the same informative subset from per-combination metrics.
                comb=[v for key,v in b["by_combination"].items() if len(key.split(","))>1]
                valid=sum(v["informative_pairs"] for v in comb)
                acc=sum((v["informative_accuracy"] or 0)*v["informative_pairs"] for v in comb)/valid if valid else None
                lines.append(f"| {seed} | {site} | {method} | {int(held.sum())} | {int(scored.sum())} | {number(err)} | {valid} | {'N/A' if acc is None else format(acc,'.1%')} |")
    lines+=["","## 尚未完成","","- 该实验检验已提取规则中的数值操作数，不是自动恢复原始数据到统计量的全部内部运算。","- 线性读出是外加探针，低误差或干预一致不能单独证明真实、唯一的神经机制。","- 三个隐藏位置全部评估，但尚未把只依赖训练/验证数据的位置选择与联合评分接回程序合成。","- 需要结合无数值损失、随机和打乱目标控制解读；不预设数值监督一定改善行为。","- 图结构、父集/算子恢复和通用因果原则仍须独立完成与验证。",""]
    (ROOT/"RESULTS_NUMERIC.md").write_text(chr(10).join(lines),encoding="utf-8")
    save_json(ROOT/"validation"/"numeric_acceptance.json",checks)
    print(checks)

if __name__=="__main__":main()
