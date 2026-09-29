"""Replay precommitted joint runs and render all controls without test selection."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ncd.io import read_json,save_json
from ncd.joint_experiment import verify_joint

def rate(x):return "N/A" if x is None else f"{x:.1%}"
def subset_metric(measurement,held):
    keys={",".join(str(i) for i,b in enumerate(mask) if b) for mask in held}
    metrics=[v for k,v in measurement["by_combination"].items() if k in keys]
    n=sum(v["informative_pairs"] for v in metrics)
    return {"pairs":sum(v["pairs"] for v in metrics),"informative_pairs":n,
            "informative_accuracy":sum((v["informative_accuracy"] or 0)*v["informative_pairs"] for v in metrics)/n if n else None}

def main():
    acceptance={};lines=["# 联合中间机制实验：191 / 192","","本报告由 scripts/joint_acceptance.py 从独立重放的工件生成。45 项回归测试记录位于 validation/pytest_joint_experiment.xml。","","## 行为保真","","| Seed | 环境 | 网络准确率 | 程序准确率 | 一致率 |","|---|---|---:|---:|---:|"]
    summaries={}
    for seed in (191,192):
        root=ROOT/"runs"/f"joint_seed{seed}"
        print(f"replay joint seed {seed}",flush=True)
        acceptance[str(seed)]=verify_joint(root)
        save_json(ROOT/"validation"/f"joint_seed{seed}_verify.json",acceptance[str(seed)])
        s=read_json(root/"summary.json");summaries[seed]=s
        for env,m in s["behavior"].items():
            lines.append(f"| {seed} | {env} | {rate(m['neural_accuracy'])} | {rate(m['program_accuracy'])} | {rate(m['fidelity'])} |")
    lines+=["","## 联合干预与对照（表达式层）","","| Seed | 环境 | 总配对 | 全部节点实际执行 | 有效配对 | 联合映射 | 打乱目标 | 变量置换 | 随机映射均值 |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    identical=True
    for seed,s in summaries.items():
        root=ROOT/"runs"/f"joint_seed{seed}"
        for key in s["modes"]["expression"]["tests"]:
            a=read_json(root/"evaluations"/"expression"/f"{key}.json")
            b=read_json(root/"evaluations"/"predicate"/f"{key}.json")
            identical &= a["methods"]["joint"]["symbolic_target"]==b["methods"]["joint"]["symbolic_target"]
            methods=a["methods"];m=methods["joint"]["overall"]
            random=[r["overall"]["informative_accuracy"] for k,r in methods.items() if k.startswith("random") and r["overall"]["informative_accuracy"] is not None]
            lines.append(f"| {seed} | {key} | {m['pairs']} | {m['all_active_intermediates_executed']} | {m['informative_pairs']} | {rate(m['informative_accuracy'])} | {rate(methods['shuffled']['overall']['informative_accuracy'])} | {rate(methods['permuted']['overall']['informative_accuracy'])} | {rate(float(np.mean(random)) if random else None)} |")
    acceptance["expression_and_predicate_targets_identical"]=bool(identical)
    lines+=["","两种干预的目标在全部已保存配对中相同："+str(bool(identical))+"。对于当前树的固定阈值，将整段表达式值换为源世界的值，和换成源世界的判断结果在输出上等价。因此不能把两者当成独立机制证据，也未识别出完整数值统计计算。","","## 未参与拟合的组合（表达式层）","","| Seed | 环境 | 留出配对 | 有效配对 | 联合映射 | 打乱目标 | 随机映射均值 |","|---|---|---:|---:|---:|---:|---:|"]
    for seed,s in summaries.items():
        root=ROOT/"runs"/f"joint_seed{seed}";item=s["modes"]["expression"]
        for key in item["tests"]:
            methods=read_json(root/"evaluations"/"expression"/f"{key}.json")["methods"]
            m={k:subset_metric(v,item["held_out_combinations"]) for k,v in methods.items()}
            joint=m["joint"];random=[v["informative_accuracy"] for k,v in m.items() if k.startswith("random") and v["informative_accuracy"] is not None]
            lines.append(f"| {seed} | {key} | {joint['pairs']} | {joint['informative_pairs']} | {rate(joint['informative_accuracy'])} | {rate(m['shuffled']['informative_accuracy'])} | {rate(float(np.mean(random)) if random else None)} |")
    lines+=["","## 同一配对的单变量独立拟合对照（ID）","","| Seed | 中间节点 | 有效配对 | 独立拟合 | 联合拟合 |","|---|---|---:|---:|---:|"]
    for seed in summaries:
        controls=read_json(ROOT/"runs"/f"joint_seed{seed}"/"evaluations"/"expression"/"alignment_test.json")["single_controls"]
        for node,v in controls.items():
            a,b=v["independent"]["overall"],v["joint"]["overall"]
            lines.append(f"| {seed} | {node} | {a['informative_pairs']} | {rate(a['informative_accuracy'])} | {rate(b['informative_accuracy'])} |")
    lines+=["","## 解释与未完成目标","","- 映射选择只使用 alignment_validation；该集合由 alignment_fit 生成语义和固定 seed+10000 构建，与拟合集和测试集世界身份隔离。","- informative 指基线网络/程序一致、符号干预改变输出，且源/目标都实际执行所选节点。零有效配对用 N/A 表示，不作为成功。","- 原始配对会复用世界；_disjoint 行不复用世界，但有效样本更少。表格不提供未经成立假设支持的显著性结论。","- 不同分支上的某些节点不可能同时执行，原始组合覆盖会包含无效组合；报告显式计数，不将总配对数作为有效样本量。","- 控制包含未干预教师（原始 JSON 的 no_intervention_accuracy）、随机、变量置换、打乱目标和相同配对的单变量独立拟合。","- 当前只在一个隐藏表示位置训练多个子空间块；尚未进行多层位置选择，也未将联合机制评分接回程序合成。","- 映射是否恢复多步算法、数值统计运算及一般因果原则仍未得到证明。后续需要独立干预表达式内部运算，并验证操作语义与中间值，而非重复等价的整表达式/判断替换。",""]
    save_json(ROOT/"validation"/"joint_acceptance.json",acceptance)
    (ROOT/"RESULTS_JOINT.md").write_text(chr(10).join(lines),encoding="utf-8")
    print(acceptance)

if __name__=="__main__":main()
