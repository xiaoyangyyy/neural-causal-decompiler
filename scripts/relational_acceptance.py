"""Replay controlled graph runs and report paired effects without selecting tests."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ncd.io import read_json,save_json
from ncd.relational_experiment import verify_relational,MODES,TESTS

def pct(x):return "N/A" if x is None else f"{x:.1%}"

def main():
    suites=list(ET.parse(ROOT/"validation"/"pytest_relational.xml").getroot().iter("testsuite"))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ("failures","errors","skipped"))
    checks={"tests":{"passed":sum(int(s.attrib["tests"]) for s in suites)}}
    summaries={}
    for seed in (493,494):
        print(f"independent relational replay {seed}",flush=True)
        root=ROOT/"runs"/f"relational_seed{seed}"
        checks[str(seed)]=verify_relational(root)
        save_json(ROOT/"validation"/f"relational_seed{seed}_verify.json",checks[str(seed)])
        summaries[seed]=read_json(root/"summary.json")
    lines=["# 显式边关系注意力：实际对照结果","","协议见 docs/RELATIONAL_PROTOCOL.md。两个模型共享世界、训练预算、参数分配和解码。关系开关会改变有效自由度，不把它表述为完全相同容量。", "",
           f"全套 {checks['tests']['passed']} 项测试通过；两个正式实验独立重放通过。","","## ID 神经图（无环投影后）","","| Seed | 节点数 | 模型 | 宏 F1 | 骨架 F1 | 方向准确率 | 整图准确率 | pair SHD |","|---|---:|---|---:|---:|---:|---:|---:|"]
    for seed,s in summaries.items():
        for n in (3,5,8):
            for mode in MODES:
                m=s["evaluation"][mode][f"n{n}_test_id"]["neural_projected"]
                lines.append(f"| {seed} | {n} | {mode} | {m['macro_f1']:.3f} | {m['skeleton_f1']:.3f} | {pct(m['directed_target_accuracy'])} | {pct(m['exact_graph_accuracy'])} | {m['mean_pair_shd']:.3f} |")
    lines+=["","## 同一测试世界的整图胜负","","| Seed | 节点数 | 仅关系模型正确 | 仅基线正确 | 整图准确率差 | 保守 95% 区间 |","|---|---:|---:|---:|---:|---|"]
    paired=[]
    for seed,s in summaries.items():
        root=ROOT/"runs"/f"relational_seed{seed}"
        for n in (3,5,8):
            key=f"n{n}_test_id"
            with np.load(root/"datasets"/key/"samples.npz") as a:truth=a["target_graph"]
            correct={}
            for mode in MODES:
                with np.load(root/"evaluations"/mode/key/"predictions.npz") as a:
                    correct[mode]=np.all(truth==a["neural"],axis=(1,2))
            a,b=correct["with_relations"],correct["without_relations"]
            delta=float(np.mean(a.astype(float)-b.astype(float)))
            # Difference per independent world lies in [-1,1], range two.
            epsilon=float(np.sqrt(2*np.log(2/.05)/len(a)))
            lo,hi=max(-1.,delta-epsilon),min(1.,delta+epsilon)
            entry={"seed":seed,"nodes":n,"worlds":len(a),"relation_only_correct":int(np.sum(a&~b)),
                "baseline_only_correct":int(np.sum(b&~a)),"mean_difference":delta,
                "hoeffding95_interval":[lo,hi],"assumptions":"fixed trained models; independent bounded world units"}
            paired.append(entry)
            lines.append(f"| {seed} | {n} | {entry['relation_only_correct']} | {entry['baseline_only_correct']} | {delta:+.1%} | [{lo:+.1%}, {hi:+.1%}] |")
    lines+=["","区间以世界为单位，适用于该固定基准混合分布；没有把同图内边当作独立样本。区间未对多个节点数/种子的同时比较作多重校正，不能用它们挑选唯一获益子组。","","## 神经图、符号图与投影分开（ID）","","| Seed | 节点数 | 模型 | 神经原始整图 | 神经投影整图 | 符号投影整图 | 程序保真 | 神经删环边数 | 符号删环边数 |","|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed,s in summaries.items():
        for n in (3,5,8):
            for mode in MODES:
                m=s["evaluation"][mode][f"n{n}_test_id"]
                lines.append(f"| {seed} | {n} | {mode} | {pct(m['neural_raw']['exact_graph_accuracy'])} | {pct(m['neural_projected']['exact_graph_accuracy'])} | {pct(m['symbolic_projected']['exact_graph_accuracy'])} | {pct(m['symbolic_projected']['pair_fidelity'])} | {m['neural_removed_cycle_edges']} | {m['symbolic_removed_cycle_edges']} |")
    lines+=["","## OOD 神经图","","| Seed | 环境 | 节点数 | 模型 | 宏 F1 | 方向准确率 | 整图准确率 |","|---|---|---:|---|---:|---:|---:|"]
    for seed,s in summaries.items():
        for split in TESTS[1:]:
            for n in (3,5,8):
                for mode in MODES:
                    m=s["evaluation"][mode][f"n{n}_{split}"]["neural_projected"]
                    lines.append(f"| {seed} | {split} | {n} | {mode} | {m['macro_f1']:.3f} | {pct(m['directed_target_accuracy'])} | {pct(m['exact_graph_accuracy'])} |")
    lines+=["","## 预先选定世界的机制恢复（每格两个世界的均值）","","| Seed | 节点数 | 图来源 | 父集准确比例 | 符号/神经 NMSE | 符号/真值 NMSE | 干预效应 MAE |","|---|---:|---|---:|---:|---:|---:|"]
    for seed,s in summaries.items():
        for n in (3,5,8):
            for mode in (*MODES,"oracle_graph_diagnostic"):
                rows=[r for r in s["mechanisms"] if r["nodes"]==n and r["mode"]==mode]
                assert len(rows)==2
                avg=lambda key:float(np.mean([r[key] for r in rows]))
                lines.append(f"| {seed} | {n} | {mode} | {pct(avg('parent_exact_fraction'))} | {avg('mean_symbolic_neural_nmse'):.4f} | {avg('mean_symbolic_truth_nmse'):.4f} | {avg('mean_intervention_effect_mae'):.4f} |")
    lines+=["","## 边界","","- 真图机制诊断与推断图端到端恢复分列。低方程拟合误差不能替代正确父集或干预效果。","- 高斯世界以 CPDAG 为目标；机制恢复中的 DAG completion 有记录，不声称识别了不可辨识方向。","- 程序仍使用局部统计特征；更强的上下文网络不保证能由当前局部语言高保真表示。","- 新关系模型不是对历史冻结教师的反编译成功证明；旧模型和工件保持不变。","- 任何局部改善都不能替代全原始目标：完整内部算法、原始统计运算逆解以及普遍因果原则仍需独立证据。",""]
    (ROOT/"RESULTS_RELATIONAL.md").write_text(chr(10).join(lines),encoding="utf-8")
    checks["paired_graph_effects"]=paired
    save_json(ROOT/"validation"/"relational_acceptance.json",checks)
    print(checks)

if __name__=="__main__":main()
