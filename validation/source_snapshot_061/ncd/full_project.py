"""One command for the complete implemented research workflow."""
from pathlib import Path
import json
from .io import save_json,read_json,digest
from .research_suite import ResearchConfig,run_research
from .graph_experiment import GraphConfig,run_graphs,verify_graphs
from .research_verify import verify_research

def run_all(directory,quick=False,seed=91):
    root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty full-project output directory")
    root.mkdir(parents=True,exist_ok=True)
    save_json(root/"status.json",{"state":"running"})
    try:
        rc=ResearchConfig.quick() if quick else ResearchConfig()
        gc=GraphConfig.quick() if quick else GraphConfig()
        rc.seed=seed;gc.seed=seed
        if not quick:gc.mechanism_worlds=2
        b=run_research(root/"bivariate",rc)
        g=run_graphs(root/"multivariate",gc)
        summary={"seed":seed,"quick":quick,"bivariate":"bivariate/summary.json","multivariate":"multivariate/summary.json",
                 "node_sizes":list(gc.nodes),"bivariate_variants":list(b["program_complexity"]),
                 "mechanism_recoveries":len(g["mechanisms"]),
                 "scope":"all implemented phases from original proposal",
                 "scientific_status":"empirical results; full internal algorithm recovery and universal causal principles not established"}
        save_json(root/"summary.json",summary)
        (root/"report.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Neural Causal Decompiler</title>
<style>body{max-width:900px;margin:50px auto;font:18px/1.7 system-ui}a{color:#164e96}</style>
<h1>Neural Causal Decompiler 完整研究</h1>
<p>两个相互衔接的实验：神经发现程序反编译，以及多变量图和显式 SCM 恢复。</p>
<ul><li><a href="bivariate/report.html">组合式 IR、内部对齐、主动反例与完整消融</a></li>
<li><a href="multivariate/report.html">3 / 5 / 8 节点图、神经机制与符号 SCM</a></li></ul>
<p>工程完成与科学结论分开。以每个报告中的真实指标评估行为保真、结构正确性、内部干预及机制恢复；
不把测试通过等同于一般性因果识别或完整内部算法恢复。</p></html>""",encoding="utf-8")
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error":str(exc)});raise

def verify_all(directory):
    root=Path(directory).resolve()
    for name,expected in read_json(root/"manifest.json")["artifacts"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path)!=expected:raise ValueError("Full-project artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete full workflow")
    b=verify_research(root/"bivariate");g=verify_graphs(root/"multivariate")
    if set(g["node_sizes"])!={3,5,8}:raise ValueError("Original multivariate node sizes missing")
    required={"full","without_alignment","without_mdl","without_cegis","without_environment_augmentation","decision_tree","pysr"}
    if set(b["methods"])!=required:raise ValueError("Original ablation missing")
    return {"status":"verified","bivariate":b,"multivariate":g,
            "science_is_not_certified_by_software_verification":True}
