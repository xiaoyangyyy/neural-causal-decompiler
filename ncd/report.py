"""Portable static report. All user / artifact text is escaped."""
from html import escape
import json

def render(summary, programs):
    rows = []
    for split, methods in summary["evaluation"].items():
        for method,m in methods.items():
            rows.append("<tr>"+ "".join(f"<td>{escape(str(x))}</td>" for x in [
                split,method,m["n"],f'{m["neural_accuracy"]:.3f}',f'{m["program_accuracy"]:.3f}',
                f'{m["fidelity"]:.3f}',m["joint_same_wrong"]])+"</tr>")
    code = "\n".join("<h3>"+escape(name)+"</h3><pre>"+escape(p.text())+"</pre>" for name,p in programs.items())
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neural Causal Decompiler 实验报告</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;color:#172b3a}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ccd5dd;text-align:left}
pre{background:#eff3f7;padding:18px;overflow:auto}h1,h2{color:#153c63}.note{background:#fff3d9;padding:16px}</style>
<h1>Neural Causal Decompiler</h1><p>双变量因果发现网络的可执行规则提取与反例审计</p>
<p class="note">这是合成 SCM 基准上的经验研究。行为一致不等于内部算法恢复；
干预对齐只验证指定映射与干预集，允许否定性结果。人工 DSL 带有统计先验。
线性高斯 Unknown 是基准约定。多变量与完整 SCM 恢复尚未实现。</p>
<h2>独立测试</h2><table><thead><tr><th>环境</th><th>方法</th><th>世界数</th><th>网络准确率</th>
<th>程序准确率</th><th>行为一致率</th><th>共同错误数</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table>
<h2>提取程序</h2>""" + code + """<h2>反例搜索</h2><pre>""" + escape(json.dumps(summary["counterexamples"],ensure_ascii=False,indent=2)) + """</pre>
<h2>内部干预验证</h2><pre>""" + escape(json.dumps({k:v for k,v in summary["alignment"].items() if k not in ("source_indices","symbolic_before","symbolic_after")},ensure_ascii=False,indent=2)) + """</pre>
<h2>复现信息</h2><pre>""" + escape(json.dumps(summary["config"],ensure_ascii=False,indent=2)) + "</pre></html>"
