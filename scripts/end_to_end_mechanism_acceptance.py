"""Apply the frozen end-to-end inferred-graph mechanism acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/end_to_end_mechanism_seed3793",ROOT/"runs/end_to_end_mechanism_seed3794"]

def main():
    records=[];lower_each=True;coverage=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());a=s["aggregate"];b=a["inferred_graph"]["baseline"];q=a["inferred_graph"]["structured"]
        lower=q["mean_symbolic_neural_nmse"]<b["mean_symbolic_neural_nmse"];lower_each &= lower
        coverage &= s["world_count"]==15 and {r["nodes"] for r in s["records"]}=={3,5,8} and len({r["environment"] for r in s["records"]})==5
        records.append({"seed":s["config"]["seed"],"replayed":True,"graph":s["graph"],"aggregate":a,
                        "inferred_structured_neural_lower":lower})
    def pooled(mode,method,key):return float(np.mean([r["aggregate"][mode][method][key] for r in records]))
    ib=pooled("inferred_graph","baseline","mean_symbolic_neural_nmse")
    iq=pooled("inferred_graph","structured","mean_symbolic_neural_nmse")
    relative=(ib-iq)/ib
    truth_ratio=pooled("inferred_graph","structured","mean_symbolic_truth_nmse")/pooled("inferred_graph","baseline","mean_symbolic_truth_nmse")
    intervention_ratio=pooled("inferred_graph","structured","mean_intervention_effect_mae")/pooled("inferred_graph","baseline","mean_intervention_effect_mae")
    complexity=pooled("inferred_graph","structured","mean_nonconstant_atoms")<=pooled("inferred_graph","baseline","mean_nonconstant_atoms")
    oracle_truth_ratio=pooled("inferred_graph","structured","mean_symbolic_truth_nmse")/pooled("oracle_graph_diagnostic","structured","mean_symbolic_truth_nmse")
    oracle_intervention_ratio=pooled("inferred_graph","structured","mean_intervention_effect_mae")/pooled("oracle_graph_diagnostic","structured","mean_intervention_effect_mae")
    criteria={"complete_replay":True,"full_coverage":coverage,"inferred_neural_lower_each_seed":lower_each,
              "pooled_neural_reduction_at_least_20pct":relative>=.20,
              "truth_and_intervention_safeguards":truth_ratio<=1.10 and intervention_ratio<=1.10,
              "complexity_no_higher":complexity,
              "within_25pct_of_oracle":oracle_truth_ratio<=1.25 and oracle_intervention_ratio<=1.25}
    result={"protocol":"docs/END_TO_END_MECHANISM_PROTOCOL.md","runs":records,
            "pooled":{"inferred_baseline_neural_nmse":ib,"inferred_structured_neural_nmse":iq,
                "relative_neural_reduction":relative,"truth_ratio_vs_inferred_baseline":truth_ratio,
                "intervention_ratio_vs_inferred_baseline":intervention_ratio,
                "inferred_to_oracle_structured_truth_ratio":oracle_truth_ratio,
                "inferred_to_oracle_structured_intervention_ratio":oracle_intervention_ratio},
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"finite end-to-end benchmark result; not graph identifiability or universal decompilation"}
    out=ROOT/"validation/end_to_end_mechanism_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
