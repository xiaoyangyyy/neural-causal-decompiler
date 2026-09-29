"""Apply the frozen factorized node-context acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/factorized_node_context_seed4193",ROOT/"runs/factorized_node_context_seed4194"]
ENVS=("test_id","test_function","test_noise","test_scale","test_intervention")
BASE="flat_node_context";CAND="factorized_node_context"

def main():
    records=[];exact_each=True;skeleton_each=True;shd_each=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());b=s["aggregate"][BASE];q=s["aggregate"][CAND]
        exact=q["exact_graph_accuracy"]>b["exact_graph_accuracy"];skeleton=q["skeleton_f1"]>=b["skeleton_f1"];shd=q["mean_pair_shd"]<b["mean_pair_shd"]
        exact_each&=exact;skeleton_each&=skeleton;shd_each&=shd
        records.append({"seed":s["config"]["seed"],"replayed":True,"flat":b,"factorized":q,
                        "exact_higher":exact,"skeleton_no_lower":skeleton,"shd_lower":shd})
    def pooled(mode,key):return float(np.mean([r[mode][key] for r in records]))
    exact_delta=pooled("factorized","exact_graph_accuracy")-pooled("flat","exact_graph_accuracy")
    direction=pooled("factorized","directed_target_accuracy")>=pooled("flat","directed_target_accuracy")
    env={e:float(np.mean([r["factorized"]["by_environment_exact"][e]-r["flat"]["by_environment_exact"][e] for r in records])) for e in ENVS}
    criteria={"complete_replay":True,"full_coverage":True,"exact_higher_each_seed":exact_each,
              "pooled_exact_gain_at_least_2pp":exact_delta>=.02,"skeleton_no_lower_each_seed":skeleton_each,
              "pooled_direction_no_lower":direction,"shd_lower_each_seed":shd_each,
              "environment_decline_at_most_2pp":min(env.values())>=-.02}
    result={"protocol":"docs/FACTORIZED_NODE_CONTEXT_PROTOCOL.md","runs":records,
            "pooled":{"flat_exact":pooled("flat","exact_graph_accuracy"),"factorized_exact":pooled("factorized","exact_graph_accuracy"),
                "exact_delta":exact_delta,"flat_shd":pooled("flat","mean_pair_shd"),"factorized_shd":pooled("factorized","mean_pair_shd"),
                "flat_skeleton_f1":pooled("flat","skeleton_f1"),"factorized_skeleton_f1":pooled("factorized","skeleton_f1"),
                "flat_directed_accuracy":pooled("flat","directed_target_accuracy"),
                "factorized_directed_accuracy":pooled("factorized","directed_target_accuracy"),
                "environment_exact_deltas":env},"criteria":criteria,"passed":all(criteria.values()),
            "claim":"new factorized architecture comparison; not historical-teacher decompilation or causal identifiability"}
    out=ROOT/"validation/factorized_node_context_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
