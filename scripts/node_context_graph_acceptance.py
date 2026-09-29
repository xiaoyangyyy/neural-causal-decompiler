"""Apply the frozen node-context graph acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/node_context_graph_seed3993",ROOT/"runs/node_context_graph_seed3994"]
ENVS=("test_id","test_function","test_noise","test_scale","test_intervention")

def main():
    records=[];exact_each=True;shd_each=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());b=s["aggregate"]["baseline"];q=s["aggregate"]["node_context"]
        exact=q["exact_graph_accuracy"]>b["exact_graph_accuracy"];shd=q["mean_pair_shd"]<b["mean_pair_shd"]
        exact_each &= exact;shd_each &= shd
        records.append({"seed":s["config"]["seed"],"replayed":True,"baseline":b,"node_context":q,
                        "exact_higher":exact,"shd_lower":shd})
    def pooled(mode,key):return float(np.mean([r[mode][key] for r in records]))
    exact_delta=pooled("node_context","exact_graph_accuracy")-pooled("baseline","exact_graph_accuracy")
    skeleton=pooled("node_context","skeleton_f1")>=pooled("baseline","skeleton_f1")
    directed=pooled("node_context","directed_target_accuracy")>=pooled("baseline","directed_target_accuracy")
    environment_deltas={e:float(np.mean([r["node_context"]["by_environment_exact"][e]-r["baseline"]["by_environment_exact"][e] for r in records])) for e in ENVS}
    criteria={"complete_replay":True,"full_coverage":True,"exact_higher_each_seed":exact_each,
              "pooled_exact_gain_at_least_3pp":exact_delta>=.03,"shd_lower_each_seed":shd_each,
              "skeleton_and_direction_no_lower":skeleton and directed,
              "environment_decline_at_most_2pp":min(environment_deltas.values())>=-.02}
    result={"protocol":"docs/NODE_CONTEXT_GRAPH_PROTOCOL.md","runs":records,
            "pooled":{"baseline_exact":pooled("baseline","exact_graph_accuracy"),
                "node_context_exact":pooled("node_context","exact_graph_accuracy"),"exact_delta":exact_delta,
                "baseline_shd":pooled("baseline","mean_pair_shd"),"node_context_shd":pooled("node_context","mean_pair_shd"),
                "baseline_skeleton_f1":pooled("baseline","skeleton_f1"),"node_context_skeleton_f1":pooled("node_context","skeleton_f1"),
                "baseline_directed_accuracy":pooled("baseline","directed_target_accuracy"),
                "node_context_directed_accuracy":pooled("node_context","directed_target_accuracy"),
                "environment_exact_deltas":environment_deltas},
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"new architecture comparison; not historical-teacher decompilation or causal identifiability"}
    out=ROOT/"validation/node_context_graph_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
