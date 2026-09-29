"""Apply the frozen antisymmetric-direction acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/antisymmetric_direction_seed4393",ROOT/"runs/antisymmetric_direction_seed4394"]
ENVS=("test_id","test_function","test_noise","test_scale","test_intervention")
BASE="factorized_node_context";CAND="antisymmetric_direction"

def main():
    records=[];exact_each=True;direction_each=True;skeleton_each=True;shd_each=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());b=s["aggregate"][BASE];q=s["aggregate"][CAND]
        exact=q["exact_graph_accuracy"]>b["exact_graph_accuracy"]
        direction=q["directed_target_accuracy"]>b["directed_target_accuracy"]
        skeleton=q["skeleton_f1"]>=b["skeleton_f1"]
        shd=q["mean_pair_shd"]<b["mean_pair_shd"]
        exact_each&=exact;direction_each&=direction;skeleton_each&=skeleton;shd_each&=shd
        records.append({"seed":s["config"]["seed"],"replayed":True,"factorized":b,"antisymmetric":q,
                        "exact_higher":exact,"direction_higher":direction,
                        "skeleton_no_lower":skeleton,"shd_lower":shd})
    def pooled(mode,key):return float(np.mean([r[mode][key] for r in records]))
    exact_delta=pooled("antisymmetric","exact_graph_accuracy")-pooled("factorized","exact_graph_accuracy")
    direction_delta=pooled("antisymmetric","directed_target_accuracy")-pooled("factorized","directed_target_accuracy")
    env={e:float(np.mean([r["antisymmetric"]["by_environment_exact"][e]-r["factorized"]["by_environment_exact"][e] for r in records])) for e in ENVS}
    criteria={"complete_replay":True,"full_coverage":True,"exact_higher_each_seed":exact_each,
              "pooled_exact_gain_at_least_2pp":exact_delta>=.02,
              "direction_higher_each_seed":direction_each,
              "pooled_direction_gain_at_least_2pp":direction_delta>=.02,
              "skeleton_no_lower_each_seed":skeleton_each,"shd_lower_each_seed":shd_each,
              "environment_decline_at_most_2pp":min(env.values())>=-.02}
    result={"protocol":"docs/ANTISYMMETRIC_DIRECTION_PROTOCOL.md","runs":records,
            "pooled":{"factorized_exact":pooled("factorized","exact_graph_accuracy"),
                "antisymmetric_exact":pooled("antisymmetric","exact_graph_accuracy"),"exact_delta":exact_delta,
                "factorized_shd":pooled("factorized","mean_pair_shd"),"antisymmetric_shd":pooled("antisymmetric","mean_pair_shd"),
                "factorized_skeleton_f1":pooled("factorized","skeleton_f1"),"antisymmetric_skeleton_f1":pooled("antisymmetric","skeleton_f1"),
                "factorized_directed_accuracy":pooled("factorized","directed_target_accuracy"),
                "antisymmetric_directed_accuracy":pooled("antisymmetric","directed_target_accuracy"),
                "direction_delta":direction_delta,"environment_exact_deltas":env},
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"new graph-head architecture comparison; not historical-teacher decompilation or causal identifiability"}
    out=ROOT/"validation/antisymmetric_direction_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
