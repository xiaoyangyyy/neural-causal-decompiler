"""Apply the frozen pair-consistent training acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/pair_consistent_seed4593",ROOT/"runs/pair_consistent_seed4594"]
ENVS=("test_id","test_function","test_noise","test_scale","test_intervention")
BASE="raw_factorized";CAND="pair_consistent"

def main():
    records=[];exact_each=True;direction_each=True;skeleton_each=True;shd_each=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());b=s["aggregate"][BASE];q=s["aggregate"][CAND]
        exact=q["exact_graph_accuracy"]>b["exact_graph_accuracy"]
        direction=q["directed_target_accuracy"]>b["directed_target_accuracy"]
        skeleton=q["skeleton_f1"]>=b["skeleton_f1"]
        shd=q["mean_pair_shd"]<b["mean_pair_shd"]
        exact_each&=exact;direction_each&=direction;skeleton_each&=skeleton;shd_each&=shd
        records.append({"seed":s["config"]["seed"],"replayed":True,"raw":b,"consistent":q,
                        "exact_higher":exact,"direction_higher":direction,
                        "skeleton_no_lower":skeleton,"shd_lower":shd})
    def pooled(mode,key):return float(np.mean([r[mode][key] for r in records]))
    exact_delta=pooled("consistent","exact_graph_accuracy")-pooled("raw","exact_graph_accuracy")
    direction_delta=pooled("consistent","directed_target_accuracy")-pooled("raw","directed_target_accuracy")
    env={e:float(np.mean([r["consistent"]["by_environment_exact"][e]-r["raw"]["by_environment_exact"][e] for r in records])) for e in ENVS}
    criteria={"complete_replay":True,"full_coverage":True,"exact_higher_each_seed":exact_each,
              "pooled_exact_gain_at_least_2pp":exact_delta>=.02,
              "direction_higher_each_seed":direction_each,
              "pooled_direction_gain_at_least_2pp":direction_delta>=.02,
              "skeleton_no_lower_each_seed":skeleton_each,"shd_lower_each_seed":shd_each,
              "environment_decline_at_most_2pp":min(env.values())>=-.02}
    result={"protocol":"docs/PAIR_CONSISTENT_TRAINING_PROTOCOL.md","runs":records,
            "pooled":{"raw_exact":pooled("raw","exact_graph_accuracy"),"consistent_exact":pooled("consistent","exact_graph_accuracy"),
                "exact_delta":exact_delta,"raw_shd":pooled("raw","mean_pair_shd"),"consistent_shd":pooled("consistent","mean_pair_shd"),
                "raw_skeleton_f1":pooled("raw","skeleton_f1"),"consistent_skeleton_f1":pooled("consistent","skeleton_f1"),
                "raw_directed_accuracy":pooled("raw","directed_target_accuracy"),
                "consistent_directed_accuracy":pooled("consistent","directed_target_accuracy"),
                "direction_delta":direction_delta,"environment_exact_deltas":env},
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"new graph-training comparison; not historical-teacher decompilation or causal identifiability"}
    out=ROOT/"validation/pair_consistent_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
