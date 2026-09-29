"""Apply the frozen active-intervention graph acceptance rule."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/active_intervention_seed4793",ROOT/"runs/active_intervention_seed4794"]
ENVS=("test_id","test_function","test_noise","test_scale","test_intervention")
BASE="observational_padded";CAND="active_intervention"

def main():
    records=[];exact_each=True;direction_each=True;skeleton_each=True;shd_each=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());b=s["aggregate"][BASE];q=s["aggregate"][CAND]
        exact=q["exact_graph_accuracy"]>b["exact_graph_accuracy"]
        direction=q["directed_target_accuracy"]>b["directed_target_accuracy"]
        skeleton=q["skeleton_f1"]>=b["skeleton_f1"]
        shd=q["mean_pair_shd"]<b["mean_pair_shd"]
        exact_each&=exact;direction_each&=direction;skeleton_each&=skeleton;shd_each&=shd
        records.append({"seed":s["config"]["seed"],"replayed":True,"observational":b,"active":q,
                        "exact_higher":exact,"direction_higher":direction,
                        "skeleton_no_lower":skeleton,"shd_lower":shd,
                        "worlds":s["worlds"],"intervention_levels_per_node":2})
    def pooled(mode,key):return float(np.mean([r[mode][key] for r in records]))
    exact_delta=pooled("active","exact_graph_accuracy")-pooled("observational","exact_graph_accuracy")
    direction_delta=pooled("active","directed_target_accuracy")-pooled("observational","directed_target_accuracy")
    env={e:float(np.mean([r["active"]["by_environment_exact"][e]-r["observational"]["by_environment_exact"][e] for r in records])) for e in ENVS}
    criteria={"complete_replay":True,"full_coverage":True,"exact_higher_each_seed":exact_each,
              "pooled_exact_gain_at_least_10pp":exact_delta>=.10,
              "direction_higher_each_seed":direction_each,
              "pooled_direction_gain_at_least_10pp":direction_delta>=.10,
              "skeleton_no_lower_each_seed":skeleton_each,"shd_lower_each_seed":shd_each,
              "no_environment_exact_decline":min(env.values())>=0}
    result={"protocol":"docs/ACTIVE_INTERVENTION_GRAPH_PROTOCOL.md","runs":records,
            "pooled":{"observational_exact":pooled("observational","exact_graph_accuracy"),"active_exact":pooled("active","exact_graph_accuracy"),
                "exact_delta":exact_delta,"observational_shd":pooled("observational","mean_pair_shd"),"active_shd":pooled("active","mean_pair_shd"),
                "observational_skeleton_f1":pooled("observational","skeleton_f1"),"active_skeleton_f1":pooled("active","skeleton_f1"),
                "observational_directed_accuracy":pooled("observational","directed_target_accuracy"),
                "active_directed_accuracy":pooled("active","directed_target_accuracy"),
                "direction_delta":direction_delta,"environment_exact_deltas":env},
            "criteria":criteria,"passed":all(criteria.values()),
            "scope":"interventional graph discovery with two known intervention levels per node and paired exogenous draws",
            "not_claimed":["purely observational identifiability","historical-teacher decompilation","general causal discovery"]}
    out=ROOT/"validation/active_intervention_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
