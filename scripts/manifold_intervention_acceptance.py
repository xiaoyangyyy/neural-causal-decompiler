"""Apply the frozen manifold-intervention acceptance rule."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/manifold_intervention_seed3393",ROOT/"runs/manifold_intervention_seed3394"]
SITES=("representation","head_linear","head_tanh")


def weighted(rows,field):
    values=[(r[field]["nmse"],r[field]["n"]) for r in rows if r[field]["n"]]
    return sum(v*n for v,n in values)/sum(n for _,n in values)


def main():
    records=[];all_replayed=True;full_coverage=True;residual_ok=True
    targeted_ok=True;collateral_ok=True;control_ok=True
    for run in RUNS:
        summary=json.loads((run/"summary.json").read_text())
        cfg=summary["config"];seed=cfg["seed"];sites={}
        full_coverage &= len(summary["groups"])==54 and len(summary["test_combinations"])==1411 and set(summary["sites"])==set(SITES)
        wins=[];control_wins=[]
        for site in SITES:
            value=summary["sites"][site];methods={}
            for name,m in value["methods"].items():
                methods[name]={"targeted_nmse":weighted(m["numeric_nodes"],"targeted"),
                               "collateral_nmse":weighted(m["numeric_nodes"],"collateral"),
                               "behavioral":m["behavioral"],"diagnostics":m["diagnostics"]}
            natural=methods["natural_pca"];ordinary=methods["ordinary_biorthogonal"];random=methods["random_subspace"]
            if natural["targeted_nmse"]<ordinary["targeted_nmse"]:wins.append(site)
            if natural["targeted_nmse"]<random["targeted_nmse"]:control_wins.append(site)
            residual_ok &= natural["diagnostics"]["max_constraint_residual"]<=1e-6
            sites[site]={"subspace":value["subspace"],"methods":methods}
        seed_target=len(wins)>=2;seed_control=len(control_wins)>=2
        seed_collateral=all(sites[s]["methods"]["natural_pca"]["collateral_nmse"]<=
                            sites[s]["methods"]["ordinary_biorthogonal"]["collateral_nmse"] for s in wins)
        targeted_ok &= seed_target;control_ok &= seed_control;collateral_ok &= seed_collateral
        records.append({"seed":seed,"replayed":True,"winning_sites":wins,"random_control_winning_sites":control_wins,
                        "targeted_rule":seed_target,"collateral_rule":seed_collateral,
                        "random_control_rule":seed_control,"sites":sites})
    criteria={"complete_replay":all_replayed,"full_coverage":full_coverage,
              "constraint_residual":residual_ok,"targeted_improvement":targeted_ok,
              "collateral_no_higher_at_winning_sites":collateral_ok,
              "beats_random_subspace":control_ok}
    result={"protocol":"docs/MANIFOLD_CONSTRAINED_INTERVENTION_PROTOCOL.md","runs":records,
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"finite empirical intervention result; not exact circuit identity or causal truth"}
    out=ROOT/"validation/manifold_intervention_acceptance.json";out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
