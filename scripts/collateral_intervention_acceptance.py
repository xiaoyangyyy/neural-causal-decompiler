"""Apply the frozen collateral-preserving tangent acceptance rule."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=[ROOT/"runs/collateral_intervention_seed3593",ROOT/"runs/collateral_intervention_seed3594"]
SITES=("representation","head_linear","head_tanh")

def weighted(rows,field):
    pairs=[(x[field]["nmse"],x[field]["n"]) for x in rows if x[field]["n"]]
    return sum(v*n for v,n in pairs)/sum(n for _,n in pairs)

def main():
    records=[];full=True;residual=True;leakage=True;collateral=True;target_guard=True;ordinary=True
    for path in RUNS:
        s=json.loads((path/"summary.json").read_text());seed=s["config"]["seed"];sites={};cw=[];ow=[]
        full &= len(s["groups"])==54 and len(s["test_combinations"])==1411 and set(s["sites"])==set(SITES)
        for site in SITES:
            methods={}
            for name,m in s["sites"][site]["methods"].items():
                methods[name]={"targeted_nmse":weighted(m["numeric_nodes"],"targeted"),
                               "collateral_nmse":weighted(m["numeric_nodes"],"collateral"),
                               "diagnostics":m["diagnostics"],"behavioral":m["behavioral"]}
            base=methods["natural_pca"];candidate=methods["collateral_preserving_pca"];unconstrained=methods["ordinary_biorthogonal"]
            residual &= candidate["diagnostics"]["max_constraint_residual"]<=1e-6
            leakage &= candidate["diagnostics"]["inactive_coordinate_rms"]<base["diagnostics"]["inactive_coordinate_rms"]
            if candidate["collateral_nmse"]<base["collateral_nmse"]:cw.append(site)
            if candidate["targeted_nmse"]<unconstrained["targeted_nmse"]:ow.append(site)
            sites[site]={"subspace":s["sites"][site]["subspace"],"methods":methods}
        seed_collateral=len(cw)>=2
        seed_guard=all(sites[x]["methods"]["collateral_preserving_pca"]["targeted_nmse"]<=sites[x]["methods"]["natural_pca"]["targeted_nmse"] for x in cw)
        seed_ordinary=len(ow)>=2
        collateral &= seed_collateral;target_guard &= seed_guard;ordinary &= seed_ordinary
        records.append({"seed":seed,"replayed":True,"collateral_winning_sites":cw,"ordinary_target_winning_sites":ow,
                        "collateral_rule":seed_collateral,"target_guard":seed_guard,"ordinary_target_rule":seed_ordinary,"sites":sites})
    criteria={"complete_replay":True,"full_coverage":full,"constraint_residual":residual,
              "inactive_leakage_lower_everywhere":leakage,"collateral_improvement":collateral,
              "target_no_higher_at_collateral_wins":target_guard,"beats_ordinary_target":ordinary}
    result={"protocol":"docs/COLLATERAL_PRESERVING_TANGENT_PROTOCOL.md","runs":records,
            "criteria":criteria,"passed":all(criteria.values()),
            "claim":"finite empirical intervention result; not exact circuit identity or causal truth"}
    out=ROOT/"validation/collateral_intervention_acceptance.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"criteria":criteria,"passed":result["passed"],
        "runs":[{"seed":r["seed"],"collateral_winning_sites":r["collateral_winning_sites"],
                 "ordinary_target_winning_sites":r["ordinary_target_winning_sites"],
                 "sites":{site:{name:{"targeted_nmse":m["targeted_nmse"],"collateral_nmse":m["collateral_nmse"],
                    "inactive_coordinate_rms":m["diagnostics"].get("inactive_coordinate_rms")} for name,m in v["methods"].items()
                    if name in ("natural_pca","collateral_preserving_pca")} for site,v in r["sites"].items()}} for r in records]},indent=2))

if __name__=="__main__":main()
