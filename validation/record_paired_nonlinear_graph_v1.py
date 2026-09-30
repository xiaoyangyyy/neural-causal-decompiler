"""Record the conditional nonlinear paired-do theorem without closing R9."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"
OUTPUT = V / "paired_nonlinear_graph_proof_record_v1.json"


def digest(relative):
    return sha256((ROOT / relative).read_bytes()).hexdigest()


def compute():
    certificate = json.loads((V / "paired_nonlinear_graph_case_v1.json").read_text())
    verification = json.loads((V / "paired_nonlinear_graph_case_verification_v1.json").read_text())
    installed = json.loads((V / "paired_nonlinear_installed_replay_v1.json").read_text())
    if (verification["status"] != "independently-verified-conditional-nonlinear-graph"
            or installed["status"] != "isolated-installed-conditional-proof-replayed"
            or certificate["original_claim_closed"] is not False
            or installed["certificate_sha256"] != digest("validation/paired_nonlinear_graph_case_v1.json")
            or verification["certificate_sha256"] != installed["certificate_sha256"]):
        raise ValueError("Missing conditional proof dependencies")
    return {
        "schema": "ncd.scoped-proof-record.v1",
        "claim": "For a differentiable acyclic additive SCM at one fixed exogenous state, paired node-do responses identify the local direct-effect support when finite-step curvature, response error and direct-effect separation bounds close.",
        "conclusion": "proved-under-explicit-pairing-curvature-and-visibility-assumptions",
        "related_original_atoms": ["R9.graph_recovery", "R9.neural_program"],
        "input_domain": "one observed-coordinate baseline and h=1/16 node-do responses for the exact three-node rational polynomial case",
        "allowed_interventions": "one do(X_i=x_i+h) per node with identical exogenous state across baseline and responses",
        "error_metric": "matrix infinity norm of direct-effect estimate; exact rational upper bound",
        "bound": verification["direct_error_bound_inf"],
        "direct_effect_margin": "2/5",
        "graph_source_target": verification["graph_source_target"],
        "model_checkpoint": None,
        "model_sha256": None,
        "estimator_source_sha256": digest("ncd/paired_nonlinear_graph.py"),
        "proof_dependencies": {
            "certificate_sha256": digest("validation/paired_nonlinear_graph_case_v1.json"),
            "independent_verifier_sha256": digest("validation/verify_paired_nonlinear_graph_case_v1.py"),
            "verification_receipt_sha256": digest("validation/paired_nonlinear_graph_case_verification_v1.json"),
            "installed_replay_sha256": digest("validation/paired_nonlinear_installed_replay_v1.json"),
            "wheel_sha256": installed["wheel_sha256"],
        },
        "uncovered": [
            "independently sampled intervention responses without shared exogenous coupling",
            "all allowed nonlinear worlds and local derivative visibility",
            "frozen Discoverer graph recovery and network-to-program fidelity",
            "finite-sample statistical and device-floating guarantees",
            "original all-instance R9 and R0-R13 objectives",
        ],
        "original_atoms_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: record --write|--verify")
    record = compute()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing scoped record retained")
        OUTPUT.write_text(json.dumps(record, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    elif json.loads(OUTPUT.read_text(encoding="utf-8")) != record:
        raise ValueError("Scoped proof record changed")
    print(json.dumps({"conclusion": record["conclusion"],
                      "original_atoms_closed": False}, sort_keys=True))