"""Independent standard-library theory verifier in an installed kernel environment."""
from pathlib import Path
from importlib.metadata import version
import hashlib, importlib, json, sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recovery_contract(certificate, proof, name):
    model = certificate.get("base_certificate", certificate)
    theory = {"models": model["models"], "noise_law": model["noise_law"], "estimator": model["estimator"]["statistic"]}
    inputs = {key: model[key] for key in ("a", "samples", "delta", "family_size")}
    key = canonical({"theory": theory, "inputs": inputs})
    old_gate = model["estimator"]["requested_confidence_gate"]
    new_gate = proof.get("requested_recovery_success_probability_gate")
    known = old_gate == "proved" or new_gate == "proved"
    refuted = new_gate == "refuted"
    if known and refuted:raise ValueError("Contradictory recovery conclusions")
    witness = None
    if refuted:
        from fractions import Fraction as Q
        if proof.get("independent_product_counterexample_verified") is not True or Q(proof["per_event_error_strict_lower"]) <= Q(model["delta"])/model["family_size"]:
            raise ValueError("Strict counterexample does not contradict the identical per-event contract")
        witness = {"independent_counterexample_verified":True,"per_event_error_strict_lower":proof["per_event_error_strict_lower"],"family_failure_strict_lower":proof["family_failure_strict_lower"]}
    new_label = "strict_error_lower" if refuted else "gaussian_tails"
    return {"id": "gaussian_recovery_" + key, "case": name, "status": "proved" if known else "refuted" if refuted else "unresolved", "inputs": inputs,
            "model_estimator_sha256": canonical(theory), "scope": "Known declared two-model Gaussian family; fixed estimator; ideal IID real observation rows; no actual trained-network or device guarantee",
            "quantity": "uniform per-event recovery failure<=delta/family_size, sufficient for family failure<=delta by union bound",
            "proved_by": [label for label, gate in (("second_moment", old_gate), (new_label, new_gate)) if gate == "proved"],
            "bound_attempts": {"second_moment": old_gate, new_label: new_gate} if new_gate is not None else {"second_moment": old_gate},
            "unresolved_attempt_is_not_a_recovery_refutation": True,"strict_refutation":witness}


def main():
    request = read(sys.argv[1])
    environment, snapshot = Path(request["environment"]).resolve(), Path(request["snapshot"]).resolve()
    if Path.cwd().resolve() != snapshot or not Path(sys.executable).resolve().is_relative_to(environment):
        raise ValueError("Theory kernel environment mismatch")
    packages = {"finite_graph_proof": "ncd-finite-graph-proof", "joint_noise_proof": "ncd-joint-noise-proof", "finite_graph_chernoff": "ncd-finite-chernoff-proof", "estimator_failure_proof": "ncd-estimator-failure-proof"}
    for name, distribution in packages.items():
        package = importlib.import_module(name)
        folder = Path(package.__file__).resolve().parent
        expected = {Path(file).name: sha for file, sha in request["source_sha256"].items() if file.startswith(name + "/")}
        if not folder.is_relative_to(environment) or version(distribution) != "0.1.0" or set(expected) != {p.name for p in folder.glob("*.py")}:
            raise ValueError("Wrong installed theory package or module set")
        for file, sha in expected.items():
            if digest(folder / file) != sha or digest(snapshot / name / file) != sha:
                raise ValueError("Changed theory source bytes")
    job = request["job"]
    mapping = {"finite_sample_mec": "finite_graph_proof", "joint_noise_boundary": "joint_noise_proof", "gaussian_tail_recovery": "finite_graph_chernoff", "fixed_estimator_failure": "estimator_failure_proof"}
    if job["kind"] not in mapping:
        raise ValueError("Unregistered theory kernel")
    artifact = Path(job["artifact"])
    if digest(artifact) != job["artifact_sha256"]:
        raise ValueError("Changed theory artifact")
    package = importlib.import_module(mapping[job["kind"]] + ".__main__")
    verification = package.verify_bundle(artifact)
    certificates = read(artifact.parent / "certificates.json")
    contracts, claims = [], []
    for name, certificate in sorted(certificates.items()):
        proof = verification["cases"][name]
        if job["kind"] == "joint_noise_boundary":
            if proof["joint_w1_l1_strict_lower"] != "7/25" or proof["independent_noise_propagation_refuted"]:
                raise ValueError("Noise counterexample scope changed")
            claims.append({"id": name + ".marginals_do_not_determine_joint_do_law", "status": "refuted", "scope": "The two explicit full-support Gaussian SCMs; identical graph, equations and marginal noises, different joint law under common do(X0=0)", "joint_w1_l1_strict_lower": "7/25", "not_all_learned_residual_laws": True})
        else:
            contracts.append(recovery_contract(certificate, proof, name))
            if job["kind"] == "finite_sample_mec":
                claims.append({"id": name + ".finite_observation_zero_error_mec", "status": "refuted", "scope": "All possibly randomized estimators on exactly the declared two different faithful Gaussian MECs; every finite number of IID observed rows"})
                claims.append({"id": name + ".second_moment_error_bound", "status": "proved", "scope": "The declared known-coefficient estimator, Gaussian laws and IID row count", "error_upper": certificate["estimator"]["uniform_error_upper"]})
            elif job["kind"] == "fixed_estimator_failure":
                claims.append({"id":name+".fixed_estimator_confidence","status":"refuted","scope":"This exact fixed estimator and the explicit independent-event counterexample in the same known Gaussian family; not all estimators or actual networks","per_event_error_strict_lower":proof["per_event_error_strict_lower"],"family_failure_strict_lower":proof["family_failure_strict_lower"]})
            else:
                claims.append({"id": name + ".gaussian_tail_bounds", "status": "proved", "scope": "Conditional Gaussian, Chernoff and Mills/Cauchy-Schwarz bounds; numeric power comparisons can retain an explicit budget gap", "new_numeric_gate": proof["requested_recovery_success_probability_gate"], "older_valid_bound_preserved": True})
    result = {"status": "verified", "conclusion": "verified", "kernel": mapping[job["kind"]], "case_count": len(certificates), "scoped_claims": claims,
              "recovery_contracts": contracts, "independent_bundle_verification": verification, "original_claim_closed": False, "original_objective_achieved": False}
    Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
