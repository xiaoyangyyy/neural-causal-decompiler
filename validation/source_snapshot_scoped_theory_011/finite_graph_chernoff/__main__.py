import argparse
import hashlib
import importlib
import json
from fractions import Fraction
from pathlib import Path
from .boundary import certify, verify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_config(path):
    config = read(path)
    required = {"schema", "output", "cases", "source_sha256", "original_objective_achieved"}
    if set(config) != required or config["schema"] != "ncd.finite-chernoff-protocol.v1" or config["original_objective_achieved"] is not False:
        raise ValueError("Protocol shape or scope mismatch")
    package = Path(importlib.import_module("finite_graph_chernoff").__file__).parent
    files = {"finite_graph_chernoff/" + p.name for p in package.glob("*.py")}
    dependency = Path(importlib.import_module("finite_graph_proof").__file__).parent
    files |= {"finite_graph_proof/" + p.name for p in dependency.glob("*.py")}
    if set(config["source_sha256"]) != files:
        raise ValueError("Incomplete loaded source binding")
    for file, expected in config["source_sha256"].items():
        loaded = importlib.import_module(file[:-3].replace("/", ".").removesuffix(".__init__"))
        if digest(loaded.__file__) != expected:
            raise ValueError("Loaded source binding mismatch")
    if not isinstance(config["cases"], dict) or not config["cases"]:
        raise ValueError("Empty theorem family")
    for key, case in config["cases"].items():
        if not key or not key.isidentifier() or set(case) != {"a", "samples", "delta", "family_size"}:
            raise ValueError("Invalid case")
        certify(**case)
    return config


def prove(config_path, resume=False):
    config_path = Path(config_path).resolve()
    config = check_config(config_path)
    root = config_path.parent.parent
    output = (root / config["output"]).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Unsafe output target")
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("Retain previous evidence; use --resume for independent replay")
        if read(output / "protocol.json") != config:
            raise ValueError("Resume protocol differs")
        return verify_bundle(output / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    save(output / "protocol.json", config)
    certificates = {key: certify(**case) for key, case in config["cases"].items()}
    results = {key: verify(value) for key, value in certificates.items()}
    save(output / "certificates.json", certificates)
    save(output / "verification.json", results)
    save(output / "manifest.json", {"schema": "ncd.finite-chernoff-bundle.v1", "files": {name: digest(output / name) for name in ("protocol.json", "certificates.json", "verification.json")}, "original_objective_achieved": False})
    return verify_bundle(output / "manifest.json")


def verify_bundle(path):
    path = Path(path).resolve()
    manifest = read(path)
    if set(manifest) != {"schema", "files", "original_objective_achieved"} or manifest["schema"] != "ncd.finite-chernoff-bundle.v1" or manifest["original_objective_achieved"] is not False:
        raise ValueError("Bundle shape or scope mismatch")
    if set(manifest["files"]) != {"protocol.json", "certificates.json", "verification.json"}:
        raise ValueError("Incomplete bundle")
    if any(digest(path.parent / name) != expected for name, expected in manifest["files"].items()):
        raise ValueError("Bundle hash mismatch")
    config = check_config(path.parent / "protocol.json")
    certificates = read(path.parent / "certificates.json")
    if set(certificates) != set(config["cases"]):
        raise ValueError("Missing theorem case")
    results = {}
    for key, case in config["cases"].items():
        certificate = certificates[key]
        if any(certificate[field] != (str(Fraction(case[field])) if field in {"a", "delta"} else case[field]) for field in case):
            raise ValueError("Frozen parameter mismatch")
        results[key] = verify(certificate)
    if results != read(path.parent / "verification.json"):
        raise ValueError("Independent verification differs")
    return {"status": "verified-in-declared-family", "cases": results, "original_objective_achieved": False}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("verify-proof")
    p.add_argument("manifest")
    args = parser.parse_args()
    result = prove(args.config, args.resume) if args.command == "prove" else verify_bundle(args.manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
