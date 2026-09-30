"""Replay both exact grammar certificates with the installed ncd proof kernel."""
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

import ncd.mechanisms
import ncd.proof_intervals

ROOT = Path(__file__).resolve().parents[1]
SITES = (Path(sys.prefix) / "Lib/site-packages").resolve()
for module in (ncd.mechanisms, ncd.proof_intervals):
    if not Path(module.__file__).resolve().is_relative_to(SITES):
        raise ValueError("Proof kernel was not imported from isolated site-packages")
sys.path.insert(0, str(ROOT / "validation"))
import verify_frozen_unary_grammar_v1 as checker

OUTPUT = ROOT / "validation/frozen_unary_grammar_installed_replay_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run():
    mechanism = Path(ncd.mechanisms.__file__).resolve()
    interval = Path(ncd.proof_intervals.__file__).resolve()
    if (digest(mechanism) != digest(ROOT / "ncd/mechanisms.py")
            or digest(interval) != digest(ROOT / "ncd/proof_intervals.py")):
        raise ValueError("Installed proof kernel differs from bound source")
    outputs = [checker.verify(unit_box=flag) for flag in (False, True)]
    for flag, receipt in zip((False, True), outputs):
        stored = checker.NARROW_RECEIPT if flag else checker.RECEIPT
        if json.loads(stored.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Installed replay differs from sealed receipt")
    return {
        "schema": "ncd.frozen-unary-grammar-installed-replay.v1",
        "status": "verified-installed-kernel-replay",
        "replay_source_sha256": digest(__file__),
        "installed_distribution_version": importlib.metadata.version("neural-causal-decompiler"),
        "installed_mechanism_module": str(mechanism.relative_to(SITES)).replace("\\", "/"),
        "installed_mechanism_sha256": digest(mechanism),
        "installed_interval_module": str(interval.relative_to(SITES)).replace("\\", "/"),
        "installed_interval_sha256": digest(interval),
        "wide_certificate_sha256": digest(checker.CERT),
        "unit_box_certificate_sha256": digest(checker.NARROW_CERT),
        "wide_normalized_lower": outputs[0]["normalized_grid_lower"],
        "unit_box_normalized_lower": outputs[1]["normalized_grid_lower"],
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: installed replay --write|--verify")
    result = run()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing installed replay receipt retained")
        OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    elif json.loads(OUTPUT.read_text(encoding="utf-8")) != result:
        raise ValueError("Installed replay receipt changed")
    print(json.dumps({
        "status": result["status"],
        "installed_distribution_version": result["installed_distribution_version"],
        "original_claim_closed": False,
    }, sort_keys=True))
