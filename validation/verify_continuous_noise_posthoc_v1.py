"""Replay and integrity check for the post-hoc continuous-noise diagnostic.

This checks deterministic reproduction, not causal identification or a formal
distribution-distance guarantee.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "validation" / "run_continuous_noise_diagnostic_v1.py"
SOURCE_FILES = {
    "validation/run_continuous_noise_diagnostic_v1.py",
    "validation/continuous_noise_candidate_v1.py",
    "ncd/mechanisms.py", "ncd/multiverse.py", "ncd/cdir.py",
    "ncd/worlds.py", "ncd/graphs.py", "ncd/statistics.py",
    "ncd/model.py", "ncd/io.py",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify(package):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Unexpected manifest schema")
    expected_files = {
        "world.json", "model/structured/explicit_scm.json",
        "model/baseline/mechanism_0.pt", "model/baseline/mechanism_1.pt",
        "model/baseline/mechanism_2.pt", "result.json",
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected or missing bundled files")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Bundled file hash mismatch: " + name)
    if set(manifest["source_files"]) != SOURCE_FILES:
        raise ValueError("Unexpected or missing source files")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source file hash mismatch: " + name)
    result = json.loads((package / "result.json").read_text(encoding="utf-8"))
    if (result["status"] != "posthoc-development-only"
            or result["original_claim_closed"] is not False
            or result["original_objective_achieved"] is not False):
        raise ValueError("Diagnostic status overstated")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "replay.json"
        command = [
            sys.executable, "-I", "-B", str(RUNNER),
            "--world", str(package / "world.json"),
            "--model-dir", str(package / "model"),
            "--output", str(output),
            "--rows", str(result["split_rows"]),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True,
                       timeout=60, capture_output=True)
        if output.read_bytes() != (package / "result.json").read_bytes():
            raise ValueError("Diagnostic replay mismatch")
    return {"status": "verified-posthoc-replay",
            "result_sha256": manifest["files"]["result.json"],
            "original_claim_closed": False,
            "original_objective_achieved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    args = parser.parse_args()
    print(json.dumps(verify(args.package), sort_keys=True))


if __name__ == "__main__":
    main()