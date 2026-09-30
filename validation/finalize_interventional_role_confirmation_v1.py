"""Wait for the independent replay handoff, then analyze and package all 300 worlds."""
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
from ncd.proof_process import run_isolated
from advance_interventional_role_confirmation_v1 import wait_for_process

VALIDATION = ROOT / "validation"
RECEIPT = VALIDATION / "interventional_role_confirmation_finalization_v1.json"
HANDOFF = VALIDATION / "interventional_role_confirmation_handoff_v1.json"
SUMMARY = VALIDATION / "interventional_role_confirmation_verified_v1.json"
ANALYSIS = VALIDATION / "interventional_role_confirmation_analysis_v1.json"
MANIFEST = VALIDATION / "interventional_role_confirmation_package_manifest_v1.json"
ARCHIVE_VERIFICATION = VALIDATION / "interventional_role_confirmation_package_verification_v1.json"
HANDOFF_PID = 15876
MEMORY = 8 * 1024**3


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def step(name, script, args, seconds, expected_status):
    execution = run_isolated(
        [sys.executable, "-I", "-B", str(VALIDATION / script), *args],
        ROOT, seconds, MEMORY)
    (VALIDATION / f"interventional_role_finalization_{name}.stdout.log").write_bytes(
        execution["stdout"])
    (VALIDATION / f"interventional_role_finalization_{name}.stderr.log").write_bytes(
        execution["stderr"])
    resources = execution["resources"]
    if (execution["exit_code"] != 0 or resources["timeout"]
            or resources["active_processes_on_return"] != 0
            or resources["peak_job_memory_bytes"] > MEMORY):
        raise RuntimeError(name + " failed or exceeded resources: " +
                           execution["stderr"].decode(errors="replace")[-2000:])
    lines = execution["stdout"].decode("utf-8-sig").splitlines()
    output = json.loads(lines[-1])
    if output["status"] != expected_status:
        raise ValueError(name + " reported wrong status")
    return {
        "stdout_sha256": sha256(execution["stdout"]).hexdigest(),
        "stderr_sha256": sha256(execution["stderr"]).hexdigest(),
        "resources": resources,
    }


def main():
    if RECEIPT.exists():
        raise FileExistsError("Existing finalization receipt retained")
    waited = wait_for_process(HANDOFF_PID)
    record = {
        "schema": "ncd.interventional-role-confirmation-finalization.v1",
        "status": "unresolved",
        "handoff_pid": HANDOFF_PID,
        "handoff_wait": waited,
        "finalizer_source_sha256": digest(__file__),
        "steps": {},
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    try:
        handoff = read(HANDOFF)
        summary = read(SUMMARY)
        if (handoff["status"] != "verified-all-300-candidate-only"
                or summary["status"] != "verified-all-300"
                or summary["verified_worlds"] != 300
                or handoff["summary_sha256"] != digest(SUMMARY)):
            raise ValueError("Independent 300-world handoff did not complete")
        record["handoff_sha256"] = digest(HANDOFF)
        record["steps"]["analysis"] = step(
            "analysis", "analyze_interventional_role_confirmation_v1.py",
            ["--write"], 600, "verified-world-level-analysis")
        analysis = read(ANALYSIS)
        if analysis["worlds"] != 300 or analysis["original_claim_closed"] is not False:
            raise ValueError("World-level analysis changed scope")
        record["analysis_sha256"] = digest(ANALYSIS)
        record["steps"]["package"] = step(
            "package", "package_interventional_role_confirmation_v1.py",
            ["--write"], 1800, "packaged-verified-all-300")
        manifest = read(MANIFEST)
        if manifest["unit_count"] != 300:
            raise ValueError("Portable package missing worlds")
        record["manifest_sha256"] = digest(MANIFEST)
        record["steps"]["package_verify"] = step(
            "package_verify", "verify_interventional_role_package_v1.py",
            [], 600, "verified-portable-package")
        stdout = (VALIDATION /
                  "interventional_role_finalization_package_verify.stdout.log")
        package_check = json.loads(stdout.read_text(encoding="utf-8-sig").splitlines()[-1])
        if (package_check["unit_count"] != 300
                or package_check["manifest_sha256"] != digest(MANIFEST)):
            raise ValueError("Independent portable package verification changed")
        write(ARCHIVE_VERIFICATION, package_check)
        record["package_verification_sha256"] = digest(ARCHIVE_VERIFICATION)
        record["status"] = "verified-and-packaged-300-candidate-only"
    except Exception as exc:
        record["reason"] = str(exc)
    write(RECEIPT, record)
    print(json.dumps(record, sort_keys=True))
    if record["status"] != "verified-and-packaged-300-candidate-only":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
