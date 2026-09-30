"""Sequential, resource-bounded handoff from formal proof to dev mechanism comparison."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.proof_process import run_isolated

VALIDATION = ROOT / "validation"
RECORD = VALIDATION / "interventional_mechanism_dev_handoff_v1.json"
PROOF_CHAIN = VALIDATION / "frozen_three_node_chain_v1.json"
EVALUATION_PLAN = VALIDATION / "interventional_mechanism_dev_evaluation_protocol_v1.json"
PYTHON = VALIDATION / "wheel_v65_env/Scripts/python.exe"
MEMORY_BYTES = 8 * 1024 ** 3


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save(record):
    temporary = RECORD.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, RECORD)


def precheck():
    if not PYTHON.is_file() or not PROOF_CHAIN.is_file():
        raise ValueError("Installed Python or live proof chain is missing")
    chain = read(PROOF_CHAIN)
    if chain["schema"] != "ncd.frozen-three-node-chain.v1":
        raise ValueError("Wrong formal proof chain")
    plan = read(EVALUATION_PLAN)
    if plan["schema"] != "ncd.interventional-mechanism-dev-evaluation-plan.v1":
        raise ValueError("Wrong development evaluation plan")
    if plan["status"] != "frozen-pending-training" or plan["original_claim_closed"] is not False:
        raise ValueError("Evaluation scope changed")
    if plan["evaluation_rows"] != 512 or plan["original_normalized_mse_threshold"] != .01:
        raise ValueError("Evaluation threshold changed")
    if plan["independent_confirmation_worlds"] != 0:
        raise ValueError("Development data cannot become confirmation")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Evaluation source changed: " + name)
    for role in ("mixed", "control"):
        key = role + "_training_plan"
        training_path = ROOT / plan[key]
        if digest(training_path) != plan[key + "_sha256"]:
            raise ValueError(role + " training plan changed")
        training = read(training_path)
        if (training["schema"] != "ncd.interventional-mechanism-dev-training-plan.v1"
                or training["epochs"] != 120 or training["width"] != 48
                or training["seed"] != 8100 or training["training_threads"] != 1
                or training["stage_seconds"] != 43200
                or training["artifact_budget_bytes"] != MEMORY_BYTES
                or training["mechanism_observation_budget"] != 512
                or training.get("true_graph_for_training", training.get("truth_graph_for_training")) is not False
                or training["true_equations_for_training"] is not False):
            raise ValueError(role + " training budget or scope changed")
        for name, expected in training["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(role + " trainer source changed: " + name)
        if digest(ROOT / training["preflight_receipt"]) != training["preflight_receipt_sha256"]:
            raise ValueError(role + " preflight receipt changed")
    return {"proof_pid": chain["pid"], "evaluation_plan_sha256": digest(EVALUATION_PLAN)}


def wait_for_formal_chain(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if handle:
        try:
            while True:
                code = kernel.WaitForSingleObject(handle, 30000)
                if code == 0:
                    break
                if code != 0x102:
                    raise RuntimeError("Cannot observe formal chain handle")
        finally:
            kernel.CloseHandle(handle)
    else:
        error = ctypes.get_last_error()
        if error != 87:
            raise ctypes.WinError(error)
    chain = read(PROOF_CHAIN)
    if chain["pid"] != pid or chain["status"] not in ("accepted-scoped-only", "unresolved"):
        raise RuntimeError("Formal chain ended without a terminal receipt")
    return chain["status"]


def artifact_bytes(plan):
    total = 0
    for role in ("mixed", "control"):
        training = read(ROOT / plan[role + "_training_plan"])
        for folder in (ROOT / training["candidate_directory"], ROOT / training["output"]):
            if folder.exists():
                total += sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    output = ROOT / plan["output"]
    if output.exists():
        total += output.stat().st_size
    return total


def run_step(record, name, command, seconds, expected_status):
    record["phase"] = name
    record["status"] = "running"
    record["updated_utc"] = utc_now()
    save(record)
    execution = run_isolated(
        [str(PYTHON), "-I", "-B", *command], ROOT, seconds, MEMORY_BYTES)
    (VALIDATION / ("interventional_mechanism_dev_handoff_v1_" + name + ".stdout.log")).write_bytes(
        execution["stdout"])
    (VALIDATION / ("interventional_mechanism_dev_handoff_v1_" + name + ".stderr.log")).write_bytes(
        execution["stderr"])
    resources = execution["resources"]
    step = {
        "exit_code": execution["exit_code"],
        "resources": resources,
        "finished_utc": utc_now(),
        "stdout_sha256": sha256(execution["stdout"]).hexdigest(),
        "stderr_sha256": sha256(execution["stderr"]).hexdigest(),
    }
    record.setdefault("steps", {})[name] = step
    save(record)
    if (execution["exit_code"] != 0 or resources["timeout"]
            or not resources["descendants_included"]
            or resources["active_processes_on_return"] != 0
            or resources["peak_job_memory_bytes"] > MEMORY_BYTES):
        raise RuntimeError(name + " failed or exceeded resource budget")
    lines = execution["stdout"].decode("utf-8-sig").strip().splitlines()
    payload = json.loads(lines[-1]) if lines else None
    if not isinstance(payload, dict) or payload.get("status") != expected_status:
        raise RuntimeError(name + " did not report expected status")
    return payload


def main():
    frozen = precheck()
    if RECORD.exists():
        raise FileExistsError("Prior development handoff record is retained")
    record = {
        "schema": "ncd.interventional-mechanism-dev-handoff.v1",
        "status": "waiting-formal-chain",
        "proof_pid": frozen["proof_pid"],
        "evaluation_plan_sha256": frozen["evaluation_plan_sha256"],
        "handoff_runner_sha256": digest(__file__),
        "started_utc": utc_now(),
        "original_claim_closed": False,
        "original_objective_achieved": False,
        "steps": {},
    }
    save(record)
    os.environ.update(
        TEMP=str(VALIDATION), TMP=str(VALIDATION),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        record["formal_chain_status"] = wait_for_formal_chain(frozen["proof_pid"])
        save(record)
        if precheck() != frozen:
            raise ValueError("Frozen input changed while waiting")
        plan = read(EVALUATION_PLAN)
        run_step(
            record, "mixed_replay",
            [str(VALIDATION / "verify_interventional_mechanism_dev_preflight_v2.py")],
            300, "verified-development-preflight")
        run_step(
            record, "control_replay",
            [str(VALIDATION / "verify_interventional_mechanism_dev_control_v1.py")],
            300, "verified-development-control")
        for role in ("mixed", "control"):
            training_path = ROOT / plan[role + "_training_plan"]
            run_step(
                record, role + "_train",
                [str(VALIDATION / "run_interventional_mechanism_dev_training_v1.py"),
                 "--plan", str(training_path), "--train"],
                43200, "trained-development-only")
            used = artifact_bytes(plan)
            record["artifact_bytes"] = used
            save(record)
            if used > MEMORY_BYTES:
                raise ValueError("Combined development artifacts exceeded 8 GiB")
        run_step(
            record, "paired_do_evaluate",
            [str(VALIDATION / "evaluate_interventional_mechanism_dev_v1.py"),
             str(EVALUATION_PLAN)],
            300, "development-only")
        result = read(ROOT / plan["output"])
        if (result["schema"] != "ncd.interventional-mechanism-dev-evaluation.v1"
                or result["protocol_sha256"] != digest(EVALUATION_PLAN)
                or result["original_claim_closed"] is not False
                or result["independent_confirmation_worlds"] != 0):
            raise ValueError("Development evaluation result changed scope")
        record["evaluation_sha256"] = digest(ROOT / plan["output"])
        record["artifact_bytes"] = artifact_bytes(plan)
        if record["artifact_bytes"] > MEMORY_BYTES:
            raise ValueError("Combined development artifacts exceeded 8 GiB")
        record["status"] = "completed-development-only"
    except Exception as exc:
        record["status"] = "unresolved"
        record["reason"] = str(exc)
    record["finished_utc"] = utc_now()
    save(record)
    print(json.dumps({
        "status": record["status"],
        "phase": record.get("phase"),
        "reason": record.get("reason"),
        "original_objective_achieved": False,
    }, sort_keys=True), flush=True)
    if record["status"] != "completed-development-only":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck", action="store_true")
    args = parser.parse_args()
    if args.precheck:
        print(json.dumps(precheck(), sort_keys=True))
    else:
        main()
