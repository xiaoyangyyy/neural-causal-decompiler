"""Wait for frozen development training, then independently replay its result."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.proof_process import run_isolated

VALIDATION = ROOT / "validation"
HANDOFF = VALIDATION / "interventional_mechanism_dev_handoff_v1.json"
HANDOFF_SOURCE = VALIDATION / "advance_interventional_mechanism_dev_v1.py"
CHECKER = VALIDATION / "verify_interventional_mechanism_dev_evaluation_v1.py"
PLAN = VALIDATION / "interventional_mechanism_dev_evaluation_protocol_v1.json"
PYTHON = VALIDATION / "wheel_v65_env/Scripts/python.exe"
RECORD = VALIDATION / "interventional_mechanism_dev_result_verification_v1.json"
MEMORY_BYTES = 8 * 1024 ** 3


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    temporary = RECORD.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, RECORD)


def precheck():
    if not PYTHON.is_file() or not CHECKER.is_file() or not HANDOFF.is_file():
        raise ValueError("Development handoff, checker or Python missing")
    handoff = read(HANDOFF)
    if (handoff["schema"] != "ncd.interventional-mechanism-dev-handoff.v1"
            or handoff["handoff_runner_sha256"] != digest(HANDOFF_SOURCE)
            or handoff["evaluation_plan_sha256"] != digest(PLAN)
            or handoff["original_objective_achieved"] is not False):
        raise ValueError("Frozen development handoff changed")
    return {
        "handoff_runner_sha256": digest(HANDOFF_SOURCE),
        "evaluation_plan_sha256": digest(PLAN),
        "checker_sha256": digest(CHECKER),
    }


def wait_for_process(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("Handoff PID missing from launch")
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
                state = kernel.WaitForSingleObject(handle, 30000)
                if state == 0:
                    return
                if state != 0x102:
                    raise RuntimeError("Cannot wait on development handoff process")
        finally:
            kernel.CloseHandle(handle)
    elif ctypes.get_last_error() != 87:
        raise ctypes.WinError(ctypes.get_last_error())


def main(pid):
    frozen = precheck()
    if RECORD.exists():
        raise FileExistsError("Prior independent result verification retained")
    record = {
        "schema": "ncd.interventional-mechanism-dev-result-verification-stage.v1",
        "status": "waiting-development-handoff",
        "handoff_pid": pid,
        "handoff_runner_sha256": frozen["handoff_runner_sha256"],
        "evaluation_plan_sha256": frozen["evaluation_plan_sha256"],
        "checker_sha256": frozen["checker_sha256"],
        "started_utc": now(),
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    save(record)
    os.environ.update(
        TEMP=str(VALIDATION), TMP=str(VALIDATION),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        wait_for_process(pid)
        handoff = read(HANDOFF)
        if handoff["status"] != "completed-development-only":
            raise ValueError("Development handoff did not finish successfully")
        if (handoff["handoff_runner_sha256"] != frozen["handoff_runner_sha256"]
                or handoff["evaluation_plan_sha256"] != frozen["evaluation_plan_sha256"]
                or digest(CHECKER) != frozen["checker_sha256"]
                or not handoff.get("evaluation_sha256")):
            raise ValueError("Handoff or checker changed while waiting")
        plan = read(PLAN)
        result_path = ROOT / plan["output"]
        if digest(result_path) != handoff["evaluation_sha256"]:
            raise ValueError("Evaluation result changed after handoff")
        record["status"] = "running"
        record["updated_utc"] = now()
        save(record)
        execution = run_isolated(
            [str(PYTHON), "-I", "-B", str(CHECKER)],
            ROOT, 300, MEMORY_BYTES)
        stdout_path = VALIDATION / "interventional_mechanism_dev_result_verification_v1.stdout.log"
        stderr_path = VALIDATION / "interventional_mechanism_dev_result_verification_v1.stderr.log"
        stdout_path.write_bytes(execution["stdout"])
        stderr_path.write_bytes(execution["stderr"])
        resources = execution["resources"]
        record["resource_guard"] = resources
        record["exit_code"] = execution["exit_code"]
        record["stdout_sha256"] = digest(stdout_path)
        record["stderr_sha256"] = digest(stderr_path)
        if (execution["exit_code"] != 0 or resources["timeout"]
                or not resources["descendants_included"]
                or resources["active_processes_on_return"] != 0
                or resources["peak_job_memory_bytes"] > MEMORY_BYTES):
            raise RuntimeError("Independent result verification failed or exceeded budget")
        lines = execution["stdout"].decode("utf-8-sig").strip().splitlines()
        output = json.loads(lines[-1]) if lines else None
        if (not isinstance(output, dict)
                or output.get("status") != "verified-development-only"
                or output.get("result_sha256") != handoff["evaluation_sha256"]
                or output.get("verifier_sha256") != frozen["checker_sha256"]
                or output.get("original_claim_closed") is not False):
            raise ValueError("Independent checker reported unexpected result")
        record["verification"] = output
        record["status"] = "verified-development-only"
    except Exception as exc:
        record["status"] = "unresolved"
        record["reason"] = str(exc)
    record["finished_utc"] = now()
    save(record)
    print(json.dumps({
        "status": record["status"],
        "reason": record.get("reason"),
        "original_objective_achieved": False,
    }, sort_keys=True), flush=True)
    if record["status"] != "verified-development-only":
        raise SystemExit(2)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--precheck":
        print(json.dumps(precheck(), sort_keys=True))
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        raise SystemExit("Usage: script --precheck | <handoff-pid>")
