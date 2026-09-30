"""Wait on the live confirmation process, then independently replay all units."""
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.proof_process import run_isolated

VALIDATION = ROOT / "validation"
LAUNCH = VALIDATION / "interventional_role_confirmation_stage_v1_launch.json"
RECEIPT = VALIDATION / "interventional_role_confirmation_handoff_v1.json"
PLAN = VALIDATION / "interventional_role_confirmation_protocol_v1.json"
STDOUT = VALIDATION / "interventional_role_confirmation_stage_v1.stdout.log"
STDERR = VALIDATION / "interventional_role_confirmation_stage_v1.stderr.log"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def wait_for_process(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return "handle-already-gone"
        raise ctypes.WinError(error)
    try:
        while True:
            status = kernel.WaitForSingleObject(handle, 30000)
            if status == 0:
                return "handle-signaled"
            if status != 0x102:
                raise RuntimeError("Unexpected process wait state")
    finally:
        kernel.CloseHandle(handle)


def main():
    if RECEIPT.exists():
        raise FileExistsError("Prior handoff receipt is retained")
    launch = read(LAUNCH)
    if launch["protocol_sha256"] != digest(PLAN):
        raise ValueError("Launch protocol hash mismatch")
    status = wait_for_process(int(launch["pid"]))
    record = {
        "schema": "ncd.interventional-role-confirmation-handoff.v1",
        "status": "unresolved",
        "launcher_pid": launch["pid"],
        "process_wait": status,
        "protocol_sha256": digest(PLAN),
        "stage_stdout_sha256": digest(STDOUT),
        "stage_stderr_sha256": digest(STDERR),
        "original_objective_achieved": False,
    }
    try:
        lines = STDOUT.read_text(encoding="utf-8-sig").splitlines()
        stage = json.loads(lines[-1])
        if (stage["status"] != "computed-pending-independent-replay"
                or stage["completed_units"] != 300
                or stage["declared_units"] != 300
                or STDERR.stat().st_size != 0):
            raise ValueError("Stage did not finish all declared units cleanly")
        command = [
            sys.executable, "-I", "-B",
            str(VALIDATION / "verify_interventional_role_stage_v1.py"),
            "--write", "--require-complete",
        ]
        replay = run_isolated(command, ROOT, 3600, 8 * 1024**3)
        (VALIDATION / "interventional_role_confirmation_verify_v1.stdout.log").write_bytes(
            replay["stdout"])
        (VALIDATION / "interventional_role_confirmation_verify_v1.stderr.log").write_bytes(
            replay["stderr"])
        record["replay_exit_code"] = replay["exit_code"]
        record["replay_resources"] = replay["resources"]
        if (replay["exit_code"] != 0 or replay["resources"]["timeout"]
                or replay["resources"]["active_processes_on_return"] != 0):
            raise RuntimeError("Independent full-cohort replay failed")
        replayed = json.loads(replay["stdout"].decode("utf-8-sig").splitlines()[-1])
        if replayed["status"] != "verified-all-300" or replayed["verified_worlds"] != 300:
            raise ValueError("Independent verifier did not accept all 300 worlds")
        summary = read(VALIDATION / "interventional_role_confirmation_verified_v1.json")
        if summary["status"] != "verified-all-300" or summary["verified_worlds"] != 300:
            raise ValueError("Full summary changed")
        record["summary_sha256"] = digest(
            VALIDATION / "interventional_role_confirmation_verified_v1.json")
        record["status"] = "verified-all-300-candidate-only"
    except Exception as exc:
        record["reason"] = str(exc)
    write(RECEIPT, record)
    print(json.dumps(record, sort_keys=True))
    if record["status"] != "verified-all-300-candidate-only":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
