from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import json, os, subprocess, sys, time
from ncd.io import digest, read_json, save_json
from ncd.proof_process import run_isolated
ROOT = Path(__file__).resolve().parents[1]
C = read_json(ROOT / "validation/full_regression_v61r2_protocol_v1.json")
OUT = ROOT / C["output"]
OUT.mkdir(exist_ok=True)
LAUNCH = ROOT / "validation/full_regression_v61r2_launch"
LAUNCH.mkdir(exist_ok=True)


def check():
    import ncd
    if ncd.__version__ != C["candidate_version"] or not Path(ncd.__file__).resolve().is_relative_to(ROOT / "validation/wheel_v61r2_env"):
        raise ValueError("Wrong installed candidate")
    for field in ("source_sha256", "test_modules"):
        for file, expected in C[field].items():
            if digest(ROOT / file) != expected:
                raise ValueError("Frozen source differs: " + file)
    if digest(ROOT / C["candidate_protocol"]) != C["candidate_protocol_sha256"]:
        raise ValueError("Candidate protocol differs")
    from ncd.proof_registry import validate_config
    validate_config(read_json(ROOT / C["candidate_protocol"]))


def active():
    query = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    p = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, timeout=30, creationflags=0x08000000)
    if p.returncode:
        raise RuntimeError("Process observation failed")
    rows = json.loads(p.stdout.decode("utf-8-sig"))
    rows = rows if isinstance(rows, list) else [rows]
    tokens = ("run_registry_v61r2_stage_v1.py", "run_partial_mechanism_installed_v1.py", "original_confirmation_worker", "run_piecewise_mechanism_stage_v1.py")
    return [{"pid": r["ProcessId"], "command": r["CommandLine"]} for r in rows if r["CommandLine"] and str(ROOT).lower() in r["CommandLine"].lower() and any(t in r["CommandLine"] for t in tokens)]


if "--worker" in sys.argv:
    check()
    temp = OUT / "temp_run_0000"
    if temp.exists() or not temp.resolve().is_relative_to(ROOT):
        raise ValueError("Unsafe or reused temporary directory")
    args = [sys.executable, "-I", "-m", "pytest", *[str(ROOT / n) for n in C["test_modules"]], "--import-mode=importlib", "-q", "--basetemp=" + str(temp), "--junitxml=" + str(OUT / "pytest.xml")]
    with (OUT / "pytest.stdout.log").open("wb") as out, (OUT / "pytest.stderr.log").open("wb") as err:
        child = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err, creationflags=0x08000000)
        budget_failure = None
        while child.poll() is None:
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > C["artifact_bytes"]:
                budget_failure = "artifact budget exhausted"
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                break
            time.sleep(.5)
        code = child.wait()
    tests, issues = 0, 1
    if (OUT / "pytest.xml").is_file():
        suite = ET.parse(OUT / "pytest.xml").getroot()[0]
        tests = int(suite.get("tests"))
        issues = sum(int(suite.get(k, "0")) for k in ("errors", "failures", "skipped"))
    passed = code == 0 and tests == C["expected_tests"] and issues == 0 and budget_failure is None
    save_json(OUT / "test_result.json", {"status": "passed" if passed else "unresolved", "tests": tests, "issues": issues, "exit_code": code, "budget_failure": budget_failure, "original_objective_achieved": False})
    sys.exit(0 if passed else 1)
else:
    record_path = LAUNCH / "launch.json"
    if record_path.exists():
        raise FileExistsError("Retain existing full regression")
    record = {"schema": "ncd.full-regression-launch.v61r2.v1", "status": "queued", "pid": os.getpid(), "queued_utc": datetime.now(timezone.utc).isoformat(), "whole_project_complete": False}
    save_json(record_path, record)
    try:
        while True:
            prerequisite = read_json(ROOT / C["prerequisite"])
            if prerequisite["status"] == "unresolved":
                raise RuntimeError("Prerequisite proof stage unresolved; full regression not started")
            try:
                jobs = active()
                record["observation_error"] = None
            except Exception as e:
                record["observation_error"] = str(e)
                save_json(record_path, record)
                time.sleep(10)
                continue
            record["observed_live_heavy_jobs"] = jobs
            save_json(record_path, record)
            if prerequisite["status"] == "verified" and not jobs:
                break
            if datetime.now(timezone.utc) >= datetime.fromisoformat(C["original_deadline_utc"]):
                raise TimeoutError("Original stage deadline exhausted while queued")
            time.sleep(10)
        check()
        record.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(), prerequisite_sha256=digest(ROOT / C["prerequisite"]))
        save_json(record_path, record)
        remaining = (datetime.fromisoformat(C["original_deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("Original stage deadline exhausted")
        execution = run_isolated([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--worker"], ROOT, min(C["stage_seconds"], remaining), C["memory_bytes"])
        (OUT / "worker.stdout.log").write_bytes(execution["stdout"])
        (OUT / "worker.stderr.log").write_bytes(execution["stderr"])
        tests = read_json(OUT / "test_result.json") if (OUT / "test_result.json").exists() else {"status": "unresolved"}
        passed = execution["exit_code"] == 0 and not execution["resources"]["timeout"] and execution["resources"].get("peak_job_memory_bytes", 0) <= C["memory_bytes"] and tests["status"] == "passed"
        record.update(status="passed" if passed else "unresolved", exit_code=execution["exit_code"], resources=execution["resources"], tests=tests)
    except Exception as e:
        record.update(status="unresolved", reason=str(e))
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(), whole_project_complete=False)
    save_json(record_path, record)
    if record["status"] != "passed":
        sys.exit(1)
