from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction as Q
import importlib, json, os, subprocess, sys, time, threading
from ncd.io import digest, read_json, save_json
ROOT = Path(__file__).resolve().parents[1]
C = read_json(ROOT / "validation/piecewise_mechanism_resume_protocol_v5.json")
OUT = ROOT / C["output"]
OUT.mkdir(exist_ok=True)
TEMP = ROOT / "validation/task_temp_E_piecewise_v5"
TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
SNAP = ROOT / "validation/source_snapshot_059"


def check_inputs(historical=False):
    for field in ("source_sha256", "inputs_sha256"):
        for file, expected in C[field].items():
            if digest(ROOT / file) != expected:
                raise ValueError("Frozen continuation input changed: " + file)
    if historical:
        import ncd, proof_workbench, torch
        if ncd.__version__ != "0.59.0":
            raise ValueError("Wrong historical core")
        for package in (ncd, proof_workbench):
            folder = Path(package.__file__).resolve().parent
            if not folder.is_relative_to(ROOT / "validation/proof_extensions_env_v1"):
                raise ValueError("Source checkout imported")
            expected_files = {Path(f).name for f in C["historical_loaded_source_sha256"] if f.startswith(package.__name__ + "/")}
            if {f.name for f in folder.glob("*.py")} != expected_files:
                raise ValueError("Historical module set changed")
        for file, expected in C["historical_loaded_source_sha256"].items():
            package = importlib.import_module(file.split("/")[0])
            installed_file = Path(package.__file__).resolve().parent / Path(file).name
            if digest(installed_file) != expected or digest(SNAP / file) != expected:
                raise ValueError("Historical source bytes changed")
        torch.set_num_threads(C["threads"])


def measure(certificate):
    def volume(box):
        result = Q(1)
        for lo, hi in box:
            result *= Q(hi) - Q(lo)
        return result
    total = volume(certificate["domain"])
    accepted = sum((volume(r["box"]) for r in certificate["nodes"] if r.get("kind") == "leaf"), Q(0))
    pending = sum((volume(certificate["nodes"][i]["box"]) for i in certificate["unresolved_cells"]), Q(0))
    if total <= 0 or accepted + pending != total:
        raise ValueError("Coverage volume mismatch")
    return {"proved_volume_fraction": str(accepted / total), "unresolved_volume_fraction": str(pending / total), "measure": "Lebesgue box volume, not data probability"}


if "--check-only" in sys.argv:
    check_inputs(True)
    print("137 frozen installed module files match without executing CLI entrypoints", flush=True)
    raise SystemExit(0)

if "--search" in sys.argv or "--verify" in sys.argv:
    check_inputs(True)
    if not OUT.resolve().is_relative_to(ROOT) or OUT.resolve() == ROOT:
        raise ValueError("Unsafe artifact target")
    def watch_artifacts():
        while True:
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > C["artifact_bytes"]:
                save_json(OUT / "budget_failure.json", {"reason": "artifact budget exhausted", "original_objective_achieved": False})
                os._exit(124)
            time.sleep(1)
    threading.Thread(target=watch_artifacts, daemon=True).start()
    from ncd.frozen_mechanism_proof import export_mechanism
    from proof_workbench.piecewise_mechanism import certify_piecewise, verify_piecewise, program
    network = export_mechanism(ROOT / C["checkpoint"])
    previous = read_json(ROOT / C["resume_certificate"])
    if previous["network"] != network or previous["domain"] != C["domain"] or previous["epsilon"] != str(Q(C["epsilon"])):
        raise ValueError("Changed network/domain/threshold")
    if "--search" in sys.argv:
        if (OUT / "certificate.json").exists() or (OUT / "checkpoint.json").exists():
            raise FileExistsError("Retain continuation artifacts")
        save_json(OUT / "protocol.json", C)
        certificate = certify_piecewise(network, C["domain"], C["epsilon"], C["max_cells"], C["search_seconds"], checkpoint=previous, checkpoint_path=OUT / "checkpoint.json")
        save_json(OUT / "certificate.json", certificate)
        if certificate["status"] == "proved":
            save_json(OUT / "program.json", program(certificate))
        save_json(OUT / "search_result.json", {"status": certificate["status"], "search_cells": certificate["search_cells"], **measure(certificate), "original_objective_achieved": False})
        print(json.dumps(read_json(OUT / "search_result.json")), flush=True)
    else:
        certificate = read_json(OUT / "certificate.json")
        if read_json(OUT / "protocol.json") != C or certificate["network"] != network or certificate["domain"] != C["domain"] or certificate["epsilon"] != str(Q(C["epsilon"])) or certificate["search_cells"] > C["max_cells"]:
            raise ValueError("Changed continuation contract")
        # Old accepted leaves and splits must remain identical. Only pending rows may refine.
        if any(row.get("kind", "pending") != "pending" and certificate["nodes"][i] != row for i, row in enumerate(previous["nodes"])):
            raise ValueError("Historical accepted proof was replaced")
        verification = verify_piecewise(certificate)
        if verification["conclusion"] == "proved":
            if read_json(OUT / "program.json") != program(certificate):
                raise ValueError("Executable program differs")
        elif (OUT / "program.json").exists():
            raise ValueError("Unresolved proof cannot export a certified total program")
        save_json(OUT / "verification.json", {**verification, **measure(certificate), "fresh_installed_process": True, "truth_access": False, "original_objective_achieved": False})
        print(json.dumps(read_json(OUT / "verification.json")), flush=True)
else:
    from ncd.proof_process import run_isolated
    import ncd
    if ncd.__version__!='0.62.0.dev3' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v62r3_env'):
        raise ValueError('Wrong continuation supervisor installation')
    package=Path(ncd.__file__).resolve().parent
    actual={'ncd/'+p.name:digest(p) for p in package.glob('*.py')}
    if actual!=C['supervisor_loaded_source_sha256']:raise ValueError('Changed continuation supervisor source')
    record_path = OUT / "launch.json"
    if record_path.exists():
        raise FileExistsError("Retain continuation launch")
    record = {"schema": "ncd.piecewise-resume-launch.v2", "status": "queued", "pid": os.getpid(), "queued_utc": datetime.now(timezone.utc).isoformat(), "original_objective_achieved": False}
    save_json(record_path, record)
    try:
        while True:
            prerequisite = read_json(ROOT / C["prerequisite"])
            if prerequisite["status"] == "unresolved":
                raise RuntimeError("Prerequisite regression unresolved; continuation not started")
            if prerequisite["status"] == "passed":
                break
            time.sleep(10)
        # The regression worker has finished. Wait for its owning supervisor to exit too.
        while True:
            query = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            response = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, timeout=30, creationflags=0x08000000)
            if response.returncode:
                raise RuntimeError("Cannot check prior heavy processes")
            rows = json.loads(response.stdout.decode("utf-8-sig"))
            rows = rows if isinstance(rows, list) else [rows]
            tokens = ("run_full_regression_v61", "run_registry_v61", "run_registry_v62r3_stage", "run_partial_mechanism_installed_v1.py", "run_piecewise_mechanism_stage_v1.py", "original_confirmation_worker")
            jobs = [r["ProcessId"] for r in rows if r["CommandLine"] and str(ROOT).lower() in r["CommandLine"].lower() and any(t in r["CommandLine"] for t in tokens)]
            record["observed_live_heavy_jobs"] = jobs
            save_json(record_path, record)
            if not jobs:
                break
            time.sleep(10)
        check_inputs()
        started = datetime.now(timezone.utc)
        record.update(status="running", started_utc=started.isoformat(), prerequisite_sha256=digest(ROOT / C["prerequisite"]))
        save_json(record_path, record)
        worker = ROOT / "validation/proof_extensions_env_v1/Scripts/python.exe"
        for mode in ("search", "verify"):
            remaining = C["stage_seconds"] - (datetime.now(timezone.utc) - started).total_seconds()
            if remaining <= 0:
                raise TimeoutError("Continuation stage deadline exhausted")
            execution = run_isolated([str(worker), "-I", "-u", str(Path(__file__).resolve()), "--" + mode], SNAP, remaining, C["memory_bytes"])
            (OUT / (mode + ".stdout.log")).write_bytes(execution["stdout"])
            (OUT / (mode + ".stderr.log")).write_bytes(execution["stderr"])
            record[mode] = {"exit_code": execution["exit_code"], "resources": execution["resources"]}
            save_json(record_path, record)
            if execution["exit_code"] or execution["resources"]["timeout"]:
                raise RuntimeError(mode + " worker unresolved; checkpoint retained")
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > C["artifact_bytes"]:
                raise RuntimeError("Artifact budget exhausted")
        result = read_json(OUT / "verification.json")
        record.update(status="verified" if result["conclusion"] == "proved" else "verified-partial", result=result)
        files = ["protocol.json", "certificate.json", "checkpoint.json", "verification.json", "search_result.json"]
        if result["conclusion"] == "proved":
            files.append("program.json")
        save_json(OUT / "manifest.json", {"schema": "ncd.piecewise-resume-bundle.v2", "files": {f: digest(OUT / f) for f in files}, "previous_certificate_sha256": C["inputs_sha256"][C["resume_certificate"]], "original_objective_achieved": False})
    except Exception as e:
        record.update(status="unresolved", reason=str(e))
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(), original_objective_achieved=False)
    save_json(record_path, record)
