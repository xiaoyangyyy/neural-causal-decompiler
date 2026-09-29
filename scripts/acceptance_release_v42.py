"""Verify 0.42 source/wheel/install and exact/approximate quotient workflows."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.42.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v42_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v42_run"
FORMAL = ROOT / "runs" / "affine_quotient_global_v1"
PROFILES = ROOT / "runs" / "certified_continuous_scale_seed4701" / "profiles"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Installed command failed: {command}")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    probe = subprocess.run(
        [str(PYTHON), "-c",
         "import json,ncd;print(json.dumps({'file':ncd.__file__,'version':ncd.__version__}))"],
        cwd=ROOT.parent, capture_output=True, text=True, check=True)
    imported = json.loads(probe.stdout)
    installed = ROOT / "validation" / "wheel_v42_env" / "Lib" / "site-packages" / "ncd"
    if (imported["version"] != "0.42.0"
            or Path(imported["file"]).parent.resolve() != installed.resolve()):
        raise ValueError("Probe did not import isolated installed 0.42 package")
    hashes = {}
    with zipfile.ZipFile(WHEEL) as archive:
        for source in sorted((ROOT / "ncd").glob("*.py")):
            member = "ncd/" + source.name
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"Wheel/source mismatch: {source.name}")
            if (installed / source.name).read_bytes() != source.read_bytes():
                raise ValueError(f"Installed/source mismatch: {source.name}")
            hashes[member] = sha(source)
    d8 = PROFILES / "profile_000" / "system.json"
    d128 = PROFILES / "profile_003" / "system.json"
    fresh = OUTPUT / "fresh_affine_d8"
    commands = [
        [str(PYTHON), "-m", "ncd.affine_observability", str(d8), str(fresh)],
        [str(PYTHON), "-m", "ncd.affine_observability", str(d8), str(fresh), "--verify"],
        [str(PYTHON), "-m", "ncd.affine_observability", str(d128),
         str(FORMAL / "frozen_affine_d128"), "--verify"],
    ]
    for name in ("ablated_affine_d128", "mixed_sum_d2", "constant_output_d2"):
        commands.append([
            str(PYTHON), "-m", "ncd.affine_observability",
            str(FORMAL / "controls" / name / "system.json"),
            str(FORMAL / name), "--verify"])
    commands.append([
        str(PYTHON), "-m", "ncd.affine_observability",
        str(ROOT / "runs" / "functional_support_global_v1" / "controls" / "narrow_tent" / "system.json"),
        str(FORMAL / "nonlinear_phase_boundary"), "--verify"])
    commands.append([
        str(PYTHON), "-m", "ncd.automatic_grid_realization", str(d128),
        str(ROOT / "runs" / "automatic_grid_global_v1" / "affine_profile_003"),
        "--verify"])
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"command_{index}.log")
    fresh_cert = json.loads((fresh / "certificate.json").read_text(encoding="utf-8"))
    if fresh_cert["status"] != "certified" or fresh_cert["quotient_dim"] != 8:
        raise ValueError("Fresh installed-wheel quotient did not certify")
    acceptance = json.loads((ROOT / "validation" / "affine_quotient_acceptance.json").read_text(encoding="utf-8"))
    if (acceptance["case_count"], acceptance["certified_count"], acceptance["unresolved_count"]) != (8, 7, 1):
        raise ValueError("Formal quotient acceptance is incomplete")
    if acceptance["exact_vs_approximate"]["finite_state_upper"] != "40824":
        raise ValueError("Prior approximate certificate was not joined")
    suites = list(ET.parse(ROOT / "validation" / "pytest_v42.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(key, 0))
                         for s in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    if tests != 189:
        raise ValueError(f"Unexpected regression count: {tests}")
    status = {"state": "verified", "version": "0.42.0",
              "wheel_sha256": sha(WHEEL), "import": imported,
              "source_modules": hashes, "installed_commands": commands,
              "tests_passed": tests,
              "affine_quotient_acceptance": {"cases": 8, "certified": 7,
                                            "unresolved": 1}}
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "version": status["version"],
                      "tests_passed": tests, "installed_commands": len(commands),
                      "wheel_sha256": status["wheel_sha256"]}))


if __name__ == "__main__":
    main()
