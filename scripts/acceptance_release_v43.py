"""Verify 0.43 source/wheel/install and invariant-slice lower workflows."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.43.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v43_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v43_run"
SOURCE = ROOT / "runs" / "certified_shifted_realization_seed14701"
SCALAR = SOURCE / "profiles" / "profile_000" / "system.json"
PRODUCT = SOURCE / "profiles" / "profile_001" / "system.json"
FORMAL = ROOT / "runs" / "invariant_slice_lower_v1"


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
    installed = ROOT / "validation" / "wheel_v43_env" / "Lib" / "site-packages" / "ncd"
    if (imported["version"] != "0.43.0"
            or Path(imported["file"]).parent.resolve() != installed.resolve()):
        raise ValueError("Probe did not import isolated installed 0.43 package")
    hashes = {}
    with zipfile.ZipFile(WHEEL) as archive:
        for source in sorted((ROOT / "ncd").glob("*.py")):
            member = "ncd/" + source.name
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"Wheel/source mismatch: {source.name}")
            if (installed / source.name).read_bytes() != source.read_bytes():
                raise ValueError(f"Installed/source mismatch: {source.name}")
            hashes[member] = sha(source)
    fresh = OUTPUT / "fresh_slice"
    commands = [
        [str(PYTHON), "-m", "ncd.invariant_slice_lower",
         str(SCALAR), str(PRODUCT), str(fresh)],
        [str(PYTHON), "-m", "ncd.invariant_slice_lower",
         str(SCALAR), str(PRODUCT), str(fresh), "--verify"],
        [str(PYTHON), "-m", "ncd.invariant_slice_lower",
         str(SCALAR), str(PRODUCT), str(FORMAL), "--verify"],
        [str(PYTHON), "-m", "ncd", "verify-certified-multiswitch-lower",
         str(ROOT / "runs" / "certified_multiswitch_lower_seed13701")],
        [str(PYTHON), "-m", "ncd", "verify-certified-shifted-realization",
         str(SOURCE)],
    ]
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"command_{index}.log")
    fresh_certificate = json.loads((fresh / "certificate.json").read_text(encoding="utf-8"))
    if fresh_certificate["status"] != "certified-lower-bound" or fresh_certificate["lower_bound"] != 28:
        raise ValueError("Fresh installed-wheel lower bound did not certify 28")
    acceptance = json.loads((ROOT / "validation" / "invariant_slice_lower_acceptance.json").read_text(encoding="utf-8"))
    if acceptance["minimum_state_interval_2d"] != [28, 81]:
        raise ValueError("Formal invariant-slice acceptance is incomplete")
    suites = list(ET.parse(ROOT / "validation" / "pytest_v43.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(key, 0))
                         for s in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    if tests != 192:
        raise ValueError(f"Unexpected regression count: {tests}")
    status = {"state": "verified", "version": "0.43.0",
              "wheel_sha256": sha(WHEEL), "import": imported,
              "source_modules": hashes, "installed_commands": commands,
              "tests_passed": tests,
              "minimum_state_interval_2d": [28, 81]}
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "version": status["version"],
                      "tests_passed": tests, "installed_commands": len(commands),
                      "wheel_sha256": status["wheel_sha256"]}))


if __name__ == "__main__":
    main()
