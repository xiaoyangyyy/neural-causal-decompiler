"""Verify 0.40 source, wheel, isolated install, and support-certificate CLI."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.40.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v40_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v40_run"
FORMAL = ROOT / "runs" / "interventional_support_global_v1"


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
    installed = ROOT / "validation" / "wheel_v40_env" / "Lib" / "site-packages" / "ncd"
    if (imported["version"] != "0.40.0"
            or Path(imported["file"]).parent.resolve() != installed.resolve()):
        raise ValueError("Probe did not import isolated installed 0.40 package")
    hashes = {}
    with zipfile.ZipFile(WHEEL) as archive:
        for source in sorted((ROOT / "ncd").glob("*.py")):
            member = "ncd/" + source.name
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"Wheel/source mismatch: {source.name}")
            if (installed / source.name).read_bytes() != source.read_bytes():
                raise ValueError(f"Installed/source mismatch: {source.name}")
            hashes[member] = sha(source)
    learned8 = ROOT / "runs" / "learned_local_global_v1" / "seed_7202" / "d_8" / "system.json"
    learned128 = ROOT / "runs" / "learned_local_global_v1" / "seed_7202" / "d_128" / "system.json"
    narrow = FORMAL / "controls" / "narrow_tent" / "system.json"
    cancel = FORMAL / "controls" / "path_cancellation" / "system.json"
    fresh = OUTPUT / "fresh_learned"
    commands = [
        [str(PYTHON), "-m", "ncd.interventional_support", str(learned8), str(fresh)],
        [str(PYTHON), "-m", "ncd.interventional_support", str(learned8), str(fresh), "--verify"],
        [str(PYTHON), "-m", "ncd.interventional_support", str(learned128),
         str(FORMAL / "learned_local_global_v1_seed_7202_d_128"), "--verify"],
        [str(PYTHON), "-m", "ncd.interventional_support", str(narrow),
         str(FORMAL / "narrow_tent"), "--verify"],
        [str(PYTHON), "-m", "ncd.interventional_support", str(cancel),
         str(FORMAL / "path_cancellation"), "--verify"],
    ]
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"command_{index}.log")
    if json.loads((fresh / "certificate.json").read_text(encoding="utf-8"))["status"] != "certified":
        raise ValueError("Fresh installed-wheel case was not certified")
    acceptance = json.loads((ROOT / "validation" / "interventional_support_acceptance.json").read_text(encoding="utf-8"))
    if (acceptance["case_count"], acceptance["certified_count"], acceptance["unresolved_count"]) != (24, 22, 2):
        raise ValueError("Formal support acceptance is incomplete")
    suites = list(ET.parse(ROOT / "validation" / "pytest_v40.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(key, 0))
                         for s in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    if tests != 174:
        raise ValueError(f"Unexpected regression count: {tests}")
    status = {"state": "verified", "version": "0.40.0",
              "wheel_sha256": sha(WHEEL), "import": imported,
              "source_modules": hashes, "installed_commands": commands,
              "tests_passed": tests,
              "support_acceptance": {"cases": 24, "certified": 22, "unresolved": 2}}
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "version": status["version"],
                      "tests_passed": tests, "installed_commands": len(commands),
                      "wheel_sha256": status["wheel_sha256"]}))


if __name__ == "__main__":
    main()
