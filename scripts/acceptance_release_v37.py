"""Verify 0.37 source/wheel/install and trained nonlinear CLI workflows."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.37.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v37_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v37_run"
FORMAL = ROOT / "runs" / "trained_nonlinear_global_v1" / "seed_6102" / "d_128"
AFFINE = ROOT / "runs" / "certified_continuous_scale_seed4701"
WEIGHTED = ROOT / "runs" / "weighted_compositional_scale_seed4701"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Installed command failed: {command}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    probe = subprocess.run(
        [str(PYTHON), "-c",
         "import json,ncd;print(json.dumps({'file':ncd.__file__,'version':ncd.__version__}))"],
        cwd=ROOT.parent, capture_output=True, text=True, check=True)
    imported = json.loads(probe.stdout)
    installed = ROOT / "validation" / "wheel_v37_env" / "Lib" / "site-packages" / "ncd"
    if (imported["version"] != "0.37.0"
            or Path(imported["file"]).parent.resolve() != installed.resolve()):
        raise ValueError("Probe did not import isolated installed 0.37 package")
    hashes = {}
    with zipfile.ZipFile(WHEEL) as archive:
        for source in sorted((ROOT / "ncd").glob("*.py")):
            member = "ncd/" + source.name
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"Wheel/source mismatch: {source.name}")
            if (installed / source.name).read_bytes() != source.read_bytes():
                raise ValueError(f"Installed/source mismatch: {source.name}")
            hashes[member] = sha(source)
    commands = [
        [str(PYTHON), "-m", "ncd.trained_nonlinear_realization",
         str(OUTPUT / "fresh_nonlinear"), "--generate", "--quick", "--seed", "6199"],
        [str(PYTHON), "-m", "ncd.trained_nonlinear_realization",
         str(OUTPUT / "fresh_nonlinear"), "--verify"],
        [str(PYTHON), "-m", "ncd.trained_nonlinear_realization",
         str(FORMAL), "--verify"],
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(AFFINE), str(WEIGHTED), "--weighted", "--verify"],
    ]
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"command_{index}.log")
    suites = list(ET.parse(ROOT / "validation" / "pytest_v37.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(key, 0))
                         for s in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    if tests != 165:
        raise ValueError(f"Unexpected regression count: {tests}")
    status = {
        "state": "verified", "version": "0.37.0",
        "wheel_sha256": sha(WHEEL), "import": imported,
        "source_modules": hashes,
        "installed_commands": commands, "tests_passed": tests,
        "scope": "fresh nonlinear generation/replay, formal 128D nonlinear replay, and affine weighted replay from isolated installed wheel",
    }
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "version": status["version"],
                      "tests_passed": tests, "installed_commands": len(commands),
                      "wheel_sha256": status["wheel_sha256"]}))


if __name__ == "__main__":
    main()

