"""Check 0.36 source, wheel, isolated installation, and certified CLI replay."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.36.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v36_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v36_run"
SOURCE = ROOT / "runs" / "certified_continuous_scale_seed4701"
WEIGHTED = ROOT / "runs" / "weighted_compositional_scale_seed4701"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT.parent, text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Installed-wheel command failed: {command}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    probe = subprocess.run(
        [str(PYTHON), "-c",
         "import json,ncd;print(json.dumps({'file':ncd.__file__,'version':ncd.__version__}))"],
        cwd=ROOT.parent, text=True, capture_output=True, check=True)
    imported = json.loads(probe.stdout)
    installed = ROOT / "validation" / "wheel_v36_env" / "Lib" / "site-packages" / "ncd"
    if imported["version"] != "0.36.0" or Path(imported["file"]).parent.resolve() != installed.resolve():
        raise ValueError("Probe did not import isolated installed 0.36 package")
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
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(SOURCE), str(WEIGHTED), "--weighted", "--verify"],
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(SOURCE), str(OUTPUT / "fresh_weighted"), "--weighted"],
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(SOURCE), str(OUTPUT / "fresh_weighted"), "--weighted", "--verify"],
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(SOURCE), str(OUTPUT / "fresh_uniform")],
        [str(PYTHON), "-m", "ncd.continuous_compositional_realization",
         str(SOURCE), str(OUTPUT / "fresh_uniform"), "--verify"],
    ]
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"command_{index}.log")
    suites = list(ET.parse(ROOT / "validation" / "pytest_v36.xml").getroot().iter("testsuite"))
    if not suites or any(int(suite.attrib.get(key, 0))
                         for suite in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    count = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    if count != 163:
        raise ValueError(f"Unexpected regression count: {count}")
    status = {
        "state": "verified", "version": "0.36.0",
        "wheel_sha256": sha(WHEEL), "import": imported,
        "source_modules": hashes, "installed_commands": commands,
        "tests_passed": count,
        "scope": "exact-rational uniform and weighted frozen-trained-network certificates from isolated installed wheel",
    }
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "version": status["version"],
                      "tests_passed": count, "installed_commands": len(commands),
                      "wheel_sha256": status["wheel_sha256"]}))


if __name__ == "__main__":
    main()

