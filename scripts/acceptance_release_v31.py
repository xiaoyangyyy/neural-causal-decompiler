"""Verify the 0.31 wheel and all certified realization CLI workflows."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / "neural_causal_decompiler-0.31.0-py3-none-any.whl"
PYTHON = ROOT / "validation" / "wheel_v31_env" / "Scripts" / "python.exe"
OUTPUT = ROOT / "validation" / "wheel_v31_run"
sys.path.insert(0, str(ROOT))
from ncd.io import digest, save_json


def run(command: list[str], log: Path, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Command failed: {command}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    probe = subprocess.run(
        [str(PYTHON), "-c", "import json,ncd;print(json.dumps({'file':ncd.__file__,'version':ncd.__version__}))"],
        cwd=ROOT.parent, text=True, capture_output=True, check=True)
    imported = json.loads(probe.stdout)
    installed = Path(imported["file"]).parent
    expected_install = ROOT / "validation" / "wheel_v31_env" / "Lib" / "site-packages" / "ncd"
    if imported["version"] != "0.31.0" or installed.resolve() != expected_install.resolve():
        raise ValueError("Probe did not import the isolated 0.31 package")

    source_hashes = {}
    with zipfile.ZipFile(WHEEL) as archive:
        for source in sorted((ROOT / "ncd").glob("*.py")):
            member = "ncd/" + source.name
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"Wheel/source mismatch: {source.name}")
            if (installed / source.name).read_bytes() != source.read_bytes():
                raise ValueError(f"Install/source mismatch: {source.name}")
            source_hashes[member] = digest(source)

    commands = [
        [str(PYTHON), "-m", "ncd", "certified-finite", "--quick", "--seed", "2899",
         "--output", str(OUTPUT / "certified-finite")],
        [str(PYTHON), "-m", "ncd", "verify-certified-finite", str(OUTPUT / "certified-finite")],
        [str(PYTHON), "-m", "ncd", "certified-traffic", "--quick",
         "--output", str(OUTPUT / "certified-traffic")],
        [str(PYTHON), "-m", "ncd", "verify-certified-traffic", str(OUTPUT / "certified-traffic")],
        [str(PYTHON), "-m", "ncd", "certified-continuous", "--quick",
         "--output", str(OUTPUT / "certified-continuous")],
        [str(PYTHON), "-m", "ncd", "verify-certified-continuous", str(OUTPUT / "certified-continuous")],
        [str(PYTHON), "-m", "ncd", "certified-continuous-scale", "--quick", "--seed", "4899",
         "--output", str(OUTPUT / "certified-continuous-scale")],
        [str(PYTHON), "-m", "ncd", "verify-certified-continuous-scale",
         str(OUTPUT / "certified-continuous-scale")],
        [str(PYTHON), "-m", "ncd", "certified-continuous-nonlinear", "--quick", "--seed", "5899",
         "--output", str(OUTPUT / "certified-continuous-nonlinear")],
        [str(PYTHON), "-m", "ncd", "verify-certified-continuous-nonlinear",
         str(OUTPUT / "certified-continuous-nonlinear")],
        [str(PYTHON), "-m", "ncd", "certified-continuous-regions", "--quick", "--seed", "8899",
         "--output", str(OUTPUT / "certified-continuous-regions")],
        [str(PYTHON), "-m", "ncd", "verify-certified-continuous-regions",
         str(OUTPUT / "certified-continuous-regions")],
        [str(PYTHON), "-m", "ncd", "certified-continuous-cover", "--quick", "--seed", "9899",
         "--output", str(OUTPUT / "certified-continuous-cover")],
        [str(PYTHON), "-m", "ncd", "verify-certified-continuous-cover",
         str(OUTPUT / "certified-continuous-cover")],
    ]
    for index, command in enumerate(commands):
        run(command, OUTPUT / f"{index}_{command[3]}.log", ROOT.parent)

    suites = list(ET.parse(ROOT / "validation" / "pytest_v31.xml").getroot().iter("testsuite"))
    if not suites or any(int(suite.attrib.get(key, 0)) for suite in suites for key in ("failures", "errors")):
        raise ValueError("Regression XML is not clean")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    status = {
        "state": "verified", "version": "0.31.0", "wheel_sha256": digest(WHEEL),
        "import": imported, "source_modules": source_hashes, "commands": commands,
        "tests_passed": tests,
        "scope": "fresh finite, traffic, continuous, continuous-scale, phase-crossing, initial-region, and global-cover runs from isolated wheel",
    }
    save_json(OUTPUT / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
