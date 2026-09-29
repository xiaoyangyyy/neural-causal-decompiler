from pathlib import Path
from importlib.metadata import version
from xml.etree import ElementTree as ET
import hashlib, importlib, json, subprocess, sys
import finite_graph_chernoff
from finite_graph_chernoff.__main__ import digest, read, save
ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "validation/finite_graph_env_v1"
OUT = ROOT / "validation/finite_chernoff_installed_v1"
OUT.mkdir(exist_ok=True)
if not Path(finite_graph_chernoff.__file__).resolve().is_relative_to(ENV) or version("ncd-finite-chernoff-proof") != "0.1.0":
    raise ValueError("Wrong installed verifier")
config = read(ROOT / "validation/finite_chernoff_protocol_v1.json")
for file, expected in config["source_sha256"].items():
    module = importlib.import_module(file[:-3].replace("/", ".").removesuffix(".__init__"))
    if digest(module.__file__) != expected:
        raise ValueError("Installed module differs")
command = [sys.executable, "-I", "-m", "finite_graph_chernoff", "verify-proof", str(ROOT / "runs/finite_gaussian_tail_recovery_v1/manifest.json")]
replay = subprocess.run(command, cwd=OUT, capture_output=True, timeout=60)
(OUT / "replay.stdout.log").write_bytes(replay.stdout)
(OUT / "replay.stderr.log").write_bytes(replay.stderr)
if replay.returncode:
    raise ValueError(replay.stderr.decode(errors="replace"))
result = json.loads(replay.stdout)
if result["cases"]["six96"]["requested_recovery_success_probability_gate"] != "proved" or result["original_objective_achieved"]:
    raise ValueError("Scope strengthened")
xml = OUT / "pytest.xml"
r = subprocess.run([sys.executable, "-I", "-m", "pytest", str(ROOT / "tests/test_finite_graph_chernoff.py"), "--import-mode=importlib", "-q", "--basetemp=" + str(OUT / "pytest_temp"), "--junitxml=" + str(xml)], cwd=OUT, capture_output=True, timeout=120)
(OUT / "pytest.stdout.log").write_bytes(r.stdout)
(OUT / "pytest.stderr.log").write_bytes(r.stderr)
suite = ET.parse(xml).getroot()[0]
if r.returncode or int(suite.get("tests")) != 15 or any(int(suite.get(k, "0")) for k in ("errors", "failures", "skipped")):
    raise ValueError("Installed tests failed")
save(OUT / "status.json", {"schema": "ncd.finite-chernoff-installed.v1", "status": "verified", "tests": 15, "verification": result, "loaded_package_from_isolated_env": True, "source_checkout_not_on_import_path": True, "source_installed_bytes_equal": True, "proof_kernel_requires_torch": False, "wheel_sha256": digest(ROOT / "dist/ncd_finite_graph_chernoff-0.1.0-py3-none-any.whl"), "bundle_sha256": digest(ROOT / "runs/finite_gaussian_tail_recovery_v1/manifest.json"), "test_source_sha256": digest(ROOT / "tests/test_finite_graph_chernoff.py"), "junit_sha256": digest(xml), "checker_sha256": digest(__file__), "original_objective_achieved": False})
print("Installed Gaussian-tail recovery rechecked; 15 installed tests passed", flush=True)
