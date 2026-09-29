from pathlib import Path
from importlib.metadata import version
from xml.etree import ElementTree as ET
import hashlib, importlib, json, subprocess, sys
import joint_noise_proof
from joint_noise_proof.__main__ import digest, read, save
ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "validation/finite_graph_env_v1"
OUT = ROOT / "validation/joint_noise_installed_v1"
OUT.mkdir(exist_ok=True)
if not Path(joint_noise_proof.__file__).resolve().is_relative_to(ENV) or version("ncd-joint-noise-proof") != "0.1.0":
    raise ValueError("Wrong installed verifier")
config = read(ROOT / "validation/joint_noise_protocol_v1.json")
for file, expected in config["source_sha256"].items():
    module = importlib.import_module(file[:-3].replace("/", ".").removesuffix(".__init__"))
    if digest(module.__file__) != expected:
        raise ValueError("Installed module differs")
command = [sys.executable, "-I", "-m", "joint_noise_proof", "verify-proof", str(ROOT / "runs/correlated_noise_marginal_boundary_v1/manifest.json")]
replay = subprocess.run(command, cwd=OUT, capture_output=True, timeout=60)
(OUT / "replay.stdout.log").write_bytes(replay.stdout)
(OUT / "replay.stderr.log").write_bytes(replay.stderr)
if replay.returncode:
    raise ValueError(replay.stderr.decode(errors="replace"))
result = json.loads(replay.stdout)
if result["cases"]["correlated"]["joint_w1_l1_strict_lower"] != "7/25" or result["original_objective_achieved"]:
    raise ValueError("Scope strengthened")
xml = OUT / "pytest.xml"
r = subprocess.run([sys.executable, "-I", "-m", "pytest", str(ROOT / "tests/test_joint_noise_proof.py"), "--import-mode=importlib", "-q", "--basetemp=" + str(OUT / "pytest_temp"), "--junitxml=" + str(xml)], cwd=OUT, capture_output=True, timeout=120)
(OUT / "pytest.stdout.log").write_bytes(r.stdout)
(OUT / "pytest.stderr.log").write_bytes(r.stderr)
suite = ET.parse(xml).getroot()[0]
if r.returncode or int(suite.get("tests")) != 10 or any(int(suite.get(k, "0")) for k in ("errors", "failures", "skipped")):
    raise ValueError("Installed tests failed")
save(OUT / "status.json", {"schema": "ncd.joint-noise-installed.v1", "status": "verified", "tests": 10, "verification": result, "loaded_package_from_isolated_env": True, "source_checkout_not_on_import_path": True, "source_installed_bytes_equal": True, "proof_kernel_requires_torch": False, "wheel_sha256": digest(ROOT / "dist/ncd_joint_noise_proof-0.1.0-py3-none-any.whl"), "bundle_sha256": digest(ROOT / "runs/correlated_noise_marginal_boundary_v1/manifest.json"), "test_source_sha256": digest(ROOT / "tests/test_joint_noise_proof.py"), "junit_sha256": digest(xml), "checker_sha256": digest(__file__), "original_objective_achieved": False})
print("Installed joint-noise counterexample rechecked; 10 installed tests passed", flush=True)
