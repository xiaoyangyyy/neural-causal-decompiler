"""Replay the nonlinear paired-do certificate from an isolated installed wheel."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "validation/paired_nonlinear_installed_v1"
CERTIFICATE = ROOT / "validation/paired_nonlinear_graph_case_v1.json"
VERIFICATION = ROOT / "validation/paired_nonlinear_graph_case_verification_v1.json"
OUTPUT = ROOT / "validation/paired_nonlinear_installed_replay_v1.json"
WHEELS = tuple((ROOT / "validation/paired_nonlinear_wheelhouse_v1").glob("*.whl"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if len(WHEELS) != 1:
        raise ValueError("Exactly one local wheel is required")
    if OUTPUT.exists():
        raise FileExistsError("Installed replay receipt retained")
    sys.path.insert(0, str(INSTALL))
    import ncd.paired_nonlinear_graph as nonlinear
    import ncd.paired_linear_graph as linear
    modules = {"ncd/paired_nonlinear_graph.py": nonlinear,
               "ncd/paired_linear_graph.py": linear}
    source_digests = {}
    with ZipFile(WHEELS[0]) as wheel:
        wheel_sources = {relative: wheel.read(relative) for relative in modules}
    for relative, module in modules.items():
        installed = Path(module.__file__).resolve()
        expected = (INSTALL / relative).resolve()
        if installed != expected:
            raise ValueError("Module imported outside isolated installation")
        if installed.read_bytes() != wheel_sources[relative]:
            raise ValueError("Installed module differs from wheel member")
        if digest(installed) != digest(ROOT / relative):
            raise ValueError("Installed module differs from certificate source")
        source_digests[relative] = digest(installed)
    certificate = read(CERTIFICATE)
    verified = read(VERIFICATION)
    if (verified["certificate_sha256"] != digest(CERTIFICATE)
            or verified["status"] != "independently-verified-conditional-nonlinear-graph"):
        raise ValueError("Independent mathematical verification missing")
    if certificate["source_sha256"]["ncd/paired_nonlinear_graph.py"] != source_digests["ncd/paired_nonlinear_graph.py"]:
        raise ValueError("Certificate targets different estimator")
    if nonlinear.recover_local_graph(**certificate["estimator_input"]) != certificate["estimator_result"]:
        raise ValueError("Isolated installed estimator changed")
    record = {
        "schema": "ncd.paired-nonlinear-installed-replay.v1",
        "status": "isolated-installed-conditional-proof-replayed",
        "wheel_sha256": digest(WHEELS[0]),
        "certificate_sha256": digest(CERTIFICATE),
        "verification_sha256": digest(VERIFICATION),
        "installed_module_sha256": source_digests,
        "installed_source_equals_checkout": True,
        "wheel_source_equals_installed": True,
        "truth_graph_read_by_estimator": False,
        "original_claim_closed": False,
    }
    OUTPUT.write_text(json.dumps(record, sort_keys=True, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()