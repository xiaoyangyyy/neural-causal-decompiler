"""Installed-wheel acceptance for active end-to-end SCM recovery."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
installed = root / "validation/wheel_v26_env"
out = root / "validation/wheel_v26_run"
wheel = root / "dist/neural_causal_decompiler-0.26.0-py3-none-any.whl"
source = root / "runs/active_intervention_seed4792_quick"
if out.exists():
    raise FileExistsError("Preserve prior 0.26 acceptance")
out.mkdir()
sha = lambda data: hashlib.sha256(data).hexdigest()
modules = {
    path.relative_to(root).as_posix(): sha(path.read_bytes())
    for path in (root / "ncd").glob("*.py")
}
with zipfile.ZipFile(wheel) as archive:
    for name, digest in modules.items():
        assert sha(archive.read(name)) == digest, name
        assert sha((installed / name).read_bytes()) == digest, name
env = dict(
    os.environ,
    PYTHONPATH=str(installed),
    NCD_PROJECT_ROOT=str(root),
)
check = (
    "import ncd;from pathlib import Path;"
    "assert ncd.__version__=='0.26.0';"
    "assert Path(ncd.__file__).resolve().is_relative_to(Path(r'"
    + str(installed)
    + "'));print(ncd.__file__)"
)
subprocess.run(
    [sys.executable, "-c", check],
    cwd=root / "validation",
    env=env,
    check=True,
)
commands = [
    [
        "active-end-to-end",
        "--output",
        str(out / "active-end-to-end"),
        "--source",
        str(source),
        "--quick",
        "--seed",
        "5092",
    ],
    [
        "verify-active-end-to-end",
        str(out / "active-end-to-end"),
    ],
]
records = []
for index, args in enumerate(commands):
    print("installed CLI", args[0], flush=True)
    log = out / f"{index}_{args[0]}.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "ncd", *args],
            cwd=root / "validation",
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    records.append(
        {"command": args, "returncode": result.returncode, "log": log.name}
    )
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {log}")
(out / "status.json").write_text(
    json.dumps(
        {
            "state": "verified",
            "version": "0.26.0",
            "wheel_sha256": sha(wheel.read_bytes()),
            "source_modules": modules,
            "commands": records,
            "scope": (
                "fresh active-end-to-end quick run and complete replay from "
                "isolated installed wheel; science not certified"
            ),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("0.26 wheel verified", flush=True)