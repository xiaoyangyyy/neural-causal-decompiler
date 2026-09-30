"""Package all 300 independently verified role-confirmation worlds as one archive."""
import argparse
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
PLAN = VALIDATION / "interventional_role_confirmation_protocol_v1.json"
SUMMARY = VALIDATION / "interventional_role_confirmation_verified_v1.json"
HANDOFF = VALIDATION / "interventional_role_confirmation_handoff_v1.json"
ARCHIVE = VALIDATION / "interventional_role_confirmation_package_v1.tar.gz"
MANIFEST = VALIDATION / "interventional_role_confirmation_package_manifest_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def declared_units(plan):
    return {f"seed_{seed}_n{nodes}_{split}_{index}"
            for seed in plan["seeds"] for nodes in plan["nodes"]
            for split in plan["environments"]
            for index in range(plan["worlds_per_cell"])}


def inputs():
    if not SUMMARY.is_file() or not HANDOFF.is_file():
        raise ValueError("Full independent summary and handoff are not complete")
    plan = read(PLAN)
    summary = read(SUMMARY)
    handoff = read(HANDOFF)
    if (summary["status"] != "verified-all-300"
            or summary["verified_worlds"] != 300
            or summary["declared_worlds"] != 300
            or summary["protocol_sha256"] != digest(PLAN)
            or handoff["status"] != "verified-all-300-candidate-only"
            or handoff["summary_sha256"] != digest(SUMMARY)):
        raise ValueError("Full independent verification is not complete")
    output = ROOT / plan["output"]
    if digest(output / "protocol.json") != digest(PLAN):
        raise ValueError("Frozen run protocol changed")
    units = output / "units"
    if {path.name for path in units.iterdir() if path.is_dir()} != declared_units(plan):
        raise ValueError("Declared world directory set differs from run")
    files = [output / "protocol.json"]
    for name in sorted(declared_units(plan)):
        folder = units / name
        result = read(folder / "result.json")
        receipt = read(folder / "verification.json")
        if (result["status"] != "computed-pending-independent-replay"
                or receipt["status"] != "verified-one-independent-world"
                or receipt["result_sha256"] != digest(folder / "result.json")
                or receipt["verifier_sha256"] != summary["verifier_sha256"]):
            raise ValueError("Unit lacks matching independent receipt: " + name)
        files.extend(path for path in folder.rglob("*") if path.is_file())
    if any(path.is_symlink() or path.suffix == ".tmp" for path in files):
        raise ValueError("Package contains symlink or temporary file")
    return plan, summary, handoff, sorted(files, key=lambda path: path.as_posix())


def write():
    if ARCHIVE.exists() or MANIFEST.exists():
        raise FileExistsError("Previous portable package retained")
    plan, summary, handoff, files = inputs()
    temp_archive = ARCHIVE.with_suffix(".tar.gz.tmp")
    entries = {}
    try:
        with temp_archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                               compresslevel=6, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w",
                                  format=tarfile.PAX_FORMAT) as archive:
                    for path in files:
                        relative = path.relative_to(ROOT).as_posix()
                        payload = path.read_bytes()
                        entries[relative] = {
                            "sha256": sha256(payload).hexdigest(),
                            "bytes": len(payload),
                        }
                        info = tarfile.TarInfo(relative)
                        info.size = len(payload)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        archive.addfile(info, io.BytesIO(payload))
        if temp_archive.stat().st_size > plan["artifact_budget_bytes"]:
            raise ValueError("Portable archive exceeds artifact budget")
        os.replace(temp_archive, ARCHIVE)
        manifest = {
            "schema": "ncd.interventional-role-confirmation-package.v1",
            "status": "packaged-verified-all-300",
            "package_source_sha256": digest(__file__),
            "protocol_sha256": digest(PLAN),
            "verified_summary_sha256": digest(SUMMARY),
            "handoff_sha256": digest(HANDOFF),
            "archive_file": ARCHIVE.name,
            "archive_sha256": digest(ARCHIVE),
            "archive_bytes": ARCHIVE.stat().st_size,
            "unit_count": 300,
            "file_count": len(entries),
            "files": entries,
            "original_claim_closed": False,
            "original_objective_achieved": False,
        }
        MANIFEST.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    finally:
        if temp_archive.exists():
            temp_archive.unlink()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.precheck == args.write:
        parser.error("Choose --precheck or --write")
    if args.precheck:
        _, _, _, files = inputs()
        print(json.dumps({"status": "ready-to-package", "files": len(files)},
                         sort_keys=True))
    else:
        result = write()
        print(json.dumps({
            "status": result["status"], "unit_count": result["unit_count"],
            "file_count": result["file_count"], "archive_bytes": result["archive_bytes"],
        }, sort_keys=True))
