"""Read-only integrity verification of the portable 300-world archive."""
from hashlib import sha256
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
MANIFEST = VALIDATION / "interventional_role_confirmation_package_manifest_v1.json"
ARCHIVE = VALIDATION / "interventional_role_confirmation_package_v1.tar.gz"
PLAN = VALIDATION / "interventional_role_confirmation_protocol_v1.json"
SUMMARY = VALIDATION / "interventional_role_confirmation_verified_v1.json"
HANDOFF = VALIDATION / "interventional_role_confirmation_handoff_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify():
    manifest = read(MANIFEST)
    if (manifest["schema"] != "ncd.interventional-role-confirmation-package.v1"
            or manifest["status"] != "packaged-verified-all-300"
            or manifest["package_source_sha256"] != digest(
                VALIDATION / "package_interventional_role_confirmation_v1.py")
            or manifest["archive_file"] != ARCHIVE.name
            or manifest["archive_sha256"] != digest(ARCHIVE)
            or manifest["archive_bytes"] != ARCHIVE.stat().st_size
            or manifest["protocol_sha256"] != digest(PLAN)
            or manifest["verified_summary_sha256"] != digest(SUMMARY)
            or manifest["handoff_sha256"] != digest(HANDOFF)
            or manifest["unit_count"] != 300
            or manifest["original_claim_closed"] is not False
            or manifest["original_objective_achieved"] is not False):
        raise ValueError("Portable package manifest changed")
    summary = read(SUMMARY)
    if summary["status"] != "verified-all-300" or summary["verified_worlds"] != 300:
        raise ValueError("Summary is not complete")
    expected = manifest["files"]
    if manifest["file_count"] != len(expected):
        raise ValueError("Manifest file count changed")
    observed = set()
    with tarfile.open(ARCHIVE, mode="r:gz") as archive:
        for member in archive:
            name = member.name
            path = Path(name)
            if (name in observed or name not in expected
                    or path.is_absolute() or ".." in path.parts
                    or not name.startswith("runs/interventional_role_confirmation_v1/")
                    or not member.isfile() or member.issym() or member.islnk()
                    or member.mtime != 0 or member.uid != 0 or member.gid != 0
                    or member.mode != 0o644):
                raise ValueError("Unsafe or unlisted archive member: " + name)
            observed.add(name)
            with archive.extractfile(member) as stream:
                payload = stream.read()
            if (len(payload) != expected[name]["bytes"]
                    or sha256(payload).hexdigest() != expected[name]["sha256"]):
                raise ValueError("Archive member hash mismatch: " + name)
    if observed != set(expected):
        raise ValueError("Archive member set incomplete")
    units = {name.split("/")[3] for name in observed
             if name.startswith("runs/interventional_role_confirmation_v1/units/")}
    if len(units) != 300:
        raise ValueError("Portable package lacks declared world count")
    return {
        "schema": "ncd.interventional-role-confirmation-package-verification.v1",
        "status": "verified-portable-package",
        "archive_sha256": digest(ARCHIVE),
        "manifest_sha256": digest(MANIFEST),
        "verifier_sha256": digest(__file__),
        "unit_count": len(units),
        "file_count": len(observed),
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
