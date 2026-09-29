from pathlib import Path
import json,os
from ncd.io import digest
from proof_extensions.__main__ import verify_bundle
ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'validation/source_snapshot_059'
manifest=json.loads((SNAPSHOT/'manifest.json').read_text())
if any(digest(SNAPSHOT/path)!=value for path,value in manifest['files'].items()):raise ValueError('Snapshot source changed')
os.chdir(SNAPSHOT)
result=verify_bundle('runs/original_proof_extensions_bundle_v1/manifest.json')
status={'schema':'ncd.historical-source-snapshot-replay.v1','status':'verified','source_files':len(manifest['files']),
    'source_snapshot_sha256':digest(SNAPSHOT/'manifest.json'),'installed_proof_replay':result,
    'original_project_complete':False}
(ROOT/'validation/source_snapshot_059_replay.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print('frozen-source installed replay verified',flush=True)
