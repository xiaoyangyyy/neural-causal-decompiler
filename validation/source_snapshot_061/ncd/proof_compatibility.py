"""Replay older public schemas in their independently installed frozen cores."""
from pathlib import Path
import json,tempfile
from .io import digest,read_json,save_json
from .proof_process import run_isolated
APPROVED_060='5a32588483f8ee3b9e20e741e7fc349e1a6d0806890186e17ece917ad5439308'
APPROVED_059='a3c8106f367bbcc84f0c14a448ce54caa68a6066860f93232f5297fae688611e'

def replay(project,operation,argument,resume=False):
 project=Path(project).resolve();argument=Path(argument).resolve()
 if operation=='prove':schema=read_json(argument)['schema']
 else:schema=read_json(argument/'summary.json')['schema']
 modern=schema in ('ncd.unified-proof-plan.v2','ncd.unified-proof-bundle.v2')
 if modern:
  snapshot=project/'validation/source_snapshot_060';manifest=snapshot/'manifest.json';expected=APPROVED_060;environment=project/'validation/wheel_v60_env';version='0.60.0.dev1';cwd=project
 else:
  snapshot=project/'validation/source_snapshot_059';manifest=snapshot/'manifest.json';expected=APPROVED_059;environment=project/'validation/wheel_v59_env';version='0.59.0';cwd=snapshot
 if digest(manifest)!=expected:raise ValueError('Unapproved legacy public-schema source snapshot')
 files={name:sha for name,sha in read_json(manifest)['files'].items() if name.startswith('ncd/')}
 if any(digest(snapshot/name)!=sha for name,sha in files.items()):raise ValueError('Legacy source snapshot changed')
 request={'environment':str(environment),'snapshot':str(snapshot),'source_sha256':files,'version':version,'operation':operation,'argument':str(argument),'resume':bool(resume)}
 with tempfile.TemporaryDirectory(prefix='ncd-public-compat-') as tmp:
  d=Path(tmp);save_json(d/'request.json',request)
  execution=run_isolated([str(environment/'Scripts/python.exe'),'-I',str(Path(__file__).with_name('proof_compatibility_runner.py')),str(d/'request.json'),str(d/'response.json')],cwd,43200,8*1024**3)
  if execution['resources']['timeout']:raise TimeoutError('Historical public-schema verification budget exhausted')
  if execution['exit_code']:raise ValueError('Historical public-schema replay rejected: '+execution['stderr'].decode(errors='replace')[-4000:])
  if execution['resources'].get('peak_job_memory_bytes',0)>8*1024**3:raise RuntimeError('Historical public-schema memory budget exceeded')
  return read_json(d/'response.json')
