"""Replay older public schemas in their independently installed frozen cores."""
from pathlib import Path
import json,tempfile
from .io import digest,read_json,save_json
from .proof_process import run_isolated
APPROVED_064='b894ba0e334cb2c70ea6a4a1d12bf21cdd64436b98bb3d9260e02e089ee9dec2'
APPROVED_063='ea22c5a5c396f7918ecb39303ca351103ed52b674b977a60396a2fe4345b1581'
APPROVED_060='5a32588483f8ee3b9e20e741e7fc349e1a6d0806890186e17ece917ad5439308'
APPROVED_062='346ee54e0e6ffa101fdd6f3a1c25a227abad367258c6490a904cdcb778fd15f4'
APPROVED_061='e86647759f286a381d2c3b925a37f19b32962ca66cda391a5dd6b1684f03b795'
APPROVED_059='a3c8106f367bbcc84f0c14a448ce54caa68a6066860f93232f5297fae688611e'

def locate_project(project,argument,schema):
    """Locate approved replay support, never infer a trusted root from cwd alone."""
    layout={
        'ncd.unified-proof-plan.v6':('source_snapshot_064r2',APPROVED_064),
        'ncd.unified-proof-bundle.v6':('source_snapshot_064r2',APPROVED_064),
        'ncd.unified-proof-plan.v5':('source_snapshot_063',APPROVED_063),
        'ncd.unified-proof-bundle.v5':('source_snapshot_063',APPROVED_063),
        'ncd.unified-proof-plan.v4':('source_snapshot_062r3',APPROVED_062),
        'ncd.unified-proof-bundle.v4':('source_snapshot_062r3',APPROVED_062),
        'ncd.unified-proof-plan.v3':('source_snapshot_061',APPROVED_061),
        'ncd.unified-proof-bundle.v3':('source_snapshot_061',APPROVED_061),
        'ncd.unified-proof-plan.v2':('source_snapshot_060',APPROVED_060),
        'ncd.unified-proof-bundle.v2':('source_snapshot_060',APPROVED_060),
    }
    folder,approved=layout.get(schema,('source_snapshot_059',APPROVED_059))
    argument=Path(argument).resolve();project=Path(project).resolve()
    seeds=[project]
    config=argument if argument.is_file() else argument/'config.json'
    if config.is_file():
        declaration=read_json(config)
        if declaration.get('project_root'):seeds.append(Path(declaration['project_root']).resolve())
        if declaration.get('root'):seeds.append((config.parent/declaration['root']).resolve())
    seeds.extend((argument.parent,Path(__file__).resolve().parent))
    candidates=[]
    for seed in seeds:
        for candidate in (seed,*seed.parents):
            if candidate not in candidates:candidates.append(candidate)
    for candidate in candidates:
        manifest=candidate/'validation'/folder/'manifest.json'
        if manifest.is_file() and digest(manifest)==approved:return candidate
    raise ValueError('No approved frozen public-schema replay context found; source manifest identity is required')


def replay(project,operation,argument,resume=False):
 project=Path(project).resolve();argument=Path(argument).resolve()
 if operation=='prove':schema=read_json(argument)['schema']
 else:schema=read_json(argument/'summary.json')['schema']
 project=locate_project(project,argument,schema)
 modern=schema in ('ncd.unified-proof-plan.v2','ncd.unified-proof-bundle.v2')
 if schema in ('ncd.unified-proof-plan.v6','ncd.unified-proof-bundle.v6'):
  snapshot=project/'validation/source_snapshot_064r2';manifest=snapshot/'manifest.json';expected=APPROVED_064;environment=project/'validation/wheel_v64r2_env';version='0.64.0.dev2';cwd=project
 elif schema in ('ncd.unified-proof-plan.v5','ncd.unified-proof-bundle.v5'):
  snapshot=project/'validation/source_snapshot_063';manifest=snapshot/'manifest.json';expected=APPROVED_063;environment=project/'validation/wheel_v63_env';version='0.63.0.dev1';cwd=project
 elif schema in ('ncd.unified-proof-plan.v4','ncd.unified-proof-bundle.v4'):
  snapshot=project/'validation/source_snapshot_062r3';manifest=snapshot/'manifest.json';expected=APPROVED_062;environment=project/'validation/wheel_v62r3_env';version='0.62.0.dev3';cwd=project
 elif schema in ('ncd.unified-proof-plan.v3','ncd.unified-proof-bundle.v3'):
  snapshot=project/'validation/source_snapshot_061';manifest=snapshot/'manifest.json';expected=APPROVED_061;environment=project/'validation/wheel_v61r2_env';version='0.61.0.dev2';cwd=project
 elif modern:
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
