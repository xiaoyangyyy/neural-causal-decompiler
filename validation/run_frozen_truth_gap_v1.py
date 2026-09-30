"""Freeze and supervise the post-acceptance truth-only fixed-instance check."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / 'validation'
PLAN = VALIDATION / 'frozen_truth_gap_protocol_v1.json'
ACCEPTANCE = VALIDATION / 'frozen_three_node_acceptance_v1.json'
COMPOSITION = ROOT / 'runs/frozen_three_node_composition_v1/manifest.json'
OUTPUT = ROOT / 'runs/frozen_three_node_truth_gap_v1'
ENV = VALIDATION / 'frozen_three_node_env_v1/Scripts/python.exe'
PACKAGE = VALIDATION / 'frozen_truth_gap_package_v1/frozen_truth_gap'
WORLD_SHA = '0466d361c03164efd0f2e011d3aaf9aceb74dfc13b0b676e2349dc167e8aae80'
HISTORICAL_SHA = '9254d42ed7a352871e6c3d9cf4ffe952556362649e79f1b3eb15112fbae436ce'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    if path.exists():
        raise FileExistsError('Retain immutable prior record: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def freeze():
    if not ENV.is_file() or not ACCEPTANCE.is_file() or not COMPOSITION.is_file():
        raise ValueError('Accepted candidate, sealed bundle, or installed environment missing')
    accepted = read(ACCEPTANCE)
    if accepted['status'] != 'accepted-scoped' or accepted['original_objective_achieved'] is not False:
        raise ValueError('Candidate has not been accepted at its scoped claim')
    if accepted['bundle_manifest_sha256'] != digest(COMPOSITION):
        raise ValueError('Candidate bundle differs from accepted manifest')
    files = ('__init__.py', 'proof.py', '__main__.py')
    loaded = {name: digest(PACKAGE / name) for name in files}
    installed = ENV.parent.parent / 'Lib/site-packages/frozen_truth_gap'
    if {name: digest(installed / name) for name in files} != loaded:
        raise ValueError('Installed truth evaluator differs from frozen source')
    plan = {
        'schema': 'ncd.frozen-truth-gap-plan.v1', 'version': '0.1.0',
        'project_root': str(ROOT), 'output': 'runs/frozen_three_node_truth_gap_v1',
        'composition_acceptance_sha256': digest(ACCEPTANCE),
        'composition_manifest_sha256': digest(COMPOSITION),
        'historical_manifest_sha256': HISTORICAL_SHA, 'world_sha256': WORLD_SHA,
        'loaded_package_sha256': loaded,
        'original_atomic_claim_counts': {'proved': 0, 'refuted': 2, 'unresolved': 36},
        'original_objective_achieved': False, 'stage_seconds': 43200,
        'memory_bytes': 8589934592, 'artifact_budget_bytes': 8589934592, 'threads': 2,
    }
    write_new(PLAN, plan)
    print(json.dumps({'status': 'frozen', 'protocol_sha256': digest(PLAN),
                      'original_objective_achieved': False}, sort_keys=True))


def stage(mode, attempt):
    from ncd.proof_process import run_isolated

    if mode not in ('prove', 'verify') or type(attempt) is not int or not 0 <= attempt <= 9999:
        raise ValueError('Expected prove or verify with a nonnegative attempt index')
    plan = read(PLAN)
    accepted = read(ACCEPTANCE)
    if accepted['status'] != 'accepted-scoped' or digest(ACCEPTANCE) != plan['composition_acceptance_sha256']:
        raise ValueError('Composition acceptance changed')
    if digest(COMPOSITION) != plan['composition_manifest_sha256']:
        raise ValueError('Composition bundle changed')
    if plan['schema'] != 'ncd.frozen-truth-gap-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong truth-only protocol')
    if plan['memory_bytes'] != 8589934592 or plan['threads'] != 2 or not 0 < plan['stage_seconds'] <= 43200:
        raise ValueError('Invalid resource budget')
    if not ENV.is_file():
        raise ValueError('Independent evaluator environment missing')
    if mode == 'prove' and (OUTPUT / 'manifest.json').exists():
        raise FileExistsError('Truth bundle is already sealed')
    if mode == 'verify' and not (OUTPUT / 'manifest.json').is_file():
        raise ValueError('Truth bundle has not been generated')
    suffix = mode + '_attempt%04d' % attempt
    receipt = VALIDATION / ('frozen_truth_gap_stage_v1_' + suffix + '.json')
    command = [str(ENV), '-I', '-B', '-m', 'frozen_truth_gap']
    command += ['prove', '--config', str(PLAN)] if mode == 'prove' else ['verify-proof', str(OUTPUT)]
    os.environ.update(TEMP=str(VALIDATION), TMP=str(VALIDATION),
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    row = {'schema': 'ncd.frozen-truth-gap-stage.v1', 'mode': mode, 'status': 'running',
           'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
           'protocol_sha256': digest(PLAN), 'acceptance_sha256': digest(ACCEPTANCE),
           'runner_sha256': digest(__file__), 'original_objective_achieved': False}
    write_new(receipt, row)
    try:
        execution = run_isolated(command, ROOT, plan['stage_seconds'], plan['memory_bytes'])
        (VALIDATION / ('frozen_truth_gap_stage_v1_' + suffix + '.stdout.log')).write_bytes(execution['stdout'])
        (VALIDATION / ('frozen_truth_gap_stage_v1_' + suffix + '.stderr.log')).write_bytes(execution['stderr'])
        row['resource_guard'] = execution['resources']
        row['exit_code'] = execution['exit_code']
        if execution['exit_code'] or execution['resources']['timeout']:
            raise RuntimeError('Truth evaluator failed or timed out')
        if (not execution['resources'].get('descendants_included') or
                execution['resources'].get('peak_job_memory_bytes', plan['memory_bytes'] + 1) > plan['memory_bytes']):
            raise RuntimeError('Descendant memory was not within the declared limit')
        lines = execution['stdout'].decode('utf-8-sig').strip().splitlines()
        result = json.loads(lines[-1]) if lines else None
        if not isinstance(result, dict) or result.get('status') != 'verified-scoped' or result.get('conclusion') != 'refuted-scoped' or result.get('strict_root_gaps') != 2 or result.get('original_objective_achieved') is not False:
            raise ValueError('Independent verifier did not report two strict scoped gaps')
        if not (OUTPUT / 'manifest.json').is_file():
            raise ValueError('Truth diagnostic manifest missing')
        size = sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        if size > plan['artifact_budget_bytes']:
            raise ValueError('Truth diagnostic artifact budget exceeded')
        row.update(status='verified-scoped', result=result, artifact_bytes=size,
                   bundle_manifest_sha256=digest(OUTPUT / 'manifest.json'))
    except Exception as exc:
        row.update(status='unresolved', reason=str(exc))
    row['finished_utc'] = datetime.now(timezone.utc).isoformat()
    receipt.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': row['status'], 'mode': mode, 'reason': row.get('reason'),
                      'original_objective_achieved': False}, sort_keys=True), flush=True)
    if row['status'] != 'verified-scoped':
        raise SystemExit(2)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'freeze':
        freeze()
    elif len(sys.argv) in (2, 3) and sys.argv[1] in ('prove', 'verify'):
        stage(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)
    else:
        raise SystemExit('Usage: run_frozen_truth_gap_v1.py freeze|prove|verify [attempt]')
