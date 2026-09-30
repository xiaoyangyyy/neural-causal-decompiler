"""Job-supervised installed-wheel proof generation and independent replay."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys

from ncd.proof_process import run_isolated

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'validation/frozen_three_node_protocol_v1.json'
PREFLIGHT = ROOT / 'validation/frozen_three_node_preflight_v1.json'
OUTPUT = ROOT / 'runs/frozen_three_node_composition_v1'
ENV = ROOT / 'validation/frozen_three_node_env_v1/Scripts/python.exe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def main(mode, attempt=0):
    if mode not in ('prove', 'verify') or type(attempt) is not int or not 0 <= attempt <= 9999:
        raise ValueError('Expected prove or verify with a nonnegative attempt index')
    if not PLAN.is_file() or not PREFLIGHT.is_file() or not ENV.is_file():
        raise ValueError('Frozen protocol, verified preflight, or independent environment missing')
    plan = json.loads(PLAN.read_bytes())
    preflight = json.loads(PREFLIGHT.read_bytes())
    if plan['schema'] != 'ncd.frozen-three-node-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong proof stage protocol')
    if preflight['status'] != 'verified-preflight' or preflight['composition_conclusion'] != 'proved-scoped':
        raise ValueError('Preflight has no checked scoped result')
    if preflight['loaded_package_sha256'] != plan['loaded_package_sha256'] or preflight['source_sha256'] != plan['source_sha256']:
        raise ValueError('Preflight source differs from frozen proof plan')
    if plan['memory_bytes'] != 8589934592 or plan['threads'] != 2 or not 0 < plan['stage_seconds'] <= 43200:
        raise ValueError('Invalid hard resource budget')
    if mode == 'verify' and not (OUTPUT / 'manifest.json').is_file():
        raise ValueError('Generated proof bundle not available')
    if mode == 'prove' and (OUTPUT / 'manifest.json').is_file():
        raise FileExistsError('Completed proof bundle is immutable')
    if mode == 'prove':
        command = [str(ENV), '-I', '-B', '-m', 'frozen_three_node', 'prove', '--config', str(PLAN)]
    else:
        command = [str(ENV), '-I', '-B', '-m', 'frozen_three_node', 'verify-proof', str(OUTPUT)]
    os.environ.update(TEMP=str(ROOT / 'validation'), TMP=str(ROOT / 'validation'),
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    suffix = mode + '_attempt%04d' % attempt
    path = ROOT / ('validation/frozen_three_node_stage_v1_' + suffix + '.json')
    if path.exists():
        raise FileExistsError('Retain previous stage record')
    record = {
        'schema': 'ncd.frozen-three-node-stage.v1', 'mode': mode, 'status': 'running',
        'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(PLAN), 'preflight_sha256': digest(PREFLIGHT),
        'runner_sha256': digest(__file__), 'original_objective_achieved': False,
    }
    save(path, record)
    try:
        execution = run_isolated(command, ROOT, plan['stage_seconds'], plan['memory_bytes'])
        stdout = ROOT / ('validation/frozen_three_node_stage_v1_' + suffix + '.stdout.log')
        stderr = ROOT / ('validation/frozen_three_node_stage_v1_' + suffix + '.stderr.log')
        stdout.write_bytes(execution['stdout'])
        stderr.write_bytes(execution['stderr'])
        record['resource_guard'] = execution['resources']
        record['exit_code'] = execution['exit_code']
        if execution['exit_code'] or execution['resources']['timeout']:
            raise RuntimeError('Isolated proof process failed or exceeded stage time')
        if not execution['resources'].get('descendants_included') or execution['resources'].get('peak_job_memory_bytes', plan['memory_bytes'] + 1) > plan['memory_bytes']:
            raise RuntimeError('Descendant memory was not bounded within the declared limit')
        lines = execution['stdout'].decode('utf-8-sig').strip().splitlines()
        result = json.loads(lines[-1]) if lines else None
        if not isinstance(result, dict) or result.get('status') != 'verified-scoped' or result.get('original_objective_achieved') is not False:
            raise ValueError('Verifier did not report a scoped success')
        if not (OUTPUT / 'manifest.json').is_file():
            raise ValueError('Missing portable proof manifest')
        total = sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        if total > plan['artifact_budget_bytes']:
            raise ValueError('Proof artifact budget exceeded')
        record.update(status='verified-scoped', result=result, artifact_bytes=total,
                      bundle_manifest_sha256=digest(OUTPUT / 'manifest.json'))
    except Exception as exc:
        record.update(status='unresolved', reason=str(exc))
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(path, record)
    print(json.dumps({'status': record['status'], 'mode': mode, 'artifact_bytes': record.get('artifact_bytes'),
                      'reason': record.get('reason'), 'original_objective_achieved': False}, sort_keys=True), flush=True)
    if record['status'] != 'verified-scoped':
        sys.exit(2)


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: run_frozen_three_node_stage_v1.py prove|verify [attempt]')
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)