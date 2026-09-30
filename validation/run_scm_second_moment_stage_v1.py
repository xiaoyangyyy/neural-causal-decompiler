"""Job-supervised installed-wheel proof and independent replay for Student5 tails."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

from ncd.proof_process import run_isolated

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'validation'
PLAN = V / 'scm_second_moment_protocol_v1.json'
OUTPUT = ROOT / 'runs/scm_second_moment_completion_v1'
ENV = V / 'scm_second_moment_env_v1/Scripts/python.exe'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, row):
    path.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def main(mode, attempt=0):
    if mode not in ('prove', 'verify') or type(attempt) is not int or not 0 <= attempt <= 9999:
        raise ValueError('Expected prove or verify with a nonnegative attempt index')
    if not PLAN.is_file() or not ENV.is_file():
        raise ValueError('Frozen protocol or installed verifier missing')
    plan = json.loads(PLAN.read_bytes())
    if plan['schema'] != 'ncd.scm-second-moment-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong proof scope')
    if plan['memory_bytes'] != 8589934592 or plan['threads'] != 2 or not 0 < plan['stage_seconds'] <= 43200:
        raise ValueError('Invalid resource budget')
    if mode == 'prove' and (OUTPUT / 'manifest.json').exists():
        raise FileExistsError('Completed proof bundle is immutable')
    if mode == 'verify' and not (OUTPUT / 'manifest.json').is_file():
        raise ValueError('No generated proof bundle to replay')
    suffix = mode + '_attempt%04d' % attempt
    path = V / ('scm_second_moment_stage_v1_' + suffix + '.json')
    if path.exists():
        raise FileExistsError('Retain previous stage record')
    command = [str(ENV), '-I', '-B', '-m', 'scm_second_moment']
    command += ['prove', '--config', str(PLAN)] if mode == 'prove' else ['verify-proof', str(OUTPUT)]
    os.environ.update(TEMP=str(V), TMP=str(V), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    row = {
        'schema': 'ncd.scm-second-moment-stage.v1', 'mode': mode,
        'status': 'running', 'pid': os.getpid(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(PLAN), 'runner_sha256': digest(__file__),
        'original_objective_achieved': False,
    }
    save(path, row)
    try:
        execution = run_isolated(command, ROOT, plan['stage_seconds'], plan['memory_bytes'])
        (V / ('scm_second_moment_stage_v1_' + suffix + '.stdout.log')).write_bytes(execution['stdout'])
        (V / ('scm_second_moment_stage_v1_' + suffix + '.stderr.log')).write_bytes(execution['stderr'])
        row['resource_guard'] = execution['resources']
        row['exit_code'] = execution['exit_code']
        if execution['exit_code'] or execution['resources']['timeout']:
            raise RuntimeError('Isolated installed proof failed or timed out')
        if not execution['resources'].get('descendants_included') or execution['resources'].get('peak_job_memory_bytes', plan['memory_bytes'] + 1) > plan['memory_bytes']:
            raise RuntimeError('Descendant memory was not within the declared limit')
        lines = execution['stdout'].decode('utf-8-sig').strip().splitlines()
        result = json.loads(lines[-1]) if lines else None
        expected = {'status': 'verified-scoped', 'prior_finite_second_moments': 295,
                    'new_infinite_second_moments': 5, 'classified_observational_worlds': 300,
                    'original_objective_achieved': False}
        if result != expected:
            raise ValueError('Installed verifier reported a different scientific conclusion')
        if not (OUTPUT / 'manifest.json').is_file():
            raise ValueError('Portable proof manifest missing')
        size = sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        if size > plan['artifact_budget_bytes']:
            raise ValueError('Proof artifact budget exceeded')
        row.update(status='verified-scoped', result=result, artifact_bytes=size,
                   bundle_manifest_sha256=digest(OUTPUT / 'manifest.json'))
    except Exception as exc:
        row.update(status='unresolved', reason=str(exc))
    row['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(path, row)
    print(json.dumps({'status': row['status'], 'mode': mode, 'reason': row.get('reason'),
                      'original_objective_achieved': False}, sort_keys=True), flush=True)
    if row['status'] != 'verified-scoped':
        raise SystemExit(2)


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: run_scm_second_moment_stage_v1.py prove|verify [attempt]')
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)