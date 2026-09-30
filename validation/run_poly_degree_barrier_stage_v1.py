"""Job-supervised installed-wheel generation and replay of the degree barrier."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

from ncd.proof_process import run_isolated

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'validation'
PLAN = V / 'poly_degree_barrier_protocol_v1.json'
OUTPUT = ROOT / 'runs/frozen_mechanism_polynomial_barrier_v1'
ENV = V / 'poly_degree_barrier_env_v1/Scripts/python.exe'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, row):
    path.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def main(mode, attempt=0):
    if mode not in ('prove', 'verify') or type(attempt) is not int or not 0 <= attempt <= 9999:
        raise ValueError('Expected prove or verify with a nonnegative attempt index')
    if not PLAN.is_file() or not ENV.is_file():
        raise ValueError('Frozen plan or independent verifier missing')
    plan = json.loads(PLAN.read_bytes())
    if plan['schema'] != 'ncd.polynomial-degree-barrier-plan.v1' or plan['original_objective_achieved'] is not False:
        raise ValueError('Wrong scientific scope')
    if plan['memory_bytes'] != 8589934592 or plan['threads'] != 2 or not 0 < plan['stage_seconds'] <= 43200:
        raise ValueError('Invalid resource limit')
    if mode == 'prove' and (OUTPUT / 'manifest.json').exists():
        raise FileExistsError('Proof bundle is already sealed')
    if mode == 'verify' and not (OUTPUT / 'manifest.json').is_file():
        raise ValueError('Proof bundle has not been generated')
    suffix = mode + '_attempt%04d' % attempt
    receipt = V / ('poly_degree_barrier_stage_v1_' + suffix + '.json')
    if receipt.exists():
        raise FileExistsError('Retain prior attempt record')
    command = [str(ENV), '-I', '-B', '-m', 'poly_degree_barrier']
    command += ['prove', '--config', str(PLAN)] if mode == 'prove' else ['verify-proof', str(OUTPUT)]
    os.environ.update(TEMP=str(V), TMP=str(V), OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    row = {
        'schema': 'ncd.polynomial-degree-barrier-stage.v1',
        'mode': mode, 'status': 'running', 'pid': os.getpid(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest(PLAN), 'runner_sha256': digest(__file__),
        'original_objective_achieved': False,
    }
    save(receipt, row)
    try:
        execution = run_isolated(command, ROOT, plan['stage_seconds'], plan['memory_bytes'])
        (V / ('poly_degree_barrier_stage_v1_' + suffix + '.stdout.log')).write_bytes(execution['stdout'])
        (V / ('poly_degree_barrier_stage_v1_' + suffix + '.stderr.log')).write_bytes(execution['stderr'])
        row['resource_guard'] = execution['resources']
        row['exit_code'] = execution['exit_code']
        if execution['exit_code'] or execution['resources']['timeout']:
            raise RuntimeError('Isolated installed proof failed or timed out')
        if not execution['resources'].get('descendants_included') or execution['resources'].get('peak_job_memory_bytes', plan['memory_bytes'] + 1) > plan['memory_bytes']:
            raise RuntimeError('Descendant memory limit not enforced')
        lines = execution['stdout'].decode('utf-8-sig').strip().splitlines()
        result = json.loads(lines[-1]) if lines else None
        expected = {'status': 'verified-scoped', 'conclusion': 'refuted-scoped',
                    'degree_bound': 6, 'points': 8, 'original_objective_achieved': False}
        if result != expected:
            raise ValueError('Installed proof reported a different conclusion')
        if not (OUTPUT / 'manifest.json').is_file():
            raise ValueError('Proof manifest missing')
        size = sum(path.stat().st_size for path in OUTPUT.rglob('*') if path.is_file())
        if size > plan['artifact_budget_bytes']:
            raise ValueError('Proof artifact budget exceeded')
        row.update(status='verified-scoped', result=result, artifact_bytes=size,
                   bundle_manifest_sha256=digest(OUTPUT / 'manifest.json'))
    except Exception as exc:
        row.update(status='unresolved', reason=str(exc))
    row['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(receipt, row)
    print(json.dumps({'status': row['status'], 'mode': mode,
                      'reason': row.get('reason'), 'original_objective_achieved': False},
                     sort_keys=True), flush=True)
    if row['status'] != 'verified-scoped':
        raise SystemExit(2)


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: run_poly_degree_barrier_stage_v1.py prove|verify [attempt]')
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)