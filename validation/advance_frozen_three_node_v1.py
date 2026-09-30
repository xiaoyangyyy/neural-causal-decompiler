"""Durable sequential continuation of the fixed three-node proof and truth audit.

The worker waits for the already-running preflight process. It never restarts
that process or overlaps its own formal heavy stages.
"""
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'validation'
RECORD = V / 'frozen_three_node_chain_v1.json'
LAUNCH = V / 'frozen_three_node_preflight_launch_v1_r2.json'
PREFLIGHT = V / 'frozen_three_node_preflight_v1.json'
PREVIEW = V / 'task_temp_E_frozen3_v1/protocol_preview.json'
PLAN = V / 'frozen_three_node_protocol_v1.json'
INSTALLED = V / 'frozen_three_node_env_v1/Lib/site-packages/frozen_three_node'
SUPERVISOR = V / 'wheel_v65_env/Scripts/python.exe'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(row):
    data = (json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    temporary = RECORD.with_name(RECORD.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, RECORD)


def wait_for_original_preflight(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        time.sleep(2)
        if not PREFLIGHT.is_file():
            raise RuntimeError('Original preflight process ended without a result')
        return
    try:
        while True:
            status = kernel.WaitForSingleObject(handle, 30000)
            if status == 0:
                break
            if status != 0x102:
                raise RuntimeError('Cannot observe original preflight process')
    finally:
        kernel.CloseHandle(handle)
    if not PREFLIGHT.is_file():
        raise RuntimeError('Original preflight process ended without a result')


def freeze_candidate_plan():
    receipt = read(PREFLIGHT)
    preview = read(PREVIEW)
    launch = read(LAUNCH)
    if launch['status'] != 'running' or launch['package_source_sha256'] != preview['loaded_package_sha256']['proof.py']:
        raise ValueError('Preflight launch differs from prepared package')
    if receipt['schema'] != 'ncd.frozen-three-node-preflight.v1' or receipt['status'] != 'verified-preflight':
        raise ValueError('Preflight did not establish a verified scoped result')
    if receipt['composition_conclusion'] != 'proved-scoped' or receipt['root_conclusions'] != {'1': 'proved', '2': 'proved'}:
        raise ValueError('Root or composition conclusion failed')
    if receipt['compatible_intervention_masks'] != 8 or receipt['original_objective_achieved'] is not False:
        raise ValueError('Preflight intervention coverage or scope changed')
    if Fraction(receipt['worst_joint_l1_error_upper']) < 0:
        raise ValueError('Invalid joint error bound')
    if receipt['loaded_package_sha256'] != preview['loaded_package_sha256'] or receipt['source_sha256'] != preview['source_sha256']:
        raise ValueError('Preflight code or source differs from prepared protocol')
    for name, expected in preview['loaded_package_sha256'].items():
        if digest(INSTALLED / name) != expected:
            raise ValueError('Installed proof package changed: ' + name)
    source = V / 'task_temp_E_frozen3_v1/source'
    for name, expected in preview['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('Preflight historical source changed: ' + name)
    if preview['schema'] != 'ncd.frozen-three-node-plan.v1' or preview['original_atomic_claim_counts'] != {'proved': 0, 'refuted': 2, 'unresolved': 36}:
        raise ValueError('Prepared protocol changed original claim scope')
    data = PREVIEW.read_bytes()
    if PLAN.exists():
        if PLAN.read_bytes() != data:
            raise ValueError('Previously frozen candidate protocol differs')
    else:
        with PLAN.open('xb') as stream:
            stream.write(data)
    launch.update(status='verified-preflight', finished_utc=now(),
                  preflight_sha256=digest(PREFLIGHT), protocol_sha256=digest(PLAN))
    LAUNCH.write_text(json.dumps(launch, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def run_step(row, name, script, *args):
    row['phase'] = name
    row['updated_utc'] = now()
    save(row)
    completed = subprocess.run([str(SUPERVISOR), '-I', '-B', str(V / script), *args],
                               cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace',
                               creationflags=0x08000000)
    row.setdefault('steps', {})[name] = {
        'exit_code': completed.returncode, 'stdout_tail': completed.stdout[-2000:],
        'stderr_tail': completed.stderr[-2000:], 'finished_utc': now(),
    }
    save(row)
    if completed.returncode:
        raise RuntimeError(name + ' returned ' + str(completed.returncode))
    return completed


def checked_stage(stem, mode):
    record = read(V / (stem + '_' + mode + '_attempt0000.json'))
    if record['mode'] != mode or record['status'] != 'verified-scoped' or record['exit_code'] != 0:
        raise ValueError(stem + ' ' + mode + ' stage did not verify')
    resources = record['resource_guard']
    if resources['timeout'] or not resources['descendants_included']:
        raise ValueError(stem + ' ' + mode + ' resource guard failed')
    return record


def main():
    if RECORD.exists():
        raise FileExistsError('Prior continuation record is retained')
    if not SUPERVISOR.is_file():
        raise ValueError('Installed supervisor environment missing')
    launch = read(LAUNCH)
    if launch['schema'] != 'ncd.frozen-three-node-preflight-launch.v1' or launch['status'] != 'running':
        raise ValueError('Expected exact live preflight launch')
    row = {'schema': 'ncd.frozen-three-node-chain.v1', 'status': 'waiting-preflight',
           'pid': os.getpid(), 'preflight_pid': launch['pid'],
           'started_utc': now(), 'original_objective_achieved': False, 'steps': {}}
    save(row)
    try:
        wait_for_original_preflight(launch['pid'])
        freeze_candidate_plan()
        row['preflight_sha256'] = digest(PREFLIGHT)
        row['protocol_sha256'] = digest(PLAN)
        save(row)
        run_step(row, 'candidate_prove', 'run_frozen_three_node_stage_v1.py', 'prove')
        checked_stage('frozen_three_node_stage_v1', 'prove')
        run_step(row, 'candidate_verify', 'run_frozen_three_node_stage_v1.py', 'verify')
        checked_stage('frozen_three_node_stage_v1', 'verify')
        run_step(row, 'candidate_accept', 'check_frozen_three_node_acceptance_v1.py')
        candidate = read(V / 'frozen_three_node_acceptance_v1.json')
        if candidate['status'] != 'accepted-scoped' or candidate['original_objective_achieved'] is not False:
            raise ValueError('Candidate acceptance scope changed')
        row['candidate_acceptance_sha256'] = digest(V / 'frozen_three_node_acceptance_v1.json')
        save(row)
        run_step(row, 'truth_freeze', 'run_frozen_truth_gap_v1.py', 'freeze')
        run_step(row, 'truth_prove', 'run_frozen_truth_gap_v1.py', 'prove')
        checked_stage('frozen_truth_gap_stage_v1', 'prove')
        run_step(row, 'truth_verify', 'run_frozen_truth_gap_v1.py', 'verify')
        checked_stage('frozen_truth_gap_stage_v1', 'verify')
        run_step(row, 'truth_accept', 'check_frozen_truth_gap_acceptance_v1.py')
        truth = read(V / 'frozen_truth_gap_acceptance_v1.json')
        if truth['status'] != 'accepted-scoped-refutation' or truth['original_objective_achieved'] is not False:
            raise ValueError('Truth-only acceptance scope changed')
        row['truth_acceptance_sha256'] = digest(V / 'frozen_truth_gap_acceptance_v1.json')
        row['status'] = 'accepted-scoped-only'
    except Exception as exc:
        row['status'] = 'unresolved'
        row['reason'] = str(exc)
    row['finished_utc'] = now()
    save(row)
    print(json.dumps({'status': row['status'], 'phase': row.get('phase'),
                      'reason': row.get('reason'), 'original_objective_achieved': False},
                     sort_keys=True), flush=True)
    if row['status'] != 'accepted-scoped-only':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
