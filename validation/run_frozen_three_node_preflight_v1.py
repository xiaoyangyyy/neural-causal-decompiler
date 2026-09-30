"""Durable preflight for the scoped three-node frozen mechanism theorem."""
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation/frozen_three_node_package_v1'))
from frozen_three_node.proof import SOURCE_NAMES, derive, digest, loaded_source_sha


def main():
    source = ROOT / 'validation/task_temp_E_frozen3_v1/source'
    if {p.name for p in source.iterdir() if p.is_file()} != set(SOURCE_NAMES):
        raise ValueError('Incomplete retained frozen preflight sources')
    start = time.monotonic()
    print('started', datetime.now(timezone.utc).isoformat(), flush=True)
    roots, certificate, program = derive(source)
    if certificate['status'] != 'proved-scoped' or len(certificate['joint_intervention_bounds']) != 8:
        raise ValueError('Composition conclusion was not proved')
    receipt = {
        'schema': 'ncd.frozen-three-node-preflight.v1',
        'status': 'verified-preflight',
        'source_sha256': {name: digest((source / name).read_bytes()) for name in SOURCE_NAMES},
        'loaded_package_sha256': loaded_source_sha(),
        'root_conclusions': {key: value['certificate']['status'] for key, value in roots.items()},
        'composition_conclusion': certificate['status'],
        'compatible_intervention_masks': len(certificate['joint_intervention_bounds']),
        'local_absolute_error_upper': certificate['local_absolute_error_upper'],
        'worst_joint_l1_error_upper': certificate['worst_joint_l1_error_upper'],
        'worst_joint_l1_error_float': float(Q(certificate['worst_joint_l1_error_upper'])),
        'program_schema': program['schema'],
        'elapsed_seconds': round(time.monotonic() - start, 3),
        'original_objective_achieved': False,
    }
    target = ROOT / 'validation/frozen_three_node_preflight_v1.json'
    if target.exists():
        raise FileExistsError('Retain prior preflight result')
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('verified-preflight', receipt['elapsed_seconds'], receipt['worst_joint_l1_error_float'], flush=True)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    main()