"""CLI for scoped three-node frozen mechanism composition and replay."""
import argparse
import json
import math
from pathlib import Path

from .proof import generate, read, verify_bundle


def execute(program, interventions):
    import numpy as np
    from proof_workbench.piecewise_mechanism import execute_program
    if program.get('schema') != 'ncd.frozen-three-node-program.v1' or program.get('noise_model') is not None:
        raise ValueError('Expected deterministic frozen three-node program')
    if program.get('graph') != [[0, 0, 0], [1, 0, 0], [1, 0, 0]]:
        raise ValueError('Wrong learned parent graph')
    if type(interventions) is not dict:
        raise ValueError('Interventions must be a mapping of node IDs to observed values')
    do = {}
    for key, value in interventions.items():
        if key not in ('0', '1', '2') or type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Intervention value outside declared box')
        do[int(key)] = float(value)
    values = np.zeros((1, 3), dtype=float)
    for j in (1, 2):
        expression = program['root_expressions'][str(j)]
        if expression.get('op') != 'constant' or set(expression) != {'op', 'value'}:
            raise ValueError('Expected checked constant root')
        values[0, j] = do.get(j, expression['value'])
    values[0, 0] = do[0] if 0 in do else execute_program(program['child_piecewise_program'], values)[0]
    return {'values': values[0].tolist(), 'intervened_nodes': sorted(do),
            'runtime_semantics': 'floating execution; proof certificate concerns ideal real evaluation'}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prove').add_argument('--config', required=True)
    commands.add_parser('verify-proof').add_argument('bundle')
    run = commands.add_parser('execute')
    run.add_argument('--program', required=True)
    run.add_argument('--interventions', required=True)
    args = parser.parse_args()
    if args.command == 'prove':
        result = generate(args.config)
    elif args.command == 'verify-proof':
        result = verify_bundle(args.bundle)
    else:
        result = execute(read(args.program), json.loads(args.interventions))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()