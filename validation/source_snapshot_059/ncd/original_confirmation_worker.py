"""One isolated confirmation unit, supervised by the bounded coordinator."""
import argparse
import json
from pathlib import Path
from .io import read_json,digest
from .model import set_seed
from .original_confirmation import (check_protocol,_training_config,_mechanism_config,_world,
    _teachers,_world_compute,verify_world,_unit_name)
from .active_intervention_experiment import run_active_intervention,verify_active_intervention


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--task',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--unit')
    args=p.parse_args();path=Path(args.protocol).resolve();protocol=read_json(path)
    root=(path.parent/protocol.get('root','..')).resolve();units=check_protocol(protocol,root)
    output=Path(args.output).resolve();study=(root/protocol['output']).resolve()
    if not output.is_relative_to(study) or args.seed not in protocol['seeds']:raise ValueError('Worker outside declared confirmation')
    set_seed(args.seed)
    if args.task=='train':run_active_intervention(output,_training_config(protocol,args.seed))
    elif args.task=='verify_training':verify_active_intervention(output)
    else:
        unit=tuple(json.loads(args.unit))
        if unit not in units or unit[0]!=args.seed or output.parent.name!=_unit_name(unit):raise ValueError('Worker unit mismatch')
        meta=read_json(study/f'training_seed_{args.seed}'/'complete.json')
        source=study/f'training_seed_{args.seed}'/meta['attempt']
        teachers=_teachers(source);world=_world(protocol,unit);config=_mechanism_config(protocol,args.seed)
        if args.task=='world':_world_compute(output,world,unit,teachers,config,digest(path),meta['teacher_sha256'])
        elif args.task=='verify_world':verify_world(output,world,unit,teachers,config,digest(path),meta['teacher_sha256'])
        else:raise ValueError('Unknown worker task')

if __name__=='__main__':main()
