from pathlib import Path
p=Path('ncd/cli.py')
s=p.read_text(encoding='utf-8')
s=s.replace('"""CLI for legacy, full bivariate and multivariate research workflows."""','"""CLI for causal discovery, program extraction and intervention audits."""')
s=s.replace('    args=parser.parse_args(argv)', '''    for command in ("joint", "numeric", "guided", "relational"):
        p=sub.add_parser(command,help=f"Run {command} experiment")
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--quick",action="store_true",help="Small integration run")
        p.add_argument("--seed",type=int)
        if command in ("numeric", "guided"):
            p.add_argument("--source",type=Path,required=True,help="Frozen joint experiment directory")
        p=sub.add_parser(f"verify-{command}",help=f"Independently replay {command} experiment")
        p.add_argument("directory",type=Path)
    args=parser.parse_args(argv)''')
s=s.replace('        elif args.command=="run-all":', '''        elif args.command in ("joint", "numeric", "guided", "relational"):
            if args.command=="numeric":
                from .numeric_experiment import run_numeric
                result=run_numeric(args.output,args.source,seed=291 if args.seed is None else args.seed,quick=args.quick)
            else:
                if args.command=="joint":
                    from .joint_experiment import JointConfig,run_joint
                    config_class,runner=JointConfig,run_joint
                elif args.command=="guided":
                    from .guided_experiment import GuidedExperimentConfig,run_guided
                    config_class,runner=GuidedExperimentConfig,run_guided
                else:
                    from .relational_experiment import RelationalConfig,run_relational
                    config_class,runner=RelationalConfig,run_relational
                config=config_class.quick() if args.quick else config_class()
                if args.seed is not None:config.seed=args.seed
                result=runner(args.output,args.source,config) if args.command=="guided" else runner(args.output,config)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=="run-all":''')
s=s.replace('            else:\n                from .full_project import verify_all', '''            elif args.command=="verify-joint":
                from .joint_experiment import verify_joint
                verifier=verify_joint
            elif args.command=="verify-numeric":
                from .numeric_experiment import verify_numeric
                verifier=verify_numeric
            elif args.command=="verify-guided":
                from .guided_experiment import verify_guided_experiment
                verifier=verify_guided_experiment
            elif args.command=="verify-relational":
                from .relational_experiment import verify_relational
                verifier=verify_relational
            else:
                from .full_project import verify_all''')
s=s.replace('("run-all","Entire original research workflow")','("run-all","Bivariate research and multivariate SCM workflow")')
p.write_text(s,encoding='utf-8')
