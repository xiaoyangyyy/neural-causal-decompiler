from pathlib import Path
p=Path('ncd/cli.py');s=p.read_text(encoding='utf-8')
s=s.replace('for command in ("joint", "numeric", "guided", "relational"):', 'for command in ("joint", "numeric", "raw-numeric", "guided", "relational"):')
s=s.replace('if command in ("numeric", "guided"):', 'if command in ("numeric", "raw-numeric", "guided"):')
s=s.replace('elif args.command in ("joint", "numeric", "guided", "relational"):', 'elif args.command in ("joint", "numeric", "raw-numeric", "guided", "relational"):')
s=s.replace('''            if args.command=="numeric":
                from .numeric_experiment import run_numeric
                result=run_numeric(args.output,args.source,seed=291 if args.seed is None else args.seed,quick=args.quick)
            else:''','''            if args.command=="numeric":
                from .numeric_experiment import run_numeric
                result=run_numeric(args.output,args.source,seed=291 if args.seed is None else args.seed,quick=args.quick)
            elif args.command=="raw-numeric":
                from .raw_numeric_experiment import RawNumericConfig,run_raw_numeric
                config=RawNumericConfig.quick() if args.quick else RawNumericConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_raw_numeric(args.output,args.source,config)
            else:''')
s=s.replace('''            elif args.command=="verify-guided":''','''            elif args.command=="verify-raw-numeric":
                from .raw_numeric_experiment import verify_raw_numeric
                verifier=verify_raw_numeric
            elif args.command=="verify-guided":''')
p.write_text(s,encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');s+='''

Raw statistical operation audit CLI:

```powershell
python -m ncd raw-numeric --source runs/joint_seed191 --output runs/my_raw_numeric --seed 793
python -m ncd verify-raw-numeric runs/my_raw_numeric
```

Use `--quick` for an integration profile. The command traces eight internal
mean/variance/std/correlation groups from raw samples; dependence and regression
internals remain outside this trace.
''';p.write_text(s,encoding='utf-8')
