from pathlib import Path
p=Path('ncd/cli.py');s=p.read_text(encoding='utf-8')
s=s.replace('("joint", "numeric", "raw-numeric", "guided", "relational")','("joint", "numeric", "raw-numeric", "dependence-numeric", "guided", "relational")')
s=s.replace('("numeric", "raw-numeric", "guided")','("numeric", "raw-numeric", "dependence-numeric", "guided")')
s=s.replace('''            elif args.command=="raw-numeric":
                from .raw_numeric_experiment import RawNumericConfig,run_raw_numeric
                config=RawNumericConfig.quick() if args.quick else RawNumericConfig()
                if args.seed is not None:config.seed=args.seed
                result=run_raw_numeric(args.output,args.source,config)
''','''            elif args.command in ("raw-numeric","dependence-numeric"):
                from .raw_numeric_experiment import RawNumericConfig,run_raw_numeric
                config=RawNumericConfig.quick() if args.quick else RawNumericConfig()
                config.target_scope="dependence" if args.command=="dependence-numeric" else "base"
                if args.seed is not None:config.seed=args.seed
                result=run_raw_numeric(args.output,args.source,config)
''')
s=s.replace('''            elif args.command=="verify-raw-numeric":
                from .raw_numeric_experiment import verify_raw_numeric
                verifier=verify_raw_numeric
''','''            elif args.command in ("verify-raw-numeric","verify-dependence-numeric"):
                from .raw_numeric_experiment import verify_raw_numeric
                verifier=verify_raw_numeric
''')
p.write_text(s,encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');s+='''

Dependence-kernel step audit CLI:

```powershell
python -m ncd dependence-numeric --source runs/joint_seed191 --output runs/my_dependence --seed 893
python -m ncd verify-dependence-numeric runs/my_dependence
```
''';p.write_text(s,encoding='utf-8')
