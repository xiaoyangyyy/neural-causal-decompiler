from pathlib import Path
p=Path('ncd/cli.py');s=p.read_text(encoding='utf-8')
s=s.replace('"dependence-numeric", "guided"','"dependence-numeric", "regression-numeric", "guided"')
s=s.replace('"dependence-numeric", "guided")','"dependence-numeric", "regression-numeric", "guided")')
s=s.replace('args.command in ("raw-numeric","dependence-numeric")','args.command in ("raw-numeric","dependence-numeric","regression-numeric")')
s=s.replace('config.target_scope="dependence" if args.command=="dependence-numeric" else "base"','config.target_scope="dependence" if args.command=="dependence-numeric" else "regression" if args.command=="regression-numeric" else "base"')
s=s.replace('("verify-raw-numeric","verify-dependence-numeric")','("verify-raw-numeric","verify-dependence-numeric","verify-regression-numeric")')
p.write_text(s,encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');s+='''

Cross-fit regression step audit CLI:

```powershell
python -m ncd regression-numeric --source runs/joint_seed191 --output runs/my_regression --seed 993
python -m ncd verify-regression-numeric runs/my_regression
```
''';p.write_text(s,encoding='utf-8')
