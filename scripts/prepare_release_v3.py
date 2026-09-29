from pathlib import Path
p=Path('pyproject.toml');s=p.read_text(encoding='utf-8').replace('version = "0.2.0"','version = "0.3.0"').replace('Rule extraction and counterexample auditing for bivariate causal discovery','Causal discovery program extraction, intervention audits and SCM recovery');p.write_text(s,encoding='utf-8')
Path('ncd/__init__.py').write_text('"""Neural Causal Decompiler research workflows."""\n__version__ = "0.3.0"\n',encoding='utf-8')
p=Path('README.md');s=p.read_text(encoding='utf-8');s+='''

## Version 0.3 experiment commands

The joint, numeric, guided and relational experiments are available through the
same CLI as the original bivariate and graph workflows. Each command preserves
its own artifact directory and has an independent replay command.

```powershell
python -m ncd joint --output runs/my_joint --quick --seed 191
python -m ncd verify-joint runs/my_joint
python -m ncd numeric --source runs/my_joint --output runs/my_numeric --quick --seed 291
python -m ncd verify-numeric runs/my_numeric
python -m ncd guided --source runs/my_joint --output runs/my_guided --quick --seed 391
python -m ncd verify-guided runs/my_guided
python -m ncd relational --output runs/my_relational --quick --seed 491
python -m ncd verify-relational runs/my_relational
```

Omit `--quick` for the respective full experiment defaults. Quick profiles are
integration checks, not confirmatory scientific experiments. Numeric and guided
runs use a frozen joint checkpoint; numeric audits require a program with enough
supported operands and fail explicitly when this prerequisite is absent.
`run-all` retains its original bivariate plus multivariate scope; it does not run
these additional audits. See [the relational comparison](RESULTS_RELATIONAL.md)
for the independently replayed 493/494 experiments. Their mixed results do not
establish a reliable improvement in exact graph recovery.
''';p.write_text(s,encoding='utf-8')
