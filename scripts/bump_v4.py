from pathlib import Path
p=Path('pyproject.toml');s=p.read_text(encoding='utf-8').replace('version = "0.3.0"','version = "0.4.0"');p.write_text(s,encoding='utf-8')
Path('ncd/__init__.py').write_text('"""Neural Causal Decompiler research workflows."""\n__version__ = "0.4.0"\n',encoding='utf-8')
