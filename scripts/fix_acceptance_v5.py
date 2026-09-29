from pathlib import Path
p=Path('scripts/acceptance_release_v5.py');s=p.read_text(encoding='utf-8');s=s.replace("if out.exists():raise FileExistsError('Preserve prior 0.5 acceptance');out.mkdir()","if out.exists():raise FileExistsError('Preserve prior 0.5 acceptance')\nout.mkdir()")
p.write_text(s,encoding='utf-8')
