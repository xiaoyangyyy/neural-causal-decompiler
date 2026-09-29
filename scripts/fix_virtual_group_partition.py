from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8').replace('for path in virtual:dep_grouped.setdefault(dep_keys[path],[]).append(path)','for path in virtual:\n            if path in dep_keys:dep_grouped.setdefault(dep_keys[path],[]).append(path)');p.write_text(s,encoding='utf-8')
