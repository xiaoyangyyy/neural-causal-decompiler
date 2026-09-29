from pathlib import Path
p=Path('scripts/verify_relational_program_experiment.py');s=p.read_text(encoding='utf-8');s=s.replace("assert [w.metadata() for w in worlds]==saved_worlds[str(seed)][key]","assert json.loads(json.dumps([w.metadata() for w in worlds]))==saved_worlds[str(seed)][key]");p.write_text(s,encoding='utf-8')
