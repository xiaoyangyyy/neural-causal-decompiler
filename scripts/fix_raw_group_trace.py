from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8')
s=s.replace('''            if any(not np.allclose(values[0],v,rtol=1e-10,atol=1e-10) for v in values[1:]):raise RuntimeError("Structural equivalence group disagrees")
            records[self.address(path)]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","value":values[0],
                "visited":np.ones(worlds,bool),"intervened":path in requested,"members":list(members)}''','''            agrees=all(np.allclose(values[0],v,rtol=1e-10,atol=1e-10) for v in values[1:])
            if path in requested and not agrees:raise RuntimeError("Group intervention failed to preserve structural equivalence")
            if agrees:
                records[self.address(path)]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","value":values[0],
                    "visited":np.ones(worlds,bool),"intervened":path in requested,"members":list(members)}''')
p.write_text(s,encoding='utf-8')
