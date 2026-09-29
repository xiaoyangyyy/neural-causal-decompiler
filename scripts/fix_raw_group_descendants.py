from pathlib import Path
p=Path('ncd/raw_program_trace.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        records={self.address(path):{"kind":self.catalog[path]["kind"],"semantic_kind":self.catalog[path]["semantic_kind"],
            "op":self.catalog[path]["op"],"value":np.asarray(collected[path]),"visited":np.ones(worlds,bool),
            "intervened":path in expanded} for path in self._physical}''','''        records={self.address(path):{"kind":self.catalog[path]["kind"],"semantic_kind":self.catalog[path]["semantic_kind"],
            "op":self.catalog[path]["op"],"value":np.asarray(collected[path]),"visited":np.ones(worlds,bool),
            "intervened":path in expanded} for path in self._physical if len(collected[path])==worlds}''')
s=s.replace('''            values=[np.asarray(collected[p]) for p in members]
            agrees=all''','''            if any(len(collected[p])!=worlds for p in members):
                if path in requested:raise RuntimeError("Group intervention made a member unreachable")
                continue
            values=[np.asarray(collected[p]) for p in members]
            agrees=all''')
p.write_text(s,encoding='utf-8')
