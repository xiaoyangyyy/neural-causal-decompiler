from pathlib import Path
s=Path('scripts/acceptance_release_v3.py').read_text(encoding='utf-8-sig')
s=s.replace("output=root/'validation/wheel_v3_run'", "output=root/'validation/wheel_v3_full'")
a=s.index('commands=[]');b=s.index('results=[]',a)
s=s[:a]+"commands=[['run-all','--output',str(output/'experiment'),'--quick','--seed','91'],['verify-all',str(output/'experiment')]]\n"+s[b:]
s=s.replace('fresh quick runs and independent replay of all four added experiment commands; scientific claims not certified','fresh quick run-all and independent verify-all from installed release; scientific claims not certified')
Path('scripts/acceptance_release_v3_full.py').write_text(s,encoding='utf-8')
