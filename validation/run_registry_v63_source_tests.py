from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/'validation/task_temp_E_v63_source';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(ROOT/'validation/source_candidate_v63'))
import pytest
raise SystemExit(pytest.main([str(ROOT/'tests/test_unified_proof_registry_v5.py'),'--import-mode=importlib','-q','--basetemp='+str(ROOT/'validation/registry_v63_source_test_temp'),'--junitxml='+str(ROOT/'validation/pytest_registry_v63_source_v1.xml')]))
