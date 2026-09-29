from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/"validation/task_temp_E_v62_source"
TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP))
sys.path.insert(0,str(ROOT/"validation/source_candidate_v62"))
import pytest
raise SystemExit(pytest.main([str(ROOT/"tests/test_unified_proof_registry_v4.py"),"--import-mode=importlib","-q","--basetemp="+str(ROOT/"validation/registry_v62_source_test_temp"),"--junitxml="+str(ROOT/"validation/pytest_registry_v62_source_v1.xml")]))
