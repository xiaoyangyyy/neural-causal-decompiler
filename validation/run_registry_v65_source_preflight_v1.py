from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];TEMP=ROOT/'validation/task_temp_E_v65_source';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0,str(ROOT/'validation/source_candidate_v65'))
import ncd
assert ncd.__version__=='0.65.0.dev1'
from ncd.proof_registry import validate_config
import json
c=json.loads((ROOT/'validation/original_proof_registry_protocol_v7.json').read_text(encoding='utf-8'));validate_config(c)
import pytest
sys.exit(pytest.main([str(ROOT/'tests/test_unified_proof_registry_v7.py'),str(ROOT/'tests/test_original_proof_integration.py'),
  '--import-mode=importlib','-q','--basetemp='+str(TEMP/'test_run_0000'),
  '--junitxml='+str(ROOT/'validation/pytest_registry_v65_source_v1.xml')]))
