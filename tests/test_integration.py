from pathlib import Path
import json
import pytest
from ncd.experiment import Config,run
from ncd.verify import verify
from ncd.io import save_json,digest

def test_full_pipeline_and_independent_replay(tmp_path):
    output=tmp_path/"experiment"
    summary=run(output,Config.quick(seed=17))
    result=verify(output)
    assert result["status"]=="verified"
    assert result["worlds_replayed"]>800
    assert set(summary["evaluation"])=={"test_id","test_function","test_noise","test_scale","test_intervention"}
    assert (output/"report.html").is_file()
    with pytest.raises(FileExistsError): run(output,Config.quick())
    # Integrity check is necessary but not sufficient: re-sign a false metric.
    changed=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    changed["evaluation"]["test_id"]["full"]["fidelity"]=.123456
    save_json(output/"summary.json",changed)
    with pytest.raises(ValueError,match="integrity"): verify(output)
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["summary.json"]=digest(output/"summary.json")
    save_json(output/"manifest.json",manifest)
    with pytest.raises(ValueError,match="metrics"): verify(output)

def test_invalid_config(tmp_path):
    with pytest.raises(ValueError): run(tmp_path/"invalid",Config(epochs=0))
    assert not (tmp_path/"invalid").exists()
