from pathlib import Path
from datetime import datetime,timezone
import json,sys
import ncd
from ncd.original_confirmation import run_confirmation
ROOT=Path(__file__).resolve().parents[1]
if not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v59_env'):
    raise RuntimeError('Confirmation replay imported source rather than installed wheel')
remaining=(datetime.fromisoformat(sys.argv[1])-datetime.now(timezone.utc)).total_seconds()-5
if remaining<=0:raise TimeoutError('Original 12-hour deadline exhausted')
result=run_confirmation(ROOT/'validation/original_confirmation_protocol.json',verify_only=True,seconds=remaining)
(ROOT/'validation/original_confirmation_launch/replay_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result['state'],result['computed_worlds'],result['science_passed'],flush=True)
