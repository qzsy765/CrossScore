from pathlib import Path
import json,hashlib
base=Path(__file__).resolve().parents[1]
d=json.loads((base/'evidence/SHA256.json').read_text())
bad=[rel for rel,h in d.items() if not (base/rel).exists() or hashlib.sha256((base/rel).read_bytes()).hexdigest()!=h]
print('PASS' if not bad else 'FAIL',len(d),'files audited')
if bad:raise SystemExit('Mismatched: '+str(bad[:10]))
