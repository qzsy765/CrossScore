#!/usr/bin/env python3
"""ID-only blind freeze for six 2x2 factorial ImageNet-200 pilot runs.
Never open an OOD dataset, OOD metrics JSON, or tune using OOD.
"""
from pathlib import Path
import os
import json,datetime,csv,numpy as np
P=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
E=P/'P1_7C_factorial_IDonly_20261026'
SEEDS=(42,43)
VARIANTS=('T1_E1','T1_E0','T6_E1','T6_E0')
rows=[]
for v in VARIANTS:
 for seed in SEEDS:
  if v=='T1_E1':
   src=P/'runs'/f'CrossScore_T6_Corrected_seed{seed}'
   # Consume only frozen pre-extracted ID metrics, not any OOD-containing records.
   ref=json.loads((E/'FROZEN_P1_6_REFERENCE_ID_ONLY_20261026.json').read_text())
   matching=[r for r in ref['seeds'] if r['seed']==seed]
   assert len(matching)==1
   d=matching[0]
   val=float(d['reference_best_val_acc'])
   acc=float(d['reference_id_test_top1'])
   mean=float(d['reference_id_energy_mean'])
   sd=float(d['reference_id_energy_std'])
   epoch=None
  else:
   src=E/f'{v}_seed{seed}'
   cp=src/'COMPLETE.json';dp=src/'ID_ONLY_DIAGNOSTICS.json'
   if not cp.exists() or not dp.exists():raise FileNotFoundError(f'ID-only result is not complete: {src}')
   check=json.loads(cp.read_text())
   d=json.loads(dp.read_text())
   if check.get('run_type')!='EXPLORATORY_FACTORIAL_ID_ONLY':raise RuntimeError('unexpected training protocol')
   val=float(check['best_val_acc'])
   acc=float(d['id_test_top1']);mean=float(d['score_stats']['energy']['mean']);sd=float(d['score_stats']['energy']['std'])
   epoch=d['checkpoint_epoch']
  rows.append({'variant':v,'seed':seed,'best_val_acc':val,'id_test_top1':acc,'id_energy_mean':mean,'id_energy_std':sd,
   'id_classification_gate_passed':bool(val>=0.80),'checkpoint_epoch':epoch,'source_dir':str(src)})
assert not (E/'ID_ONLY_FROZEN_RESULTS.json').exists(), 'already frozen - do not overwrite'
with (E/'ID_only_seed_table.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]))
 w.writeheader();w.writerows(rows)
agg=[]
for v in VARIANTS:
 this=[x for x in rows if x['variant']==v]
 agg.append({'variant':v,'n_seeds':2,'mean_val_acc':float(np.mean([x['best_val_acc'] for x in this])),
 'mean_test_acc':float(np.mean([x['id_test_top1'] for x in this])),
 'mean_energy_sd':float(np.mean([x['id_energy_std'] for x in this])),
 'both_seeds_pass_id_gate':all(x['id_classification_gate_passed'] for x in this)})
plan=json.loads((E/'FROZEN_PLAN_20261026.json').read_text())
record={'status':'ID_ONLY_FROZEN_WITHOUT_OOD','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'n_new_runs_complete':6,'n_preexisting_reference_runs':2,'seeds':list(SEEDS),'results':agg,
 'warning':'n=2 per arm: descriptive mechanism pilot only. OOD may be opened only after separate ID-only freeze gate; do not overwrite negative formal P1-6.'}
with (E/'ID_ONLY_FROZEN_RESULTS.json').open('x') as f: f.write(json.dumps(record,indent=2)+'\n')
print('ID_ONLY_FREEZE_PASS',json.dumps(agg,indent=2))

