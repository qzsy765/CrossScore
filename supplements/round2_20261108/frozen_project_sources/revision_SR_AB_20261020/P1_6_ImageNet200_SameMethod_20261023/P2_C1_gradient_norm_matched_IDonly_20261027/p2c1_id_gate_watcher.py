#!/usr/bin/env python3
"""Frozen P2C1 CPU gate: preserve every ID-only outcome, never evaluates OOD or launches models."""
from pathlib import Path
import json,hashlib,datetime,time,csv,os
import numpy as np
P=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023/P2_C1_gradient_norm_matched_IDonly_20261027')
SEEDS=[44,45,46];ARMS=['T1_E1_NORMMATCH','T6_E1_NORMMATCH']
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=json.loads((P/'FROZEN_WATCHER_PLAN.json').read_text())
 assert sha(Path(__file__))==plan['source_sha256']
 assert sha(P/'FROZEN_CONTROLLER_PLAN.json')==plan['controller_plan_sha256']
 assert sha(P/'FROZEN_AMENDMENT_01.json')==plan['amendment_sha256']
 assert sha(P/'train_p2c1_normmatch_v2.py')==plan['trainer_sha256']
 assert sha(P/'FROZEN_V2_SMOKE_GATE.json')==plan['preflight_gate_sha256']
 assert not (P/'P2C1_ID_ONLY_FROZEN_GATE.json').exists()
 print('P2C1_WATCH_STARTED',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 for tick in range(760):
  errors=[s for s in SEEDS if (P/f'node_seed{s}_ERROR.json').exists()]
  if errors:raise RuntimeError(('P2C1_NODE_FAILURE',errors))
  completed=[s for s in SEEDS if (P/f'node_seed{s}_DONE.json').exists()]
  if len(completed)==3:break
  if tick%10==0:print('P2C1_WAIT',len(completed),'/3',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
  time.sleep(90)
 else:raise TimeoutError('P2C1_ID_ONLY_NOT_COMPLETED_WITHIN_19H')
 allrows=[]
 for s in SEEDS:
  for arm in ARMS:
   name=f'{arm}_seed{s}';r=P/name
   ck=r/'best.pt';history=r/'history.csv'
   a=json.loads((r/'COMPLETE.json').read_text())
   idinfo=json.loads((r/'ID_ONLY_DIAGNOSTICS.json').read_text())
   man=json.loads((r/'manifest.json').read_text())
   assert a['complete_epochs']==60 and a['run_type']=='P2C1_NORM_MATCHED_ID_ONLY'
   assert man['seed']==s and man['variant']==arm
   assert man['student_probability_temperature']==(1. if arm.startswith('T1') else 6.)
   assert man['run_type']=='P2C1_NORM_MATCHED_ID_ONLY'
   assert idinfo['no_ood_evaluated']==True and idinfo['ID_test_N']==9000
   assert idinfo['run']==name and abs(idinfo['checkpoint_val_acc']-a['best_val_acc'])<1e-6
   with history.open() as f:rows=list(csv.DictReader(f))
   assert len(rows)==60 and [int(float(x['epoch'])) for x in rows]==list(range(1,61))
   values={field:np.array([float(x[field]) for x in rows],dtype=float) for field in ['grad_ratio_actual_mean','grad_coeff_mean','grad_coeff_saturated_fraction','skipped_amp_steps','val_acc']}
   assert all(np.isfinite(v).all() for v in values.values())
   meanratio=float(np.mean(values['grad_ratio_actual_mean']))
   saturation=float(np.mean(values['grad_coeff_saturated_fraction']))
   row={'seed':s,'arm':arm,'best_ID_val_top1':a['best_val_acc'],'ID_test_top1':idinfo['id_test_top1'],
     'ID_energy_std':idinfo['score_stats']['energy']['std'],
     'observed_logit_grad_ratio_mean':meanratio,'coefficient_watchdog_fraction':saturation,
     'AMP_skipped_steps_total':int(np.sum(values['skipped_amp_steps'])),
     'checkpoint_sha256':sha(ck),'ID_diagnostics_sha256':sha(r/'ID_ONLY_DIAGNOSTICS.json'),'training_complete_sha256':sha(r/'COMPLETE.json'),
     'ID_classification_gate':bool(a['best_val_acc']>=.80),
     'grad_norm_match_gate':bool(abs(meanratio-.50)<=.025 and saturation<=.10)}
   allrows.append(row)
 classification=all(x['ID_classification_gate'] for x in allrows)
 gradmatch=all(x['grad_norm_match_gate'] for x in allrows)
 status='ID_ONLY_FROZEN_PASS' if classification and gradmatch else ('ID_ONLY_FROZEN_NORM_MATCH_INCOMPLETE' if classification else 'ID_ONLY_FROZEN_ADVERSE')
 record={'stage':'P2C1 paired T1/T6 matched logit gradient with 3 independent seeds, ID-only freeze','status':status,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'n_models':6,'seeds':SEEDS,'arms':ARMS,'three_seed_complete':True,'no_OOD_read':True,'OOD_auto_unblind':False,
 'all_classification_pass':classification,'all_logit_gradient_match_pass':gradmatch,'records':allrows,'training_source_sha256':sha(P/'train_p2c1_normmatch_v2.py'),
 'amendment_sha256':sha(P/'FROZEN_AMENDMENT_01.json'),
 'inference_limit':'Logit confidence+entropy gradient norm only. Rank/backbone remain unnormalized; post hoc results require external confirmation.',
 'negative_results_preserved':True,'no_formal_retraining':True}
 with (P/'P2C1_ID_ONLY_FROZEN_GATE.json').open('x') as f:json.dump(record,f,indent=2)
 print('P2C1_ID_GATE_FROZEN',status,sha(P/'P2C1_ID_ONLY_FROZEN_GATE.json'),'MODELS',len(allrows),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  f=P/'P2C1_WATCHER_ERROR.json'
  with f.open('x') as out:json.dump({'error':repr(e),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_OOD_read':True},out,indent=2)
  raise
