#!/usr/bin/env python3
"""P1-7E post-gate locked held-out-seed replication assessment.
No retraining or tuning; no result discarded on sign, magnitude or significance.
"""
from pathlib import Path
import os
import csv,json,datetime,hashlib,itertools
import numpy as np
P=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
ST=P/'P1_7E_heldout_seed_replication_20261026'
OUT=ST/'OOD_evaluations'
gate=json.loads((ST/'ID_ONLY_FROZEN_GATE.json').read_text())
assert gate['status']=='ID_ONLY_FROZEN_PASS' and gate['seeds']==[44,45,46]
seeds=[44,45,46]
domains=['ssb_hard','ninco','inaturalist','texture','openimage_o']
scores=['energy','neg_msp','neg_maxlogit','entropy','aux_uncert','centered_energy']
metrics=['AUROC','AUPR','FPR95_complete_ties']
assert (OUT/'STAGED_OOD_EXECUTION_PROTOCOL.json').is_file()
new={}
prior={}
ce={}
for seed in seeds:
 n=OUT/f'T6_E1_seed{seed}'
 assert (n/'COMPLETE.json').is_file(),n
 new[seed]=json.loads((n/'metrics.json').read_text())['ood_metrics']
 reference=P/'P1_7B_score_family_20261025'/f'CrossScore_T6_Corrected_seed{seed}'
 assert (reference/'COMPLETE.json').is_file()
 prior[seed]=json.loads((reference/'metrics.json').read_text())['ood_metrics']
 frozen=P/'evaluations'/f'CE_seed{seed}'
 assert (frozen/'COMPLETE.json').is_file()
 ce[seed]=json.loads((frozen/'metrics.json').read_text())['ood_metrics']
def macro(data,score,metric,doms):
 return np.array([np.mean([float(data[seed][score][d][metric]) for d in doms]) for seed in seeds],dtype=float)
def ce_macro(metric,doms):
 return np.array([np.mean([float(ce[seed][d][metric]) for d in doms]) for seed in seeds],dtype=float)
raw=[]
for score in scores:
 for domain in domains+['near_macro','far_macro','five_macro']:
  names={'near_macro':domains[:2],'far_macro':domains[2:],'five_macro':domains}.get(domain,[domain])
  for metric in metrics:
   x=macro(new,score,metric,names); y=macro(prior,score,metric,names)
   if not np.isfinite(x).all() or not np.isfinite(y).all():raise AssertionError('NaN metrics')
   for name,values in [('T6_E1_heldout',x),('T1_E1_frozen',y)]:
    raw.append({'score':score,'metric':metric,'domain':domain,'arm':name,
      'seed44':float(values[0]),'seed45':float(values[1]),'seed46':float(values[2]),
      'mean':float(values.mean()),'sd':float(values.std(ddof=1))})
   if score=='energy':
    z=ce_macro(metric,names)
    raw.append({'score':score,'metric':metric,'domain':domain,'arm':'CE_frozen',
      'seed44':float(z[0]),'seed45':float(z[1]),'seed46':float(z[2]),
      'mean':float(z.mean()),'sd':float(z.std(ddof=1))})
deltas=[]
for score in scores:
 for domain in domains+['near_macro','far_macro','five_macro']:
  names={'near_macro':domains[:2],'far_macro':domains[2:],'five_macro':domains}.get(domain,[domain])
  for metric in metrics:
   x=macro(new,score,metric,names)
   pairs=[('T1_E1_frozen',macro(prior,score,metric,names))]
   if score=='energy':pairs.append(('CE_frozen',ce_macro(metric,names)))
   for name,control in pairs:
    diff=x-control
    p=np.mean([abs(np.mean(diff*np.array(sig)))>=abs(np.mean(diff))-1e-12 for sig in itertools.product([-1,1],repeat=len(seeds))])
    deltas.append({'score':score,'metric':metric,'domain':domain,'comparison':'T6_E1 minus '+name,
       'delta_seed44':float(diff[0]),'delta_seed45':float(diff[1]),'delta_seed46':float(diff[2]),
       'delta_mean':float(diff.mean()),'delta_sd':float(diff.std(ddof=1)),
       'consistent_direction':bool(all(diff>=0) or all(diff<=0)),
       'exact_sign_flip_two_sided_p':float(p)})
for name,records in [('ALL_METRICS_BY_SEED.csv',raw),('PAIRED_DELTAS_BY_SEED.csv',deltas)]:
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
focus={}
for arm in ['T6_E1_heldout','T1_E1_frozen','CE_frozen']:
 focus[arm]={}
 for metric in metrics:
  found=next(r for r in raw if r['arm']==arm and r['score']=='energy' and r['domain']=='five_macro' and r['metric']==metric)
  focus[arm][metric]={'mean':found['mean'],'seed_values':[found['seed44'],found['seed45'],found['seed46']]}
def primary_diff(control):
 return next(r for r in deltas if r['score']=='energy' and r['metric']=='AUROC' and r['domain']=='five_macro' and r['comparison']=='T6_E1 minus '+control)
def iacc(seed):
 j=json.loads((ST/f'T6_E1_seed{seed}'/'ID_ONLY_DIAGNOSTICS.json').read_text())
 return j['id_test_top1']
id_acc=[iacc(s) for s in seeds]
flag={
 'ID_gate_pass_all':gate['status']=='ID_ONLY_FROZEN_PASS',
 'min_ID_top1':float(np.min(id_acc)),
 'repair_gain_over_T1_E1_>=_0.15':bool(primary_diff('T1_E1_frozen')['delta_mean']>=0.15),
 'candidate_within_0.02_CE_energy_AUROC':bool(primary_diff('CE_frozen')['delta_mean']>=-0.02),
 'no_CE_superiority_significance_claim':True}
manifest={
 'study':'P1-7E fixed T6_E1 new seeds 44-46, paired controls',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'within-seen-ImageNet200 benchmark replication, not independent out-of-domain generalization proof',
 'n_seed_pairs':3,'seeds':seeds,
 'main_frozen_energy_five_macro':focus,
 'paired_primary_vs_T1_E1':primary_diff('T1_E1_frozen'),
 'paired_primary_vs_CE':primary_diff('CE_frozen'),
 'ID_test_accuracy':id_acc,
 'descriptive_gates':flag,
 'stats_warning':'Exact paired randomization p min .25 with 3 pairs, cannot support p<.05; multiple correlated OOD datasets are not new independent seeds.',
 'no_selection':'All new seed runs included regardless of outcome; no OOD-tuned threshold; CE and T1_E1 controls not retrained.',
 'source_manifest_sha256':hashlib.sha256((ST/'FROZEN_REPLICATION_PROTOCOL.json').read_bytes()).hexdigest(),
 'ID_frozen_gate_sha256':hashlib.sha256((ST/'ID_ONLY_FROZEN_GATE.json').read_bytes()).hexdigest()
}
with (OUT/'P1_7E_FINAL_SCIENTIFIC_RESULT.json').open('x') as f:
 f.write(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('P1_7E_FROZEN_SCIENCE_DONE',json.dumps(focus,indent=2),flush=True)
print('PRIMARY_DELTAS',json.dumps({'vs_original':primary_diff('T1_E1_frozen'),'vs_CE':primary_diff('CE_frozen')},indent=2))

