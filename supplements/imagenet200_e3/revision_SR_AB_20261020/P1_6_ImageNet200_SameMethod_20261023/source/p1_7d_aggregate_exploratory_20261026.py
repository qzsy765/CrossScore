#!/usr/bin/env python3
"""Aggregate exactly two paired frozen ID-gated runs per 2x2 arm.
Exploratory only. Primary endpoint: unweighted five-domain mean Energy AUROC.
"""
from pathlib import Path
import os
import csv,json,datetime,itertools,hashlib
import numpy as np
E=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
OUT=E/'P1_7D_factorial_OOD_exploratory_20261026'
DOMAINS=['ssb_hard','ninco','inaturalist','texture','openimage_o']
SCORES=['energy','neg_msp','neg_maxlogit','entropy','aux_uncert','centered_energy']
METRICS=['AUROC','AUPR','FPR95_complete_ties']
VARIANTS=['T1_E1','T1_E0','T6_E1','T6_E0']
SEEDS=[42,43]
assert (OUT/'FROZEN_P1_7D_PROTOCOL.json').is_file()
assert json.loads((E/'P1_7C_factorial_IDonly_20261026/ID_ONLY_FROZEN_RESULTS.json').read_text())['status']=='ID_ONLY_FROZEN_WITHOUT_OOD'
db={}
for v in VARIANTS:
 for seed in SEEDS:
  if v=='T1_E1':
   f=E/'P1_7B_score_family_20261025'/f'CrossScore_T6_Corrected_seed{seed}'/'metrics.json'
  else:
   folder=OUT/f'{v}_seed{seed}'
   assert (folder/'COMPLETE.json').is_file(),f'incomplete {folder}'
   f=folder/'metrics.json'
  j=json.loads(f.read_text())
  assert j['score_family']==SCORES
  for score in SCORES:
   assert set(j['ood_metrics'][score])==set(DOMAINS)
   for d in DOMAINS:
    for m in METRICS:
     val=float(j['ood_metrics'][score][d][m])
     assert np.isfinite(val) and 0<=val<=1,(v,seed,score,d,m,val)
  db[(v,seed)]=j
records=[]
summary=[]
for score in SCORES:
 for domain in DOMAINS+['near_macro','far_macro','five_macro']:
  ds={'near_macro':DOMAINS[:2],'far_macro':DOMAINS[2:],'five_macro':DOMAINS}.get(domain,[domain])
  for metric in METRICS:
   m={v:np.array([np.mean([float(db[(v,s)]['ood_metrics'][score][d][metric]) for d in ds]) for s in SEEDS]) for v in VARIANTS}
   for v in VARIANTS:
    vals=m[v]
    records.append({'score':score,'domain':domain,'metric':metric,'variant':v,'seed42':float(vals[0]),'seed43':float(vals[1]),'mean':float(vals.mean()),'sample_sd':float(vals.std(ddof=1))})
   for v in VARIANTS[1:]:
    delta=m[v]-m['T1_E1']
    # With n=2 two-sided paired randomization p value cannot support conventional significance.
    pexact=np.mean([abs(np.mean(delta*np.array(signs)))>=abs(delta.mean())-1e-12 for signs in itertools.product([-1,1],repeat=2)])
    summary.append({'score':score,'domain':domain,'metric':metric,'variant':v,'delta_mean_vs_T1_E1':float(delta.mean()),
    'delta_seed42':float(delta[0]),'delta_seed43':float(delta[1]),'both_seeds_same_sign':bool(np.sign(delta[0])==np.sign(delta[1])),
    'exact_sign_flip_two_sided_p':float(pexact)})
for name,rows in [('all_score_metrics_by_seed.csv',records),('paired_changes_vs_T1_E1.csv',summary)]:
 with (OUT/name).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
focus=[]
for v in VARIANTS:
 obj={'variant':v,'primary_energy_auroc_macro':next(x['mean'] for x in records if x['score']=='energy' and x['domain']=='five_macro' and x['metric']=='AUROC' and x['variant']==v)}
 for score in ['neg_msp','neg_maxlogit','entropy','aux_uncert']:
  obj[score+'_auroc_macro']=next(x['mean'] for x in records if x['score']==score and x['domain']=='five_macro' and x['metric']=='AUROC' and x['variant']==v)
 obj['energy_fpr95_macro']=next(x['mean'] for x in records if x['score']=='energy' and x['domain']=='five_macro' and x['metric']=='FPR95_complete_ties' and x['variant']==v)
 obj['energy_auroc_seed_values']=[next(x['seed42'] for x in records if x['score']=='energy' and x['domain']=='five_macro' and x['metric']=='AUROC' and x['variant']==v),
 next(x['seed43'] for x in records if x['score']=='energy' and x['domain']=='five_macro' and x['metric']=='AUROC' and x['variant']==v)]
 focus.append(obj)
out={'stage':'P1-7D post-ID-only-freeze exploratory factorial OOD result','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'primary_metric':'energy macro AUROC on 5 OOD domains','repeated_seeds':[42,43],'n_arms':4,'new_runs':6,
'disclosure':'two seeds only, post-hoc mechanism study after P1-6 failure, no claim of independent confirmation',
'reference_data':'P1-7B existing T1_E1 frozen corrected-T6 student-T1 baseline; not a new training run',
'focus':focus,'paired_delta_rows':len(summary),
'full_result_file':str(OUT/'all_score_metrics_by_seed.csv'),
'delta_file':str(OUT/'paired_changes_vs_T1_E1.csv'),
'original_P1_7D_protocol_sha256':hashlib.sha256((OUT/'FROZEN_P1_7D_PROTOCOL.json').read_bytes()).hexdigest(),
'original_ID_only_gate_sha256':hashlib.sha256((E/'P1_7C_factorial_IDonly_20261026/ID_ONLY_FROZEN_RESULTS.json').read_bytes()).hexdigest()}
with (OUT/'P1_7D_FINAL_EXPLORATORY_SUMMARY.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
print('P1_7D_ALL_6_AND_FOUR_ARMS_SUMMARIZED',json.dumps(focus,indent=2),flush=True)

