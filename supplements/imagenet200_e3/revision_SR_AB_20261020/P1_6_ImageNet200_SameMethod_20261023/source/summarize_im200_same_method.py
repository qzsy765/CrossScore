#!/usr/bin/env python3
import csv,itertools,json,math
from pathlib import Path
import os
import numpy as np
from scipy.stats import t as student_t, ttest_1samp
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))
EXP=ROOT/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
SEEDS=[42,43,44,45,46]
MODES=['CE','CrossScore_T6_Corrected']
def flatten(d,prefix=''):
 out={}
 for k,v in d.items():
  key=f'{prefix}.{k}' if prefix else k
  if isinstance(v,dict): out.update(flatten(v,key))
  elif isinstance(v,(int,float)): out[key]=float(v)
 return out
def main():
 records={}
 for mode in MODES:
  for seed in SEEDS:
   run=f'{mode}_seed{seed}'; p=EXP/'evaluations'/run/'metrics.json'
   if not p.is_file(): raise FileNotFoundError(p)
   records[(mode,seed)]=flatten(json.loads(p.read_text()))
 allkeys=set.intersection(*(set(v) for v in records.values()))
 keys=sorted(k for k in allkeys if k=='id_test_top1' or k.startswith(('near_macro.','far_macro.','all_5_ood_macro.','ood_metrics.')))
 rows=[]; paired=[]
 for key in keys:
  for mode in MODES:
   vals=np.array([records[(mode,s)][key] for s in SEEDS],dtype=float)
   rows.append({'metric':key,'mode':mode,'n':len(vals),'mean':float(vals.mean()),'sd':float(vals.std(ddof=1))})
  ce=np.array([records[('CE',s)][key] for s in SEEDS],dtype=float)
  cs=np.array([records[('CrossScore_T6_Corrected',s)][key] for s in SEEDS],dtype=float)
  diff=cs-ce; mean=float(diff.mean()); sd=float(diff.std(ddof=1)); se=sd/math.sqrt(len(diff))
  crit=float(student_t.ppf(.975,len(diff)-1)); ci=(mean-crit*se,mean+crit*se)
  tt=ttest_1samp(diff,0.0); obs=abs(mean); signed=[]
  for signs in itertools.product([-1.,1.],repeat=len(diff)): signed.append(abs(float(np.mean(diff*np.asarray(signs)))))
  signp=float(sum(v>=obs-1e-12 for v in signed)/len(signed))
  paired.append({'metric':key,'n_paired':len(diff),'mean_CrossScore_minus_CE':mean,'sd_paired_difference':sd,
   'ci95_t_low':ci[0],'ci95_t_high':ci[1],'paired_t_statistic':float(tt.statistic),'paired_t_p_two_sided':float(tt.pvalue),
   'exact_sign_flip_p_two_sided':signp,'positive_seed_differences':int((diff>0).sum()),
   'seed_differences':';'.join(f'{s}:{d:+.6f}' for s,d in zip(SEEDS,diff))})
 with (EXP/'summary_mean_sd.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 with (EXP/'paired_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=paired[0].keys());w.writeheader();w.writerows(paired)
 summary={'protocol':'CrossScore-SR-AB-P1-6-ImageNet200-same-objective-scratch-v1','seeds':SEEDS,'n_seeds':len(SEEDS),
  'primary_metric':'all_5_ood_macro.AUROC','paired_comparison':'CrossScore_T6_Corrected minus CE','methods':MODES,
  'interpretation':'Report mean��sample SD, paired t-test/CI, and exact two-sided sign-flip p. With n=5, exact sign-flip p has minimum 0.0625; do not claim p<0.05 from that test.',
  'mean_sd_file':str(EXP/'summary_mean_sd.csv'),'paired_file':str(EXP/'paired_comparison.csv')}
 (EXP/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
