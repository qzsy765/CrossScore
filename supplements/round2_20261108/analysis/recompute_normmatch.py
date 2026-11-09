from pathlib import Path
import argparse,json,csv,itertools,math
import numpy as np
from scipy.stats import t
p=argparse.ArgumentParser(description='Recompute norm-matched summaries from domain-level records')
p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();P=a.evidence_dir;O=a.output_dir;O.mkdir(exist_ok=True,parents=True)
R=json.loads((P/'P2C1_INDEPENDENT_SCIENTIFIC_RESULT.json').read_text())
rows=list(csv.DictReader((P/'ALL_DOMAIN_METRICS.csv').open()))
metrics=['AUROC','AUPR','FPR95_complete_ties'];seeds=R['seeds'];checks=[];cube={}
assert len(rows)==450 and len(seeds)==3
def summary(x):
 x=np.asarray(x,dtype=float);m=float(x.mean());sd=float(x.std(ddof=1));h=float(t.ppf(.975,2))*sd/math.sqrt(3)
 signs=np.asarray(list(itertools.product([-1,1],repeat=3)))
 pv=float(np.mean(np.abs((signs*x).mean(axis=1))>=abs(m)-1e-14))
 return dict(mean=m,sample_sd=sd,t95_low=m-h,t95_high=m+h,two_sided_exact_sign_flip_p=pv,per_seed=x.tolist())
out={}
for arm,scores in R['focus'].items():
 for score,expected in scores.items():
  for metric in metrics:
   x=[]
   for seed in seeds:
    sel=[r for r in rows if r['arm']==arm and r['score']==score and int(r['seed'])==seed]
    assert len(sel)==5
    val=float(np.mean([float(r[metric]) for r in sel]));x.append(val);cube[arm,score,metric,seed]=val
   s=summary(x);out[arm+'/'+score+'/'+metric]=s
   for k in ['mean','sample_sd','t95_low','t95_high']:
    d=abs(s[k]-expected[metric][k]);checks.append(d);assert d<(1e-10 if k.startswith('t95_') else 1e-12)
for name,values in R['paired_energy_comparisons'].items():
 left,right=name.split('_minus_')
 for metric in metrics:
  s=summary([cube[left,'energy',metric,seed]-cube[right,'energy',metric,seed] for seed in seeds]);out[name+'/'+metric]=s
  for k in ['mean','sample_sd','t95_low','t95_high','two_sided_exact_sign_flip_p']:
   if k in values[metric]:
    d=abs(s[k]-values[metric][k]);checks.append(d);assert d<(1e-10 if k.startswith('t95_') else 1e-12)
result=dict(status='PASS_ARCHIVED_DOMAIN_SUMMARY_RECOMPUTATION',n_domain_rows=len(rows),n_scalar_checks=len(checks),max_abs_difference=max(checks),results=out,scope='CPU summaries from domain records; raw per-image arrays are not bundled')
(O/'NORMMATCH_SUMMARY_RECOMPUTATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
