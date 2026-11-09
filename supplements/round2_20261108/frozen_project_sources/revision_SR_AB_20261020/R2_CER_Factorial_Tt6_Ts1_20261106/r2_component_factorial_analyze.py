#!/usr/bin/env python3
"""Seed-level factorial effects; domains are averaged, never treated as seeds."""
import csv,hashlib,itertools,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import t
STUDY=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contrast(values,seed,axes):
 return sum(math.prod(1 if comp[a] else -1 for a in axes)*values[comp,seed] for comp in itertools.product([0,1],repeat=3))/(2**(3-len(axes)))
def summarize(x):
 n=len(x);avg=statistics.mean(x);sd=statistics.stdev(x);half=float(t.ppf(.975,n-1))*sd/math.sqrt(n)
 p=sum(abs(sum(a*b for a,b in zip(sign,x))/n)>=abs(avg)-1e-14 for sign in itertools.product([-1,1],repeat=n))/(2**n)
 return {'n_seeds':n,'mean':avg,'sample_sd':sd,'t95_low':avg-half,'t95_high':avg+half,'two_sided_exact_sign_flip_p':p,'seed_values':x}
def main():
 plan=json.loads((STUDY/'FROZEN_FACTORIAL_PROTOCOL.json').read_text());rows=[];metrics=[];hashes={}
 for p in sorted((STUDY/'OOD_evaluations').glob('*/metrics.json')):
  done=json.loads((p.parent/'COMPLETE.json').read_text());assert sha(p)==done['metrics_sha256'];assert sha(p.parent/'score_arrays.npz')==done['raw_sha256']
  m=json.loads(p.read_text());metrics.append(m);hashes[str(p)]=sha(p)
 assert len(metrics)==40 and len({(tuple(m['components']),m['seed']) for m in metrics})==40
 names=['AUROC','AUPR','FPR95_complete_ties'];domains=['ssb_hard','ninco','inaturalist','texture','openimage_o']
 # Validate metric field spelling before producing a potentially misleading table.
 for m in metrics:
  for score,dm in m['ood_metrics'].items():
   for domain in domains:
    for key in names:assert key in dm[domain],(key,list(dm[domain]))
    rows.append({'label':m['label'],'seed':m['seed'],'C':m['components'][0],'E':m['components'][1],'R':m['components'][2],'source':m['source'],'score':score,'domain':domain,**{k:dm[domain][k] for k in names}})
 with (STUDY/'ALL_DOMAIN_METRICS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 cell={};effects={};macro={}
 for score in metrics[0]['ood_metrics']:
  for key in names:
   values={(tuple(m['components']),m['seed']):statistics.mean(m['ood_metrics'][score][d][key] for d in domains) for m in metrics}
   macro[(score,key)]=values
   for comps in itertools.product([0,1],repeat=3):cell[f'{score}/{key}/{comps}']=summarize([values[comps,s] for s in plan['seeds']])
   for count in [1,2,3]:
    for axes in itertools.combinations(range(3),count):
     estimates=[]
     for seed in plan['seeds']:
      # Main effect = 4-cell on-minus-off average; 2-way = average 2 DIDs; 3-way = signed 8-cell sum.
      estimates.append(contrast(values,seed,axes))
     effects[f'{score}/{key}/'+''.join('CER'[a] for a in axes)]=summarize(estimates)
 output={'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','protocol_sha256':sha(STUDY/'FROZEN_FACTORIAL_PROTOCOL.json'),'input_metric_sha256':hashes,'cell_summaries':cell,'factorial_effects':effects,'primary_endpoint':'energy/AUROC equally weighted mean over five OOD domains; seeds42-46','effect_definitions':'on-minus-off main effects; averaged difference-in-differences for two-way interactions; signed eight-cell sum for three-way interaction','limitations':['Previously exposed OOD domains and post-review design','12 historical anchors and28 new trainings; hardware timing block is not randomized','Historical initial RNG state digests unavailable','Five seeds; exact two-sided sign-flip minimum p=0.0625','Many secondary scores, endpoints and effects; no confirmatory multiplicity-adjusted significance claim','Fixed teacherT6/studentT1 scope; temperature sensitivity is a separate older experiment','FPR95 is lower-is-better; AUROC and AUPR higher-is-better']}
 (STUDY/'FACTORIAL_SCIENTIFIC_RESULT.json').write_text(json.dumps(output,indent=2)+'\n');print(output['status'],flush=True)
if __name__=='__main__':main()
