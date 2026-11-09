#!/usr/bin/env python3
"""Independent CPU recomputation with sklearn; never import the evaluated metric code."""
import csv,datetime,hashlib,itertools,json,math,platform,statistics,sys
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import t
import sklearn
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,auc
ROOT=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
STUDY=ROOT/'R2_Recovery_Audit_and_P2C1_OOD_20261106'
IM=ROOT/'P1_6_ImageNet200_SameMethod_20261023'
OUT=STUDY/'INDEPENDENT_CPU_AUDIT_20261106'
DOMAINS=['ssb_hard','ninco','inaturalist','texture','openimage_o']
SCORES=['energy','neg_msp','neg_maxlogit','entropy','aux_uncert','centered_energy']
N={'test':9000,'ssb_hard':49000,'ninco':5879,'inaturalist':10000,'texture':5160,'openimage_o':15869}
METRICS=['AUROC','AUPR','FPR95_complete_ties'];SEEDS=[44,45,46]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def metric(a,b):
 y=np.r_[np.zeros(len(a),dtype=np.int8),np.ones(len(b),dtype=np.int8)];s=np.r_[a,b].astype(np.float64)
 pr,re,_=precision_recall_curve(y,s);fp,tp,_=roc_curve(y,s,drop_intermediate=False)
 return {'AUROC':float(roc_auc_score(y,s)),'AUPR':float(auc(re,pr)),'FPR95_complete_ties':float(fp[np.flatnonzero(tp>=.95)[0]])}
def summary(x):
 m=statistics.mean(x);sd=statistics.stdev(x);half=float(t.ppf(.975,len(x)-1))*sd/math.sqrt(len(x))
 return {'mean':m,'sample_sd':sd,'per_seed':x,'t95_low':m-half,'t95_high':m+half}
def paired(x,y):
 d=[a-b for a,b in zip(x,y)];out=summary(d);out['two_sided_exact_sign_flip_p']=sum(abs(statistics.mean(a*b for a,b in zip(d,sign)))>=abs(out['mean'])-1e-14 for sign in itertools.product([-1,1],repeat=len(d)))/(2**len(d));return out
def main():
 OUT.mkdir(exist_ok=False);assert (STUDY/'ALL_SIX_OOD_COMPLETE.json').exists()
 plan=json.loads((STUDY/'FROZEN_OOD_PROTOCOL.json').read_text());gatepath=IM/'P2_C1_gradient_norm_matched_IDonly_20261027/P2C1_ID_ONLY_FROZEN_GATE.json';assert sha(gatepath)==plan['id_gate_sha256'];gate=json.loads(gatepath.read_text())
 items=[]
 for seed in SEEDS:
  for arm in ['T1_E1_NORMMATCH','T6_E1_NORMMATCH']:
   name=arm+'_seed'+str(seed);items.append((arm,seed,STUDY/'OOD_evaluations'/name,IM/'P2_C1_gradient_norm_matched_IDonly_20261027'/name/'best.pt',True))
  for arm,name in [('CE_FROZEN',f'CE_seed{seed}'),('T1_E1_FROZEN',f'CrossScore_T6_Corrected_seed{seed}')]:items.append((arm,seed,IM/'P1_7B_score_family_20261025'/name,IM/'runs'/name/'best.pt',False))
  name=f'T6_E1_seed{seed}';checkpoints=list((IM/'P1_7E_heldout_seed_replication_20261026').rglob(name+'/best.pt'));assert len(checkpoints)==1,(name,checkpoints)
  items.append(('T6_E1_FROZEN',seed,IM/'P1_7E_heldout_seed_replication_20261026/OOD_evaluations'/name,checkpoints[0],False))
 rows=[];checks=[];inputs=[];macros={};distributions={};new_raw=[];maxdiff=0.
 for arm,seed,p,checkpoint,isnew in items:
  meta=json.loads((p/'metrics.json').read_text());cp_sha=sha(checkpoint);assert cp_sha==meta.get('checkpoint_sha256',meta.get('model_hash_sha256')),(arm,seed)
  if isnew:
   done=json.loads((p/'COMPLETE.json').read_text());assert sha(p/'metrics.json')==done['metrics_sha256'] and sha(p/'score_arrays.npz')==done['raw_scores_sha256']
   old=next(x for x in gate['records'] if x['seed']==seed and x['arm']==arm);assert cp_sha==old['checkpoint_sha256'];assert json.loads((p/'ID_REPLAY_PASS.json').read_text())['before_any_OOD']
  inputs.append({'arm':arm,'seed':seed,'checkpoint':str(checkpoint),'checkpoint_sha256':cp_sha,'metrics':str(p/'metrics.json'),'metrics_sha256':sha(p/'metrics.json'),'raw_scores':str(p/'score_arrays.npz'),'raw_scores_sha256':sha(p/'score_arrays.npz')})
  with np.load(p/'score_arrays.npz') as raw:
   for kind,n in N.items():
    for score in SCORES:
     v=raw[kind+'__'+score];assert v.ndim==1 and len(v)==n and np.isfinite(v).all(),(p,kind,score)
   for score in SCORES:
    for domain in DOMAINS:
     value=metric(raw['test__'+score],raw[domain+'__'+score]);expected=meta['ood_metrics'][score][domain]
     for key in METRICS:
      d=abs(value[key]-expected[key]);maxdiff=max(maxdiff,d);checks.append({'arm':arm,'seed':seed,'score':score,'domain':domain,'metric':key,'abs_difference':d,'pass':d<=1e-12});assert d<=1e-12,checks[-1]
     rows.append({'arm':arm,'seed':seed,'score':score,'domain':domain,**value})
    macros[arm,seed,score]={key:statistics.mean(r[key] for r in rows if r['arm']==arm and r['seed']==seed and r['score']==score) for key in METRICS}
   distributions[arm,seed]={score:{'ID_mean':float(raw['test__'+score].mean()),'ID_sd':float(raw['test__'+score].std())} for score in SCORES}
   if isnew:
    for key in raw.files:new_raw.append((arm+'_seed'+str(seed)+'___'+key,raw[key].copy()))
 focus={arm:{score:{key:summary([macros[arm,s,score][key] for s in SEEDS]) for key in METRICS} for score in SCORES} for arm in ['CE_FROZEN','T1_E1_FROZEN','T6_E1_FROZEN','T1_E1_NORMMATCH','T6_E1_NORMMATCH']}
 oldpath=IM/'P2_B_gradient_weight_intervention_20261026/OOD_evaluations/P2B_FINAL_SCIENTIFIC_RESULT.json';old=json.loads(oldpath.read_text())
 for arm in ['CE_FROZEN','T1_E1_FROZEN','T6_E1_FROZEN']:
  for key in METRICS:assert np.max(np.abs(np.asarray(focus[arm]['energy'][key]['per_seed'])-old['focus'][arm][key]['per_seed']))<1e-12,(arm,key)
 comparisons={}
 for a,b in [('T6_E1_NORMMATCH','T1_E1_NORMMATCH'),('T1_E1_NORMMATCH','CE_FROZEN'),('T6_E1_NORMMATCH','CE_FROZEN'),('T1_E1_NORMMATCH','T1_E1_FROZEN'),('T6_E1_NORMMATCH','T6_E1_FROZEN')]:comparisons[a+'_minus_'+b]={key:paired(focus[a]['energy'][key]['per_seed'],focus[b]['energy'][key]['per_seed']) for key in METRICS}
 result={'status':'PASS_INDEPENDENT_RAW_SCORE_RECALCULATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independent_metric_implementation':'sklearn roc_auc_score, precision_recall_curve+auc, roc_curve(drop_intermediate=False); no import of frozen b2_protocol','n_models':15,'n_new_models':6,'n_historical_controls':9,'n_score_domain_blocks':len(rows),'n_metric_checks':len(checks),'max_abs_metric_difference':maxdiff,'tolerance':1e-12,'sample_counts':N,'seeds':SEEDS,'metric_notes':'OOD positive; higher AUROC/AUPR better; FPR95 lower better; trapezoidal PR area, not average_precision_score','protocol_sha256':sha(STUDY/'FROZEN_OOD_PROTOCOL.json'),'id_gate_sha256':sha(gatepath),'audit_source_sha256':sha(Path(__file__)),'original_P2B_control_aggregate_sha256':sha(oldpath),'all_old_control_macro_replays_match':True,'focus':focus,'paired_energy_comparisons':comparisons,'ID_distributions':{a+'_seed'+str(s):v for (a,s),v in distributions.items()},'ID_gate_records':gate['records'],'inputs':inputs,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'limitations':['Only C+E class-logit gradient norm matched to0.5 ofCE; rank andbackbone gradients notnormalized','Magnitude matching does notmatch gradientdirection, individualC/E contributions orcomplete optimization trajectories','Same previously examined OOD domains; post-review exploratory evaluation','n3seeds; exacttwo-sidedp minimum.25; tintervals descriptive andassumption dependent; no multiplicity-adjusted significance claim','All6formal training records retain4-7AMP skippedsteps','This is nota C/E/R factorial; component attribution awaits that study']}
 (OUT/'P2C1_INDEPENDENT_SCIENTIFIC_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'ALL_1350_METRIC_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
 with (OUT/'ALL_DOMAIN_METRICS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'P2C1_ALL_SIX_RAW_SCORES.npz',**dict(new_raw))
 (OUT/'COMPLETE.json').write_text(json.dumps({'status':'PASS','scientific_result_sha256':sha(OUT/'P2C1_INDEPENDENT_SCIENTIFIC_RESULT.json'),'checks_sha256':sha(OUT/'ALL_1350_METRIC_CHECKS.json'),'domain_csv_sha256':sha(OUT/'ALL_DOMAIN_METRICS.csv'),'combined_new_raw_sha256':sha(OUT/'P2C1_ALL_SIX_RAW_SCORES.npz')},indent=2)+'\n')
 print(json.dumps({'status':result['status'],'n_metric_checks':len(checks),'max_abs_difference':maxdiff,'energy':{a:v['energy'] for a,v in focus.items()},'paired_energy_comparisons':comparisons}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  OUT.mkdir(exist_ok=True);(OUT/'AUDIT_ERROR.json').write_text(json.dumps({'error':repr(e)},indent=2)+'\n');raise
