#!/usr/bin/env python3
"""P1-5 seed101 RFKD versus CrossScore mechanism-matched independent audit.
Read-only. Precommitted before first OOD evaluation result. No GPU, model selection,
or training. All 4 RFKD results must be immutable, and 4 B2 source controls
must exist. Recompute every AUROC/AUPR/FPR95 from original per-image score arrays.
"""
from pathlib import Path
import json,hashlib,csv,datetime,math,itertools,os
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc
R=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
B=R/'B2_V1.2_student_20261020'
P=R/'P1_SR_ReviewerEvidence_20261022/P1_5_RFKD_Mechanism_Comparator_20261022'
O=P/'P1_5_SEED101_SCIENTIFIC_AUDIT_20261022'
if O.exists():raise FileExistsError('Scientific audit output exists; no overwrite')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 return h.hexdigest()
r=json.loads((P/'P1_5_RELEASE.json').read_text())
q=json.loads((P/'P1_5_EVAL_RELEASE.json').read_text())
pre=json.loads((P/'P1_5_ID_ONLY_PRE_OOD_AUDIT_PASS.json').read_text())
assert r['N_initial_gate']==4 and len(pre['model_records'])==4
assert pre['status']=='PASS_4_TRAINED_MODELS_NO_OOD_REVEALED'
assert sha(P/'P1_5_RELEASE.json')==pre['P1_5_scientific_release_sha256']
assert sha(P/'P1_5_EVAL_RELEASE.json')==pre['P1_5_eval_release_sha256']
assert sha(P/'frozen_src/b2_evaluate.py')==q['P1_5_modified_evaluator_SHA256']
events=[json.loads(x) for x in (P/'dependent_audit_events.jsonl').read_text().splitlines() if x.strip()]
assert events[-1]['status']=='P1_5_FOUR_MODEL_OOD_COMPLETE'
assert len([z for z in events if z['status']=='P1_5_OOD_FINISHED'])==4
assert not any(z['status']=='P1_5_DEPENDENT_EVAL_HALTED' for z in events)
def calc(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 assert a.ndim==b.ndim==1 and len(a)==10000 and len(b)>0
 assert np.isfinite(a).all() and np.isfinite(b).all()
 y=np.r_[np.zeros(len(a),np.int8),np.ones(len(b),np.int8)]
 s=np.r_[a,b]
 fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
 prec,rec,_=precision_recall_curve(y,s)
 return {'AUROC':float(roc_auc_score(y,s)),
  'FPR95_complete_ties':float(fpr[np.searchsorted(tpr,.95,side='left')]),
  'AUPR':float(auc(rec,prec))}
methods=[('RFKD_EnergyOrder_T6_Matched','CrossScore_T6_Corrected'),
 ('RFKD_EnergyOrder_T1_PublishedSpec','CrossScore_LegacyAsCode')]
rows=[];mac=[];pair=[];modelhash={};maximum_err=0
for ds in r['dataset_values']:
 for rf_method,cs_method in methods:
  scoremacro={}
  for role,method,base in [('RFKD',rf_method,P),('CrossScore',cs_method,B)]:
   tag=f'{ds}__seed101__{method}'
   out=base/'eval_formal'/tag/'primary'
   assert out.is_dir(),str(out)
   res=json.loads((out/'results.json').read_text());done=json.loads((out/'DONE.json').read_text())
   modelpath=base/'formal_runs'/tag/'epoch200_final.pt'
   actual_hash=sha(modelpath)
   assert res['model_sha256']==done['model_sha256']==actual_hash
   assert res['checkpoint_selection']=='primary' and res['smoke'] is False
   assert len(res['per_image_files'])==18 and set(res['metrics'])==set(q['OOD_domains'])
   assert res['ood_counts']==q['N_OOD_expected']
   modelhash[tag]=actual_hash
   for domain,score in itertools.product(q['OOD_domains'],q['score_types']):
    filepath=out/res['per_image_files'][domain+'__'+score]
    with np.load(filepath,allow_pickle=False) as f:
     assert set(f.files)=={'id','ood'}
     ids=f['id'];oods=f['ood']
    assert len(ids)==10000 and len(oods)==q['N_OOD_expected'][domain]
    m=calc(ids,oods)
    for met,v in m.items():
     expected=res['metrics'][domain][score]['OOD_positive'][met]
     err=abs(v-expected);maximum_err=max(maximum_err,err)
     if err>3e-6:raise AssertionError((tag,domain,score,met,err))
    rows.append({'dataset':ds,'comparison':rf_method+'__VS__'+cs_method,
     'role':role,'method':method,'domain':domain,'score':score,
     **m,'checkpoint_sha256':actual_hash,
     'score_npz_SHA256':sha(filepath),
     'run_type':'one historical paired-seed comparison only'})
   for score in q['score_types']:
    chunk=[row for row in rows if row['dataset']==ds and row['method']==method and row['score']==score]
    assert len(chunk)==6
    z={met:float(np.mean([row[met] for row in chunk]))
      for met in ('AUROC','FPR95_complete_ties','AUPR')}
    for met,v in z.items():
     expected=res['macro'][score]['OOD_positive'][met]
     err=abs(v-expected);maximum_err=max(maximum_err,err)
     if err>3e-6:raise AssertionError((tag,score,met,err))
    mac.append({'dataset':ds,'method':method,'score':score,**z})
  for score in q['score_types']:
   ce=next(a for a in mac if a['dataset']==ds and a['method']==cs_method and a['score']==score)
   rf=next(a for a in mac if a['dataset']==ds and a['method']==rf_method and a['score']==score)
   rec={'dataset':ds,'score':score,'RFKD_method':rf_method,'CrossScore_method':cs_method}
   for met in ('AUROC','FPR95_complete_ties','AUPR'):
    rec[met+'_RFKD']=rf[met]
    rec[met+'_CrossScore']=ce[met]
    rec[met+'_CrossScore_minus_RFKD']=ce[met]-rf[met]
   pair.append(rec)
print('P1_5_INDEPENDENT_AUDIT',json.dumps({'n_recomputed':len(rows)*3,
 'max_abs_error':maximum_err,
 'energy_pairs':[r for r in pair if r['score']=='energy_neg']},ensure_ascii=False),flush=True)
assert len(rows)==2*2*2*6*3 and len(mac)==2*2*2*3 and len(pair)==2*2*3
O.mkdir(exist_ok=False)
def save(name,arr):
 with (O/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(arr[0]));w.writeheader();w.writerows(arr)
save('P1_5_eight_models_all_domains_scores.csv',rows)
save('P1_5_eight_models_macros.csv',mac)
save('P1_5_seed101_CrossScore_vs_RFKD_pairs.csv',pair)
report={
 'name':'P1-5 first 4 RFKD models versus paired B2 CrossScore: independent OOD audit',
 'status':'P1_5_4_RFKD_MODELS_AND_4_B2_PAIRS_AUDIT_PASS',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'n_RF_KD_models':4,'n_B2_matched_controls':4,
 'n_model_domain_score_records':len(rows),
 'n_independently_recalculated_metrics':len(rows)*3,
 'max_error':maximum_err,'sha256_checkpoints':modelhash,
 'science_limits':[
  'One training seed per dataset x teacher target x temperature. No hypothesis significance claim.',
  'Official published RFKD checkpoint was unavailable; mechanism-matched reimplementation not authenticated official code.',
  'When comparing T1 and T6, only teacher supervision temperature changes; RFKD teacher Energy ordering unchanged.',
  'CIFAR10/100 six OOD domains are the same frozen benchmark, and not independent seed replicates.',
  'CrossScore manuscript differs in some historic original experimental settings; this is a new controlled matched protocol.',
  'No test/OOD-based model selection, posthoc hyperparameter optimization or excluded negative domain.',
  'Any decision to extend seeds 202/303/404/505 is separate, based on scientific reviewers needs, not win rate.'],
 'frozen_release_sha256':sha(P/'P1_5_RELEASE.json'),
 'frozen_evaluation_sha256':sha(P/'P1_5_EVAL_RELEASE.json'),
 'id_only_audit_sha256':sha(P/'P1_5_ID_ONLY_PRE_OOD_AUDIT_PASS.json'),
 'per_domain_csv_SHA256':sha(O/'P1_5_eight_models_all_domains_scores.csv')}
with (O/'P1_5_INDEPENDENT_AUDIT_PASS.json').open('x') as f:
 json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
for file in O.iterdir():file.chmod(0o444)
print('P1_5_AUDIT_PASS',sha(O/'P1_5_INDEPENDENT_AUDIT_PASS.json'),str(O),flush=True)
