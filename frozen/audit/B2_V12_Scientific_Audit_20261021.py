#!/usr/bin/env python3
"""B2 V1.2 post-hoc read-only integrity & descriptive/paired analysis.
No model re-training, hyperparameter changes, checkpoint selection or result editing.
Creates an immutable separately staged scientific-audit report.
"""
import os,json,csv,math,hashlib,itertools,datetime,statistics,sys
from pathlib import Path
import numpy as np
ROOT=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
S=ROOT/'B2_V1.2_student_20261020'
E=S/'B2_FINAL_OOD_EVAL_20261021'
ST=E/'scientific_audit_20261021'
if ST.exists():raise FileExistsError("Refuse scientific audit overwrite")
manifest=json.loads((S/'task_manifest_40.json').read_text())
freeze=json.loads((S/'FREEZE_HASHES.json').read_text())
audit=json.loads((E/'PRE_EVAL_AUDIT_PASS.json').read_text())
def hashfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
assert hashfile(S/'task_manifest_40.json')==freeze['manifest_sha256']
assert audit['models_checked']==40 and len(audit['model_file_sha256s'])==40
assert len(manifest['tasks'])==40
OOD=['svhn','textures','lsun_crop','lsun_resize','places365','isun']
expected={'svhn':26032,'textures':5640,'lsun_crop':10000,'lsun_resize':10000,'places365':36500,'isun':8925}
KINDS=('energy_neg','msp_neg','entropy')
methods=['CE','VanillaKD_T4','CrossScore_LegacyAsCode','CrossScore_T6_Corrected']
seeds=[101,202,303,404,505]
datasets=['cifar100','cifar10']
metrics=('AUROC','FPR95_complete_ties','AUPR')
raw=[];wide={};all_sha={};problem=[];perimage_sha={}
pretrain={}
for job in manifest['tasks']:
 tag=job['task_id']
 path=S/'eval_formal'/tag/'primary'
 d=json.loads((path/'DONE.json').read_text())
 r=json.loads((path/'results.json').read_text())
 model_path=Path(job['run_directory'])/'epoch200_final.pt'
 assert model_path.is_file() and hashfile(model_path)==audit['model_file_sha256s'][tag],tag+' invalid model'
 assert d['model_sha256']==r['model_sha256']==audit['model_file_sha256s'][tag],tag+' checkpoint changed'
 assert d['eval_sha256']==freeze['code_sha256']['b2_evaluate.py'] and d['protocol']==manifest['protocol_id']
 assert d['tag']==tag and d['six_ood_present'] and r['checkpoint_selection']=='primary' and r['smoke']==False
 assert r['protocol']==manifest['protocol_id'] and r['dataset']==job['dataset'] and r['method']==job['method'] and r['seed']==job['seed']
 assert r['score_sign']=='higher score = more OOD'
 assert set(r['metrics'])==set(OOD) and set(r['ood_counts'])==set(OOD)
 assert set(r['per_image_files'])=={x+'__'+kind for x in OOD for kind in KINDS},tag+' score manifest incomplete'
 assert r['ood_counts']==expected,tag+' sample count mismatch'
 assert len(r['per_image_files'])==18
 train=json.loads((Path(job['run_directory'])/'DONE.json').read_text())
 pretrain[tag]=train['best_val_accuracy']
 all_sha[tag]=hashfile(path/'results.json')
 # Independence: metrics macro must equal exactly mean of six domain metrics, for 3 scores and both positive types.
 for kind in KINDS:
  for pol in ('OOD_positive','ID_positive'):
   for metric in metrics:
    avg=float(np.mean([r['metrics'][od][kind][pol][metric] for od in OOD]))
    assert math.isclose(r['macro'][kind][pol][metric],avg,rel_tol=0,abs_tol=1e-11),tag+' mean inconsistency'
 for od in OOD:
  for kind in KINDS:
   npz=path/r['per_image_files'][od+'__'+kind]
   with np.load(npz,allow_pickle=False) as n:
    a=n['id'];b=n['ood']
   assert a.ndim==b.ndim==1 and len(a)==10000 and len(b)==expected[od],tag+' score count '+od+' '+kind
   assert np.isfinite(a).all() and np.isfinite(b).all(),tag+' nonfinite scores'
   if kind=='energy_neg':
    # full score-based paired audit of each main metric, independent from saved aggregated metrics.
    from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,roc_curve
    yy=np.r_[np.zeros(len(a),dtype=np.int8),np.ones(len(b),dtype=np.int8)]
    xx=np.r_[a,b]
    checkauroc=roc_auc_score(yy,xx)
    assert abs(checkauroc-r['metrics'][od]['energy_neg']['OOD_positive']['AUROC'])<1e-10,tag+' raw energy auroc differs'
    # FPR95 using complete-tie threshold grouping and >=.95 TPR; scikit-learn roc_curve.
    fpr,tpr,_=roc_curve(yy,xx,drop_intermediate=False)
    ix=np.flatnonzero(tpr>=.95)
    checkfpr=float(fpr[ix[0]]) if len(ix) else 1.
    assert abs(checkfpr-r['metrics'][od]['energy_neg']['OOD_positive']['FPR95_complete_ties'])<1e-10,tag+' raw energy FPR95 differs'
    perimage_sha[tag+'__'+od+'__'+kind]=hashfile(npz)
  for kind in KINDS:
   for pol in ('OOD_positive','ID_positive'):
    for metric in metrics:
     val=r['metrics'][od][kind][pol][metric]
     assert math.isfinite(val) and -1e-10<=val<=1.0000000001
     raw.append({'dataset':job['dataset'],'seed':job['seed'],'method':job['method'],
       'domain':od,'score':kind,'positive':pol,'metric':metric,'value':val,'tag':tag})
 wide[(job['dataset'],job['seed'],job['method'])]=r
 print('AUDITED_MODEL',len(wide),tag,'macro_auroc',round(r['macro']['energy_neg']['OOD_positive']['AUROC'],6),flush=True)
assert len(wide)==40 and len(all_sha)==40
for port in ('10153','10302'):
 f=E/'events'/(port+'.jsonl')
 entries=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
 assert entries[-1]['status']=='EVAL_HOST_FINISHED' and not any(e['status']=='EVAL_HOST_HALTED' for e in entries)
 assert sum(x['status']=='EVAL_DONE' for x in entries)==20
 for x in entries:
  if x['status']=='EVAL_DONE':
   assert x['detail']['result_sha256']==all_sha[x['task']],x['task']+' event hash mismatch'

def avg_sd(a):
 a=np.array(a,dtype=float)
 return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'n':len(a),'values':[float(x) for x in a]}
# exact paired sign randomization distribution, 2^5 possibilities; do not assume normal.
def signflip_p(diff):
 diff=np.array(diff,dtype=float);obs=abs(diff.mean())
 null=np.array([abs(np.mean(diff*np.array(sign))) for sign in itertools.product((-1,1),repeat=len(diff))])
 return float((null>=obs-1e-12).mean())
TC_95_N5=2.7764451051977987
summary=[];paired=[];domains=[];seedrows=[];scores=[]
for ds in datasets:
 for meth in methods:
  for score in KINDS:
   for metric in metrics:
    vals=[wide[(ds,s,meth)]['macro'][score]['OOD_positive'][metric] for s in seeds]
    v=avg_sd(vals)
    summary.append({'dataset':ds,'method':meth,'score':score,'metric':metric,
      'mean':v['mean'],'sd':v['sd'],'n':5})
  for od in OOD:
   for metric in metrics:
    vals=[wide[(ds,s,meth)]['metrics'][od]['energy_neg']['OOD_positive'][metric] for s in seeds]
    vals=avg_sd(vals)
    domains.append({'dataset':ds,'method':meth,'domain':od,'metric':metric,'mean':vals['mean'],'sd':vals['sd'],'n':5})
  for sd in seeds:
   data=wide[(ds,sd,meth)]['macro']['energy_neg']['OOD_positive']
   seedrows.append({'dataset':ds,'seed':sd,'method':meth,**data,
    'last_train_val_best_acc':pretrain[f'{ds}__seed{sd}__{meth}']})
 for meth in methods[1:]:
  for ref in ['CE','VanillaKD_T4']:
   if meth==ref:continue
   for score in ['energy_neg','msp_neg','entropy']:
    for metric in metrics:
     diff=np.array([wide[(ds,sd,meth)]['macro'][score]['OOD_positive'][metric]-
       wide[(ds,sd,ref)]['macro'][score]['OOD_positive'][metric] for sd in seeds])
     delta=float(diff.mean());sdelta=float(diff.std(ddof=1))
     paired.append({'dataset':ds,'method':meth,'reference':ref,'score':score,'metric':metric,
       'mean_delta':delta,'sd_delta':sdelta,'ci95_t_low':delta-TC_95_N5*sdelta/np.sqrt(5),
       'ci95_t_high':delta+TC_95_N5*sdelta/np.sqrt(5),
       'exact_signflip_p_two_sided':signflip_p(diff),
       'per_seed_delta':[round(float(x),9) for x in diff],
       'n_positive':int((diff>0).sum()),'n_negative':int((diff<0).sum()),'n_zero':int((diff==0).sum())})
# print main concise table
print('===PRIMARY_TABLE===',flush=True)
for ds in datasets:
 for meth in methods:
  z={x['metric']:x for x in summary if x['dataset']==ds and x['method']==meth and x['score']=='energy_neg'}
  print(json.dumps({'dataset':ds,'method':meth,
    'AUROC':z['AUROC']['mean'],'AUROC_sd':z['AUROC']['sd'],
    'FPR95':z['FPR95_complete_ties']['mean'],'FPR95_sd':z['FPR95_complete_ties']['sd'],
    'AUPR':z['AUPR']['mean'],'AUPR_sd':z['AUPR']['sd']},ensure_ascii=False),flush=True)
print('===PRIMARY_PAIRS===',flush=True)
for ds in datasets:
 for meth in methods[1:]:
  for ref in ['CE','VanillaKD_T4']:
   if meth==ref:continue
   for metric in metrics:
    q=next(x for x in paired if x['dataset']==ds and x['method']==meth and x['reference']==ref and x['score']=='energy_neg' and x['metric']==metric)
    print(json.dumps(q,ensure_ascii=False),flush=True)
print('===DOMAIN_AUROC===',flush=True)
for ds in datasets:
 for od in OOD:
  row={'dataset':ds,'domain':od}
  for meth in methods:
   row[meth]=round(next(x['mean'] for x in domains if x['dataset']==ds and x['domain']==od and x['method']==meth and x['metric']=='AUROC'),6)
  print(json.dumps(row,ensure_ascii=False),flush=True)
print('===SEED_AUROC===',flush=True)
for ds in datasets:
 for sd in seeds:
  row={'dataset':ds,'seed':sd}
  for meth in methods:
   row[meth]=round(wide[(ds,sd,meth)]['macro']['energy_neg']['OOD_positive']['AUROC'],6)
  print(json.dumps(row,ensure_ascii=False),flush=True)

# Write immutable outputs create-only after all validation succeeded.
ST.mkdir(exist_ok=False)
def emitcsv(filename,rows):
 p=ST/filename
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader()
  for row in rows:
   rr={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()}
   w.writerow(rr)
 return hashfile(p)
outfiles={}
for name,rs in [('all_domain_metrics_4320.csv',raw),('method_macro_statistics.csv',summary),
  ('paired_comparisons.csv',paired),('domain_statistics.csv',domains),('seed_macro_energy.csv',seedrows)]:
 outfiles[name]=emitcsv(name,rs)
validation={'protocol':manifest['protocol_id'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'SCIENTIFIC_AUDIT_PASS','n_models':40,'n_ood_per_model':6,
 'n_domain_evaluations':240,'n_per_image_score_files':720,
 'n_macro_primary':8,'checkpoint_selection':'epoch200_final.pt, not ID-val-best',
 'score_types':KINDS,'ood_domains':OOD,'metric_types':metrics,
 'main_score':'energy_neg','main_metric':'six-domain equally weighted AUROC, OOD-positive',
 'all_result_json_sha256':all_sha,'primary_score_npz_sha256':perimage_sha,
 'csv_sha256':outfiles,'teacher_hash_provenance':freeze['teacher_sha256'],
 'manifest_sha256':freeze['manifest_sha256'],'evaluation_source_sha256':freeze['code_sha256']['b2_evaluate.py'],
 'statistical_caution':'n=5 paired model seeds; exact two-sided signflip min attainable 0.0625; t CI relies on approximate normality; six OOD domains not independent repeated seeds',
 'note':'All loss methods frozen; negative results preserved; six-domain macro is unweighted across OOD domains and independent of domain cardinality'}
with (ST/'SCIENTIFIC_AUDIT_PASS.json').open('x') as f:
 json.dump(validation,f,indent=2,ensure_ascii=False);f.write('\n')
for file in ST.iterdir():
 if file.is_file():file.chmod(0o444)
print('FINAL_AUDIT_PASS',json.dumps({'path':str(ST),'completed':40,'domain_metrics':len(raw),
 'tables':outfiles,'complete_at_utc':validation['created_utc']},ensure_ascii=False),flush=True)
