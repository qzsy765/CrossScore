#!/usr/bin/env python3
"""P1-1 fixed ODIN same-checkpoint four-model final integrity / metrics audit.
Create-only evidence folder. Do not modify submitted manuscript, source results,
checkpoints, or previously frozen release. No GPU/training required.
"""
from pathlib import Path
import hashlib,json,csv,datetime,math,statistics
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc
R=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
P=R/'P1_SR_ReviewerEvidence_20261022'
OUT=P/'P1_1_ODIN_SCIENTIFIC_AUDIT'
if OUT.exists():raise FileExistsError('P1 ODIN final audit output exists')
def sha(f):
 h=hashlib.sha256()
 with open(f,'rb') as fd:
  for chunk in iter(lambda:fd.read(4<<20),b''):h.update(chunk)
 return h.hexdigest()
release=json.loads((P/'P1_1_ODIN_RELEASE.json').read_text())
prior=json.loads((P/'P1_0_RELEASE.json').read_text())
assert release['P1_0_release_sha256']==sha(P/'P1_0_RELEASE.json')
assert release['eval_script_sha256']==sha(R/'audit_scripts/P1_SR_ODIN_eval.py')
assert release['n_models']==4 and prior['status']=='P1_0_REPLAY_VERIFIED_ODIN_AND_RFKD_NOT_YET_COMPLETE'
assert release['temperature']==1000 and release['pixel_epsilon']==0.0014
ids=10000
counts={'svhn':26032,'textures':5640,'lsun_crop':10000,'lsun_resize':10000,'places365':36500,'isun':8925}
events=[json.loads(x) for x in (P/'events_odin_10153.jsonl').read_text().splitlines() if x.strip()]
assert events[0]['status']=='P1_ODIN_HOST_START'
assert events[-1]['status']=='P1_ODIN_HOST_FINISHED'
assert sum(x['status']=='P1_ODIN_MODEL_DONE' for x in events)==4
assert not any(x['status']=='P1_ODIN_HOST_HALTED' for x in events)
def independent_metric(idvals,oodvals):
 y=np.r_[np.zeros(len(idvals),dtype=np.int8),np.ones(len(oodvals),dtype=np.int8)]
 s=np.r_[idvals,oodvals]
 fpr,tpr,threshold=roc_curve(y,s,drop_intermediate=False)
 auroc=float(roc_auc_score(y,s))
 precision,recall,threshold=precision_recall_curve(y,s)
 aupr=float(auc(recall,precision))
 # Important: tie-consistent step ROC definition may differ slightly from old
 # checker which used sample-level first-hit FPR95 without completing ties.
 ix=np.flatnonzero(tpr>=.95)
 fpr95=float(fpr[ix[0]]) if len(ix) else 1.
 return {'AUROC':auroc,'AUPR':aupr,'FPR95_complete_ties':fpr95}
rows=[];differ=[];source_hashes={};max_auroc=0.;max_aupr=0.;max_fpr95=0.
for dataset in release['dataset_order']:
 for role in release['role_order']:
  tag=dataset+'__'+role
  folder=P/'formal_odin'/tag
  assert folder.is_dir(),tag
  resultfile=folder/'results.json'
  done=json.loads((folder/'DONE.json').read_text())
  data=json.loads(resultfile.read_text())
  assert not data['probe'] and not done['probe'] and data['tag']==done['tag']==tag
  assert data['n_ID']==ids
  assert data['checkpoint_sha256']==prior['weights'][tag]['sha256']==done['checkpoint_sha256']
  assert sha(prior['weights'][tag]['checkpoint'])==done['checkpoint_sha256']
  assert done['results_sha256']==sha(resultfile)
  assert data['T']==release['temperature'] and data['pixel_epsilon']==release['pixel_epsilon']
  assert set(data['ODIN_scores'])==set(counts)
  assert done['id_scores_sha256']==sha(folder/'id_scores.npz')
  assert set(done['all_domains'])==set(counts)
  source_hashes[tag]={'results_json':sha(resultfile),'ID_npz':sha(folder/'id_scores.npz')}
  with np.load(folder/'id_scores.npz',allow_pickle=False) as src:
   iid={k:np.asarray(src[k],dtype=np.float64) for k in ('ODIN','temperature_only')}
  assert all(v.shape==(ids,) and np.isfinite(v).all() for v in iid.values())
  for domain,number in counts.items():
   e=data['ODIN_scores'][domain]
   assert e['n_ood']==number
   fn=folder/(domain+'__scores.npz')
   assert sha(fn)==e['score_array_sha256']
   with np.load(fn,allow_pickle=False) as src:
    ood={k:np.asarray(src[k],dtype=np.float64) for k in ('ODIN','temperature_only')}
   assert all(v.shape==(number,) and np.isfinite(v).all() for v in ood.values())
   for method,arr in (('ODIN','ODIN'),('T_only_control','temperature_only')):
    check=independent_metric(iid[arr],ood[arr])
    orig=e[method]
    auroc_error=abs(check['AUROC']-orig['AUROC'])
    aupr_error=abs(check['AUPR']-orig['AUPR'])
    fpr95_error=abs(check['FPR95_complete_ties']-orig['FPR95'])
    max_auroc=max(max_auroc,auroc_error);max_aupr=max(max_aupr,aupr_error)
    max_fpr95=max(max_fpr95,fpr95_error)
    assert auroc_error<1e-9,(tag,domain,method,'AUROC drift',auroc_error)
    assert aupr_error<1e-9,(tag,domain,method,'AUPR drift',aupr_error)
    rows.append({'dataset':dataset,'student':role,'ood':domain,'score':method,
     'AUROC':check['AUROC'],'AUPR':check['AUPR'],
     'FPR95_original_sample_based':orig['FPR95'],
     'FPR95_independent_completed_ties':check['FPR95_complete_ties'],
     'FPR95_difference_due_to_ties':fpr95_error,'source_results_sha256':sha(resultfile)})
    if fpr95_error>1e-10:
     differ.append({'dataset':dataset,'student':role,'ood':domain,'score':method,
       'difference':fpr95_error,'original':orig['FPR95'],'tie_safe':check['FPR95_complete_ties']})
  for method in ('ODIN','T_only_control'):
   for metric in ('AUROC','AUPR'):
    actual=float(np.mean([r[metric] for r in rows if r['dataset']==dataset and
             r['student']==role and r['score']==method]))
    assert abs(actual-data['macro'][method][metric])<1e-9
 print('P1_ODIN_VERIFIED_MODEL',tag,'ODIN_AUROC',data['macro']['ODIN']['AUROC'],
       'ODIN_FPR95_ORIGINAL',data['macro']['ODIN']['FPR95'],flush=True)
assert len(rows)==4*6*2
OUT.mkdir(exist_ok=False)
with (OUT/'P1_1_complete_per_domain_scores.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'P1_1_FPR95_tie_definition_differences.csv').open('x',newline='') as f:
 if differ:
  writer=csv.DictWriter(f,fieldnames=list(differ[0]));writer.writeheader();writer.writerows(differ)
 else:f.write('No sample-vs-completed-tie differences observed\n')
report={'status':'P1_ODIN_FINAL_AUDIT_PASS',
 'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'methods':['ODIN(T=1000,epsilon=0.0014)','temperature_only_control(T=1000)','historic_energy_neg','historic_msp_neg'],
 'models_verified':4,'domain_score_rows_verified':len(rows),
 'score_counts':counts,'source_hashes':source_hashes,
 'max_auroc_error':max_auroc,'max_aupr_error':max_aupr,
 'FPR95_original_vs_tie_safe_max_abs_diff':max_fpr95,
 'n_FPR95_definition_discrepancies':len(differ),
 'mandatory_caveat':'FPR95 in previous scores was sample-first-hit at TPR95; use fully completed ties and state convention if changed.',
 'not_primary_manuscript_result':True,
 'fair_comparison_scope':'Same historical frozen ResNet18 checkpoints, same ID and six OOD inputs, scoring only',
 'limitations':['4 single historical checkpoints, not 5-seed confirmation',
                'ODIN parameters fixed a priori not optimized on CIFAR OOD',
                'RFKD direct training baseline not provided'],
 'csv_sha256':{p.name:sha(p) for p in OUT.glob('*.csv')}}
with (OUT/'P1_1_FINAL_AUDIT_PASS.json').open('x') as f:
 json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
for p in OUT.iterdir():p.chmod(0o444)
print('P1_ODIN_FINAL_AUDIT_PASS',json.dumps({'rows':len(rows),
 'max_auroc_error':max_auroc,'max_AUPR_error':max_aupr,
 'n_FPR95_tie_effects':len(differ),'audit_sha256':sha(OUT/'P1_1_FINAL_AUDIT_PASS.json')},ensure_ascii=False),flush=True)
