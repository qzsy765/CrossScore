#!/usr/bin/env python3
"""P1-3 reviewer-requested feature/logit/energy diagnostics from original historical weights.
No model training, no hyperparameter tuning or score optimization; diagnostics only.
Published manuscript method remains normative. Any class-label feature geometric statistics
use official ID test labels as posthoc descriptive diagnostics, NOT OOD test-time calibration.
"""
import argparse,sys,json,hashlib,time,os,datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
P1=ROOT/'P1_SR_ReviewerEvidence_20261022'
STAGE=P1/'P1_3_ID_OOD_Representation_Diagnostics_20261022'
sys.path.insert(0,str(ROOT/'audit_scripts'))
from eval_uploaded_checkpoints import Wrapped,id_loader,loader_for,MEAN,STD,OOD_NAMES
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 with p.open('x') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def load_eval(model,dl,is_ID,max_batches=None):
 feat=[None]
 def hook(mod,inp,out):feat[0]=out.detach().flatten(1)
 h=model.backbone.avgpool.register_forward_hook(hook)
 rows={key:[] for key in ('energy','msp','margin_top1_top2','logit_mean','logit_std',
      'max_logit','feature_L2','feature_abs_mean','logit_top1','logit_top2')}
 labels=[];preds=[];idfeat=[];n=0
 try:
  with torch.no_grad():
   for j,(x,y) in enumerate(dl):
    if max_batches is not None and j>=max_batches:break
    x=x.cuda(non_blocking=True)
    z=model(x).float()
    f=feat[0]
    assert f is not None and f.shape==(len(x),512)
    zz=torch.topk(z,2,dim=1).values
    p=torch.softmax(z,dim=1).amax(dim=1)
    sc={'energy':-torch.logsumexp(z,dim=1),'msp':1-p,
        'margin_top1_top2':zz[:,0]-zz[:,1],
        'logit_mean':z.mean(dim=1),
        'logit_std':z.std(dim=1,unbiased=False),
        'max_logit':zz[:,0],'feature_L2':torch.linalg.vector_norm(f,dim=1),
        'feature_abs_mean':torch.mean(torch.abs(f),dim=1),
        'logit_top1':zz[:,0],'logit_top2':zz[:,1]}
    for key in rows:
     a=sc[key].detach().cpu().numpy().astype('float32')
     assert np.isfinite(a).all()
     rows[key].append(a)
    if is_ID:
     idfeat.append(f.cpu().numpy().astype('float32'))
     labels.append(y.numpy().astype('int64'))
     preds.append(z.argmax(dim=1).cpu().numpy().astype('int64'))
    n+=len(x)
    if j and j%140==0:print('P1_FEATURE_PROGRESS','ID' if is_ID else 'OOD',j,'batches',n,'images',flush=True)
 finally:h.remove()
 scores={name:np.concatenate(vec) for name,vec in rows.items()}
 assert all(len(z)==n and np.isfinite(z).all() for z in scores.values())
 return scores, (np.concatenate(idfeat) if is_ID else None),\
            (np.concatenate(labels) if is_ID else None),\
            (np.concatenate(preds) if is_ID else None),n
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--dataset',choices=('cifar10','cifar100'),required=True)
 parser.add_argument('--role',choices=('CE','CrossScore_OriginalWeightCandidate'),required=True)
 parser.add_argument('--probe',action='store_true')
 parser.add_argument('--batch-size',type=int,default=128)
 parser.add_argument('--workers',type=int,default=2)
 a=parser.parse_args()
 import socket
 release=json.loads((STAGE/'P1_3_RELEASE.json').read_text())
 if sha(__file__)!=release['evaluator_sha256']:raise RuntimeError('Evaluator SHA drift')
 assert release['no_training'] and not release['OOD_score_selection']
 info=json.loads((P1/'P1_0_RELEASE.json').read_text())['weights'][a.dataset+'__'+a.role]
 assert sha(info['checkpoint'])==info['sha256']
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 tag=a.dataset+'__'+a.role
 where=STAGE/('probe' if a.probe else 'formal')/tag
 if where.exists():raise FileExistsError('Output exists '+str(where))
 model=Wrapped(False,100 if a.dataset=='cifar100' else 10)
 model.load_state_dict(torch.load(info['checkpoint'],weights_only=True,map_location='cpu'),strict=True)
 model.eval().cuda().requires_grad_(False)
 dl,_=id_loader(a.dataset,a.batch_size,a.workers)
 idsc,features,labels,pred,ni=load_eval(model,dl,True,1 if a.probe else None)
 assert ni==(a.batch_size if a.probe else 10000),ni
 assert features.shape==(ni,512) and labels.shape==(ni,) and pred.shape==(ni,)
 where.mkdir(parents=True,exist_ok=False)
 idfile=where/'id_diagnostic_scores_features.npz'
 np.savez_compressed(idfile,**idsc,features=features,labels=labels,pred=pred)
 info_out={'tag':tag,'probe':a.probe,'dataset':a.dataset,'role':a.role,
           'checkpoint_sha256':info['sha256'],'release_sha256':sha(STAGE/'P1_3_RELEASE.json'),
           'n_ID':ni,'ID_accuracy':float((labels==pred).mean()),
           'ID_results_sha256':sha(idfile),
           'metrics':'P1-3 frozen logit scale, margin, mean, energy, feature norm and ID feature space',
           'label_usage':'ID test labels used only for descriptive diagnostics; no detector fitting',
           'OOD':{}}
 for name in OOD_NAMES:
  dl,expect=loader_for(name,a.batch_size,a.workers)
  score,_,_,_,n=load_eval(model,dl,False,1 if a.probe else None)
  assert n==(a.batch_size if a.probe else expect),(name,n,expect)
  out=where/(name+'__diagnostics.npz')
  np.savez_compressed(out,**score)
  info_out['OOD'][name]={'count':n,'sha256':sha(out)}
  print('P1_FEATURE_OOD_DONE',tag,name,n,'mean_energy',float(np.mean(score['energy'])),
   'mean_logit_std',float(np.mean(score['logit_std'])),'mean_feature_L2',float(np.mean(score['feature_L2'])),flush=True)
 info_out['utc_complete']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 dump(where/'results.json',info_out)
 dump(where/'DONE.json',{'tag':tag,'results_sha256':sha(where/'results.json'),
  'checkpoint_sha256':info['sha256'],'probe':a.probe,'n_ID':ni,'OOD_domains':list(OOD_NAMES)})
 for x in where.iterdir():x.chmod(0o444)
 print('P1_FEATURE_MODEL_COMPLETE',tag,'ID_acc',info_out['ID_accuracy'],flush=True)
if __name__=='__main__':main()
