#!/usr/bin/env python3
"""Energy-based OOD evaluation for P1-6 ImageNet-200 same-objective runs."""
import argparse,json,sys,time
from pathlib import Path
import os
import numpy as np,torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))
EXP=ROOT/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'revision_SR_AB_20261020/B2_V1.2_student_20261020/frozen_src'))
from im200_data_eval_recovery import dataset
from b2_protocol import eval_ood_scores
class AuxHead(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(512,128),nn.ReLU(inplace=True),nn.Linear(128,1))
 def forward(self,x): return self.net(x).squeeze(1)
class Student(nn.Module):
 def __init__(self):
  super().__init__(); self.backbone=resnet18(weights=None,num_classes=200); self.ood_head=AuxHead()
 def forward(self,x):
  b=self.backbone; z=b.maxpool(b.relu(b.bn1(b.conv1(x))))
  z=b.layer1(z); z=b.layer2(z); z=b.layer3(z); z=b.layer4(z); feat=b.avgpool(z).flatten(1)
  return b.fc(feat),self.ood_head(feat)
@torch.inference_mode()
def score_dataset(model,kind,dev,batch_size,workers):
 ds=dataset(kind); dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=False)
 scores=[]; correct=n=0; t=time.time()
 for bi,(x,y) in enumerate(dl):
  x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); z,_=model(x); z=z.float()
  scores.append((-torch.logsumexp(z,dim=1)).cpu().numpy())
  if kind=='test': correct+=(z.argmax(1)==y).sum().item(); n+=y.numel()
  if bi%40==0: print(f'EVAL {kind} batch={bi}/{len(dl)} sec={time.time()-t:.1f}',flush=True)
 return np.concatenate(scores), (correct/n if n else None)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
 run=Path(a.run_dir); ckpath=run/'best.pt'
 if not ckpath.is_file(): raise FileNotFoundError(ckpath)
 out=EXP/'evaluations'/run.name; out.mkdir(parents=True,exist_ok=False); dev=torch.device('cuda:0')
 ck=torch.load(ckpath,map_location='cpu',weights_only=False); model=Student(); model.load_state_dict(ck['model'],strict=True); model.to(dev).eval()
 domains=['ssb_hard','ninco','inaturalist','texture','openimage_o']; raw={}; metrics={}; idscore,idacc=score_dataset(model,'test',dev,a.batch_size,a.workers); raw['id_test_energy']=idscore
 for d in domains:
  s,_=score_dataset(model,d,dev,a.batch_size,a.workers); raw[d+'_energy']=s
  metrics[d]=eval_ood_scores(idscore,s)['OOD_positive']
 np.savez_compressed(out/'energy_scores_per_image.npz',**raw)
 def macro(names):
  return {k:float(np.mean([metrics[d][k] for d in names])) for k in ['AUROC','AUPR','FPR95_complete_ties']}
 result={'protocol':'CrossScore-SR-AB-P1-6-ImageNet200-same-objective-v1','run':run.name,'mode':ck.get('manifest',{}).get('mode'),
  'seed':ck.get('manifest',{}).get('seed'),'selected_epoch':ck.get('epoch'),'selected_checkpoint_val_acc':ck.get('val_acc'),
  'id_test_top1':idacc,'score':'-logsumexp(logits), higher=more OOD','positive_class':'OOD',
  'ood_metrics':metrics,'near_macro':macro(['ssb_hard','ninco']),
  'far_macro':macro(['inaturalist','texture','openimage_o']),
  'all_5_ood_macro':macro(domains),'raw_scores_file':str(out/'energy_scores_per_image.npz')}
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 (out/'COMPLETE.json').write_text(json.dumps({'run':run.name,'evaluated':True,'metrics_file':str(out/'metrics.json')},indent=2)+'\n')
 print('EVAL_COMPLETE '+json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__': main()
