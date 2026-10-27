#!/usr/bin/env python3
"""ImageNet-200 scale validation of the frozen CrossScore MSP three-loss objective.
The official image lists are unchanged. Training images are staged to node-local
tmpfs to avoid slow shared-filesystem image reads. Student is trained from scratch.
"""
import argparse,csv,hashlib,json,random,time,socket,sys,os
from pathlib import Path
import numpy as np,torch
import torch.nn as nn,torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,Dataset
from torchvision.models import resnet18
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))
# Keep this script's directory first so the local, env-configurable im200_data.py is used.
sys.path.insert(0,str(Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023/source'))
sys.path.insert(1,str(ROOT/'scripts'))
from im200_data import ImgList,LISTROOT,train_tf,eval_tf,dataset,TRAINROOT
from im200_models import teacher_restricted
class TrainSet(Dataset):
 def __init__(self): self.ds=ImgList(LISTROOT/'train_imagenet200.txt','train',train_tf())
 def __len__(self): return len(self.ds)
 def __getitem__(self,i): return self.ds[i]
class AuxHead(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(512,128),nn.ReLU(inplace=True),nn.Linear(128,1))
 def forward(self,x): return self.net(x).squeeze(1)
class Student(nn.Module):
 def __init__(self): super().__init__(); self.backbone=resnet18(weights=None,num_classes=200); self.ood_head=AuxHead()
 def forward(self,x):
  b=self.backbone; z=b.maxpool(b.relu(b.bn1(b.conv1(x))))
  z=b.layer1(z); z=b.layer2(z); z=b.layer3(z); z=b.layer4(z); feat=b.avgpool(z).flatten(1)
  return b.fc(feat),self.ood_head(feat)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()
def seedall(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=True
@torch.inference_mode()
def eval_acc(m,ld,dev,cap=0):
 m.eval(); c=n=0
 for i,(x,y) in enumerate(ld):
  if cap and i>=cap: break
  x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); z,_=m(x); c+=(z.argmax(1)==y).sum().item(); n+=y.numel()
 if not n: raise RuntimeError('empty validation')
 return c/n
def teacher_signals(z,T=6.):
 p=F.softmax(z.float()/T,dim=1); c=p.amax(1); e=-(p*p.clamp_min(1e-8).log()).sum(1)
 return c,e,1.-c
def rank_hinge(h,u,margin=.5):
 td=u[:,None]-u[None,:]; sd=h.float()[:,None]-h.float()[None,:]
 mask=~torch.eye(len(u),dtype=torch.bool,device=u.device)
 return F.relu(margin-td.sign()*sd)[mask].mean()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['CE','CrossScore_T6_Corrected'],required=True); p.add_argument('--variant',choices=['T1_E1_GRADDOWN','T6_E1_GRADUP'],required=True)
 p.add_argument('--seed',type=int,choices=[42,43,44,45,46],required=True); p.add_argument('--epochs',type=int,default=60)
 p.add_argument('--batch-size',type=int,default=512); p.add_argument('--workers',type=int,default=16)
 p.add_argument('--lr',type=float,default=.1); p.add_argument('--weight-decay',type=float,default=.0005)
 p.add_argument('--temperature',type=float,default=6.); p.add_argument('--margin',type=float,default=.5)
 p.add_argument('--max-train-batches',type=int,default=0); p.add_argument('--max-val-batches',type=int,default=0); p.add_argument('--out',required=True)
 a=p.parse_args()
 a.student_temperature={'T1_E1_GRADDOWN':1.,'T6_E1_GRADUP':6.}[a.variant]
 a.lambda_uncert={'T1_E1_GRADDOWN':0.25,'T6_E1_GRADUP':4.}[a.variant]
 a.lambda_conf={'T1_E1_GRADDOWN':0.20,'T6_E1_GRADUP':5.}[a.variant]
 if a.mode!='CrossScore_T6_Corrected': raise ValueError('factorial pilot CrossScore only')
 if a.epochs<1 or bool(a.max_train_batches)!=bool(a.max_val_batches): raise ValueError('invalid epochs/smoke caps')
 typ='SMOKE_P2B' if a.max_train_batches else 'P2B_GRADIENT_WEIGHT_ID_ONLY'; out=Path(a.out); out.mkdir(parents=True,exist_ok=False); (out/'.active').mkdir()
 start=time.time()
 try:
  if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
  seedall(a.seed); dev=torch.device('cuda:0'); model=Student().to(dev)
  teacher=tidx=None; tpath=ROOT/'results/im200_checkpoints/resnet50-0676ba61.pth'
  if a.mode=='CrossScore_T6_Corrected':
   teacher,tidx=teacher_restricted(); teacher=teacher.to(dev).eval().requires_grad_(False); tidx=tidx.to(dev)
  tr=TrainSet(); va=dataset('val')
  loader_generator=torch.Generator().manual_seed(a.seed)
  tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,
    persistent_workers=(a.workers>0),prefetch_factor=2 if a.workers>0 else None,generator=loader_generator,drop_last=False)
  vl=DataLoader(va,batch_size=512,shuffle=False,num_workers=min(8,a.workers),pin_memory=True,
    persistent_workers=(a.workers>0),prefetch_factor=2 if a.workers>0 else None)
  opt=SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=a.weight_decay,nesterov=False)
  steps=min(len(tl),a.max_train_batches) if a.max_train_batches else len(tl)
  sch=CosineAnnealingLR(opt,T_max=max(1,a.epochs*steps),eta_min=1e-6)
  scaler=torch.cuda.amp.GradScaler(init_scale=1024.,enabled=True)
  manifest={'protocol':'P2-B-ImageNet200-frozen-gradient-weight-intervention-IDonly-v1','run_type':typ,'mode':a.mode,'seed':a.seed,
   'student':'ResNet-18, 200 classes, random initialization','teacher':'ImageNet-1K-pretrained ResNet-50 restricted to ImageNet-200 WNIDs' if teacher else None,
   'teacher_checkpoint':str(tpath) if teacher else None,'teacher_checkpoint_sha256':sha256(tpath) if teacher else None,
   'teacher_train_scope':'pretrained on ImageNet-1K; not fine-tuned on ImageNet-200','id_train_n':len(tr),'id_val_n':len(va),
   'ood_data_used_for_training':False,'selection':'ID validation accuracy only; epoch-0 eligible',
   'train_transform':'RandomResizedCrop(224)+RandomHorizontalFlip+ImageNet normalization; teacher and student share the same augmented batch',
   'eval_transform':'Resize(256)+CenterCrop(224)+ImageNet normalization','epochs':a.epochs,'batch_size':a.batch_size,
   'optimizer':'SGD(momentum=.9,Nesterov=False)','lr':a.lr,'weight_decay':a.weight_decay,'scheduler':'per-step cosine to 1e-6',
   'teacher_temperature':a.temperature,'student_probability_temperature':a.student_temperature,'lambda_conf':a.lambda_conf,'lambda_order':1.,
   'lambda_uncert':a.lambda_uncert,'variant':a.variant,'experimental_purpose':'P2-B gradient magnitude exploratory causality test; not submitted method; no OOD for checkpoint selection', 'rank_margin':a.margin,'energy_score':'-logsumexp(student logits); higher=more OOD',
   'max_train_batches':a.max_train_batches,'max_val_batches':a.max_val_batches,'host':socket.gethostname(),
   'train_root':str(TRAINROOT),'num_workers':a.workers,'persistent_workers':bool(a.workers>0),'prefetch_factor':2 if a.workers>0 else None,
   'train_data_staged_to_node_local_tmpfs':str(TRAINROOT).startswith('/dev/shm/')}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  with (out/'history.csv').open('w',newline='') as f: csv.writer(f).writerow(['epoch','lr','loss','ce','conf','order','uncert','train_acc','val_acc','skipped_amp_steps','sec'])
  best=eval_acc(model,vl,dev,a.max_val_batches)
  torch.save({'model':model.state_dict(),'arch':'resnet18_aux','num_classes':200,'epoch':0,'val_acc':best,'manifest':manifest},out/'best.pt')
  print(f'BASE epoch=0 val_acc={best:.6f} mode={a.mode} seed={a.seed}',flush=True)
  for ep in range(1,a.epochs+1):
   t0=time.time(); model.train(); S={k:0. for k in ['loss','ce','conf','order','uncert','correct','n']}; skipped=0
   for bi,(x,y) in enumerate(tl):
    if a.max_train_batches and bi>=a.max_train_batches: break
    x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
     z,h=model(x); z=z.float(); ce=F.cross_entropy(z,y); conf=z.sum()*0.; order=z.sum()*0.; uncert=z.sum()*0.
     if teacher is not None:
      with torch.no_grad(): tz=teacher(x).float().index_select(1,tidx)
      tc,te,tu=teacher_signals(tz,a.temperature); sp=F.softmax(z/a.student_temperature,1); sc=sp.amax(1); se=-(sp*sp.clamp_min(1e-8).log()).sum(1)
      conf=F.mse_loss(sc,tc); uncert=F.mse_loss(se,te); order=rank_hinge(h,tu,a.margin)
     loss=ce+a.lambda_conf*conf+order+a.lambda_uncert*uncert
    oldscale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if scaler.get_scale()>=oldscale: sch.step()
    else: skipped+=1
    n=y.numel()
    for k,v in [('loss',loss),('ce',ce),('conf',conf),('order',order),('uncert',uncert)]: S[k]+=float(v.detach())*n
    S['correct']+=(z.argmax(1)==y).sum().item(); S['n']+=n
   vaa=eval_acc(model,vl,dev,a.max_val_batches); n=max(1,int(S['n']))
   row=[ep,opt.param_groups[0]['lr'],S['loss']/n,S['ce']/n,S['conf']/n,S['order']/n,S['uncert']/n,S['correct']/n,vaa,skipped,time.time()-t0]
   with (out/'history.csv').open('a',newline='') as f: csv.writer(f).writerow(row)
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':scaler.state_dict(),'epoch':ep,'val_acc':vaa,'manifest':manifest},out/'last.pt')
   print('EPOCH '+json.dumps({'epoch':ep,'mode':a.mode,'seed':a.seed,'loss':row[2],'conf':row[4],'order':row[5],'uncert':row[6],'train_acc':row[7],'val_acc':vaa,'skipped_amp_steps':skipped,'sec':row[-1]},separators=(',',':')),flush=True)
   if vaa>best:
    best=vaa; torch.save({'model':model.state_dict(),'arch':'resnet18_aux','num_classes':200,'epoch':ep,'val_acc':best,'manifest':manifest},out/'best.pt')
  summary={'protocol':manifest['protocol'],'run_type':typ,'mode':a.mode,'seed':a.seed,'best_val_acc':best,
   'elapsed_seconds':time.time()-start,'best_checkpoint':str(out/'best.pt'),'complete_epochs':a.epochs}
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
  print('COMPLETE '+json.dumps(summary,separators=(',',':')),flush=True)
 finally:
  (out/'.active').rmdir()
if __name__=='__main__': main()
