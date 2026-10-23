"""CrossScore SR-B2 V1.0: testable protocol primitives.

Legacy loss is deliberately identical to awareKD1.zip implementation; corrected T6
is separately named and must not be described as the historical algorithm.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF

SEEDS=(101,202,303,404,505)
SPLIT_SEED=20261020
MEAN=(.4914,.4822,.4465)
STD=(.2470,.2435,.2616)
NUM_CLASSES={'cifar10':10,'cifar100':100}
OOD=('svhn','textures','lsun_crop','lsun_resize','places365','isun')
PROTOCOL_NAME='CrossScore-SR-AB-B2-V1.0-20261020'
METHODS=('RFKD_EnergyOrder_T1_PublishedSpec','RFKD_EnergyOrder_T6_Matched')

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(512,128),nn.ReLU(inplace=True),nn.Linear(128,1))
    def forward(self,x):return self.mlp(x).squeeze(1)

class Student(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        self.backbone.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.backbone.maxpool=nn.Identity()
        self.backbone.fc=nn.Linear(512,num_classes)
        self.ood_head=Head()
    def forward(self,x):
        z=self.backbone.conv1(x)
        z=self.backbone.bn1(z);z=self.backbone.relu(z);z=self.backbone.maxpool(z)
        z=self.backbone.layer1(z);z=self.backbone.layer2(z)
        z=self.backbone.layer3(z);z=self.backbone.layer4(z)
        feat=self.backbone.avgpool(z).flatten(1)
        return self.backbone.fc(feat),self.ood_head(feat)

class Teacher(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone=models.resnet50(weights=None)
        self.backbone.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.backbone.maxpool=nn.Identity()
        self.backbone.fc=nn.Linear(2048,num_classes)
    def forward(self,x):return self.backbone(x)

def full_sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fh:
        for block in iter(lambda:fh.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def weights_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def fixed_split(n=50000,seed=SPLIT_SEED):
    if n!=50000:raise ValueError('Protocol requires 50000 official training images')
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    train=order[5000:];val=order[:5000]
    assert len(set(train)&set(val))==0 and len(set(train)|set(val))==50000
    return train,val

class DeterministicCIFARDataset(Dataset):
    """Frozen RandomCrop(pad=4)+RandomHorizontalFlip with deterministic sample+epoch seed.

    This removes dependence on worker scheduling and supports exact resume and paired seeds.
    """
    def __init__(self,official_train,indices,seed,training):
        self.base=official_train;self.indices=list(indices)
        self.seed=seed;self.training=training;self.epoch=0
        self.to_tensor=transforms.ToTensor()
        self.norm=transforms.Normalize(MEAN,STD)
    def set_epoch(self,n):self.epoch=int(n)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        src=self.indices[i];pil,label=self.base[src]
        if self.training:
            g=torch.Generator().manual_seed((self.seed*1000000007+self.epoch*1000003+src)%(2**63-1))
            pil=TF.pad(pil,[4,4,4,4],fill=0)
            y=int(torch.randint(0,9,(1,),generator=g).item())
            x=int(torch.randint(0,9,(1,),generator=g).item())
            pil=TF.crop(pil,y,x,32,32)
            if torch.rand((),generator=g).item()<.5:pil=TF.hflip(pil)
        return self.norm(self.to_tensor(pil)),label

def get_train_val(dataset_name,root,seed,batch_size=128,num_workers=4):
    cls={'cifar100':datasets.CIFAR100,'cifar10':datasets.CIFAR10}[dataset_name]
    # ONLY the official training split is loaded. No CIFAR official test-set access.
    base=cls(root=str(Path(root)/dataset_name),train=True,download=False,transform=None)
    train_idx,val_idx=fixed_split(len(base))
    tr=DeterministicCIFARDataset(base,train_idx,seed,True)
    va=DeterministicCIFARDataset(base,val_idx,seed,False)
    return tr,va,train_idx,val_idx

def make_loader(ds,seed,epoch,training,batch_size=128,workers=4):
    ds.set_epoch(epoch)
    g=torch.Generator().manual_seed(seed+epoch*1000003)
    return DataLoader(ds,batch_size=batch_size,shuffle=training,num_workers=workers,
        generator=g,pin_memory=torch.cuda.is_available(),drop_last=False,
        persistent_workers=False)

def teacher_signals(logits,temperature):
    p=F.softmax(logits/temperature,dim=1)
    conf=p.amax(dim=1)
    ent=-(p*p.clamp_min(1e-8).log()).sum(dim=1)
    return conf,ent,1-conf

def rank_hinge(shead,score,margin=.5):
    tdiff=score[:,None]-score[None,:]
    sdiff=shead[:,None]-shead[None,:]
    mask=~torch.eye(len(score),dtype=torch.bool,device=score.device)
    return F.relu(margin-tdiff.sign()*sdiff)[mask].mean()

def legacy_loss(slog,tlog,shead):
    # Exact MSP-based behavior of awareKD1.zip under --teacher_score_type=msp.
    with torch.no_grad():
        tconf,tent,tscore=teacher_signals(tlog,1.)
    sconf,sent,_=teacher_signals(slog,1.)
    conf=F.mse_loss(sconf,tconf)
    ent=F.mse_loss(sent,tent)
    rank=rank_hinge(shead,tscore)
    return {'loss_conf':conf,'loss_uncert':ent,'loss_order':rank,'loss_total':conf+ent+rank}

def corrected_loss(slog,tlog,shead,T=6.,lambda_conf=1.,lambda_order=1.,lambda_uncert=1.):
    # Teacher alone uses T=6 for MSP/confidence, entropy and MSP ordering.
    with torch.no_grad():
        tconf,tent,tscore=teacher_signals(tlog,T)
    sconf,sent,_=teacher_signals(slog,1.)
    conf=F.mse_loss(sconf,tconf)
    ent=F.mse_loss(sent,tent)
    rank=rank_hinge(shead,tscore)
    return {'loss_conf':conf,'loss_uncert':ent,'loss_order':rank,
        'loss_total':lambda_conf*conf+lambda_uncert*ent+lambda_order*rank}

def objective(method,slog,shead,labels,tlog=None,loss_lambdas=(1.,1.,1.)):
    ce=F.cross_entropy(slog,labels)
    if method=='CE':return ce,{'ce':float(ce.detach()),'distill':0.}
    if tlog is None:raise ValueError('Teacher logits required for '+method)
    if method=='VanillaKD_T4':
        kl=F.kl_div(F.log_softmax(slog/4.,dim=1),F.softmax(tlog.detach()/4.,dim=1),reduction='batchmean')*(4.**2)
        return ce+kl,{'ce':float(ce.detach()),'distill':float(kl.detach())}
    if method=='CrossScore_LegacyAsCode':parts=legacy_loss(slog,tlog,shead)
    elif method=='CrossScore_T6_Corrected':parts=corrected_loss(slog,tlog,shead,6.,*loss_lambdas)
    else:raise ValueError(method)
    return ce+parts['loss_total'],{'ce':float(ce.detach()),'distill':float(parts['loss_total'].detach()),
        'conf':float(parts['loss_conf'].detach()),'order':float(parts['loss_order'].detach()),'uncert':float(parts['loss_uncert'].detach())}

def _sorted_roc(y,s):
    y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
    if y.size==0 or y.min()==y.max() or not np.isfinite(s).all():raise ValueError('Invalid labels/scores')
    order=np.argsort(s,kind='stable')[::-1]
    ss=s[order]; yy=y[order]
    distinct=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(yy)[distinct];fp=np.cumsum(1-yy)[distinct]
    tpr=np.r_[0,tp/tp[-1]];fpr=np.r_[0,fp/fp[-1]]
    prec=tp/(tp+fp);rec=tp/tp[-1]
    auroc=float(np.trapz(tpr,fpr))
    aupr=float(np.trapz(np.r_[1,prec],np.r_[0,rec]))
    ix=np.flatnonzero(tpr>=.95)
    fpr95=float(fpr[ix[0]]) if ix.size else 1.
    return {'AUROC':auroc,'AUPR':aupr,'FPR95_complete_ties':fpr95}

def eval_ood_scores(id_scores,ood_scores):
    a=np.asarray(id_scores);b=np.asarray(ood_scores)
    y=np.r_[np.zeros(len(a),dtype=np.int8),np.ones(len(b),dtype=np.int8)]
    s=np.r_[a,b]
    return {'OOD_positive':_sorted_roc(y,s),
            'ID_positive':_sorted_roc(1-y,-s)}
