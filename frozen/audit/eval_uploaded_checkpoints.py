#!/usr/bin/env python3
"""Scientific Reports B: read-only exact historical-score checkpoint evaluation.
Never trains or edits source checkpoints. Results written to separate revision dir.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
# Metrics implemented in NumPy; validated against scikit-learn ROC AUC and PR-curve AUC.

SRC=Path("/root/private_data/CrossScore_SR_AB_upload_20261020/checkpoints_inbox/checkpoints")
DATA=Path("/root/private_data/datasets/public")
OUT=Path("/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020/eval_original_checkpoints")
OOD_NAMES=("svhn","textures","lsun_crop","lsun_resize","places365","isun")
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

# Minimal MATLAB v5 image-array reader for public SVHN test_32x32.mat.
# Avoids modifying the established PyTorch environment to add scipy.
import struct, zlib
from PIL import Image
from torch.utils.data import Dataset

def load_mat5_arrays(raw):
    def item(buf,pos):
        w1,w2=struct.unpack_from("<II",buf,pos)
        if w1>>16:
            typ=w1&65535; n=w1>>16
            return typ,buf[pos+4:pos+4+n],pos+8
        typ=w1;n=w2
        return typ,buf[pos+8:pos+8+n],pos+8+(n if typ==15 else ((n+7)//8)*8)
    def matrix(blob):
        typ,block,_=item(blob,0)
        if typ!=14: raise ValueError("Unsupported MATLAB element "+str(typ))
        pos=0;parts=[]
        while pos<len(block):
            typ,payload,pos=item(block,pos)
            parts.append((typ,payload))
        dims=tuple(np.frombuffer(parts[1][1],dtype="<i4"))
        name=parts[2][1].decode("utf8")
        typ,data=parts[3]
        dt={1:"i1",2:"u1",3:"<i2",4:"<u2",5:"<i4",6:"<u4",9:"<f8",12:"<i8",13:"<u8"}[typ]
        return name,np.frombuffer(data,dtype=dt).reshape(dims,order="F")
    out={}
    pos=128
    while pos+8<=len(raw):
        typ,data,nxt=item(raw,pos)
        if typ==15: out.update([matrix(zlib.decompress(data))])
        elif typ==14: out.update([matrix(raw[pos:nxt])])
        else: raise ValueError("Unexpected MATLAB block "+str(typ))
        pos=nxt
    return out

class SVHN_NoSciPy(Dataset):
    def __init__(self,path,transform):
        obj=load_mat5_arrays(Path(path).read_bytes())
        x=obj["X"];y=obj["y"].reshape(-1).astype(np.int64)
        if x.shape[:3]!=(32,32,3) or x.shape[3]!=len(y):
            raise ValueError("Unexpected SVHN shapes")
        y[y==10]=0
        self.x=np.transpose(x,(3,2,0,1)).copy()
        self.y=y;self.transform=transform
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        pic=Image.fromarray(np.transpose(self.x[i],(1,2,0)))
        return self.transform(pic),int(self.y[i])

class LightHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(512,128),nn.ReLU(inplace=True),nn.Linear(128,1))
    def forward(self,x): return self.mlp(x).squeeze(1)

class Wrapped(nn.Module):
    def __init__(self,teacher,nc):
        super().__init__()
        self.backbone=models.resnet50(weights=None) if teacher else models.resnet18(weights=None)
        self.backbone.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.backbone.maxpool=nn.Identity()
        self.backbone.fc=nn.Linear(self.backbone.fc.in_features,nc)
        if not teacher:self.ood_head=LightHead()
    def forward(self,x):return self.backbone(x)

def loader_for(name,bs,num_workers):
    tr=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    if name=="svhn": ds=SVHN_NoSciPy(DATA/"svhn"/"test_32x32.mat",transform=tr)
    elif name=="textures": ds=datasets.ImageFolder(str(DATA/"texture"),transform=tr)
    elif name=="lsun_crop": ds=datasets.ImageFolder(str(DATA/"lsun_crop"),transform=tr)
    elif name=="lsun_resize": ds=datasets.ImageFolder(str(DATA/"lsun_resize"),transform=tr)
    elif name=="places365": ds=datasets.ImageFolder(str(DATA/"places365"),transform=tr)
    elif name=="isun": ds=datasets.ImageFolder(str(DATA/"isun"),transform=tr)
    else: raise ValueError(name)
    if len(ds)==0:raise RuntimeError("Empty OOD dataset "+name)
    return DataLoader(ds,batch_size=bs,shuffle=False,num_workers=num_workers,
                      pin_memory=True),len(ds)

def id_loader(dataset,bs,num_workers):
    tr=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    if dataset=="cifar100":
        ds=datasets.CIFAR100(root=str(DATA/"cifar100"),train=False,download=False,transform=tr)
    else:
        ds=datasets.CIFAR10(root=str(DATA/"cifar10"),train=False,download=False,transform=tr)
    return DataLoader(ds,batch_size=bs,shuffle=False,num_workers=num_workers,pin_memory=True),len(ds)

@torch.inference_mode()
def scores_and_correct(m, loader, dev, score_types):
    scores={s:[] for s in score_types};correct=0;n=0
    for x,y in loader:
        z=m(x.to(dev,non_blocking=True)).float()
        if "energy_neg" in scores: scores["energy_neg"].append((-torch.logsumexp(z,dim=1)).cpu().numpy())
        if "msp_neg" in scores: scores["msp_neg"].append((1.0-torch.softmax(z,dim=1).amax(dim=1)).cpu().numpy())
        if "entropy" in scores:
            p=torch.softmax(z,dim=1);scores["entropy"].append((-(p * p.clamp_min(1e-12).log()).sum(dim=1)).cpu().numpy())
        correct+=int((z.argmax(dim=1).cpu()==y).sum());n+=y.numel()
    return {k:np.concatenate(v) for k,v in scores.items()},correct/max(n,1),n

def measure(a,b):
    yy=np.concatenate((np.zeros_like(a),np.ones_like(b)))
    sc=np.concatenate((a,b))
    rank=np.argsort(sc)[::-1]
    y=yy[rank]
    tps=np.cumsum(y)
    fps=1+np.arange(len(y))-tps
    pnum=tps[-1]
    nnum=fps[-1]
    tpr=tps/pnum
    fpr=fps/nnum
    indices=np.where(tpr>=0.95)[0]
    fpr95=float(fpr[indices[0]]) if len(indices) else 1.0
    # The first/last index of each distinct score threshold reproduces
    # sklearn.metrics ROC AUC and precision_recall_curve + auc.
    distinct=np.r_[np.where(np.diff(sc[rank]))[0],len(rank)-1]
    ut=tps[distinct];uf=fps[distinct]
    auroc=float(np.trapz(np.r_[0,ut/pnum],np.r_[0,uf/nnum]))
    prec=ut/(ut+uf)
    recall=ut/pnum
    aupr=float(np.trapz(np.r_[1,prec],np.r_[0,recall]))
    return {"AUROC":auroc,"FPR95":fpr95,"AUPR":aupr}
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["cifar10","cifar100"],required=True)
    p.add_argument("--models",default="teacher_baseline,student_baseline_no_distill,S0507_temper60_msp_alpha00dim128")
    p.add_argument("--oods",default=",".join(OOD_NAMES))
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--scores",default="energy_neg,msp_neg")
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA not available")
    torch.backends.cudnn.benchmark=True
    device=torch.device("cuda:0")
    chosen=a.models.split(",")
    odnames=a.oods.split(",")
    types=a.scores.split(",")
    for t in types:
        assert t in ("energy_neg","msp_neg","entropy")
    iddata,idl=id_loader(a.dataset,a.batch_size,a.num_workers)
    oods={}
    for od in odnames:
        oods[od]=loader_for(od,a.batch_size,a.num_workers)
    print("DATASET",a.dataset,"ID",idl,"OOD_COUNT",{od:v[1] for od,v in oods.items()},flush=True)
    output_dir=OUT/a.dataset
    output_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for name in chosen:
        is_teacher=(name=="teacher_baseline")
        stem=(SRC/a.dataset/"teacher_baseline.pth") if is_teacher else (SRC/a.dataset/"MSP"/(name+".pth"))
        if not stem.is_file():raise FileNotFoundError(stem)
        model=Wrapped(is_teacher,100 if a.dataset=="cifar100" else 10)
        state=torch.load(stem,map_location="cpu",weights_only=True)
        model.load_state_dict(state,strict=True)
        model.to(device).eval()
        print("STRICT_LOAD_OK",name,len(state),"sha256",digest(stem)[:18],flush=True)
        sid,acc,_=scores_and_correct(model,iddata,device,types)
        result={"dataset":a.dataset,"model":name,"source_checkpoint":str(stem),
            "sha256":digest(stem),"id_count":idl,"id_acc":acc,
            "ood_counts":{od:vv[1] for od,vv in oods.items()},
            "normalization":{"mean":MEAN,"std":STD},
            "ood_transform":"Resize(32);CenterCrop(32);ToTensor;Normalize",
            "score_direction":"higher = more OOD",
            "scores":{}, "computed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        for typ in types:
            result["scores"][typ]={"per_ood":{}}
        for od,(ld,n) in oods.items():
            so,_,_=scores_and_correct(model,ld,device,types)
            for typ in types:
                m=measure(sid[typ],so[typ])
                result["scores"][typ]["per_ood"][od]=m
                print("RESULT",name,typ,od,"n",n,"AUROC",round(m["AUROC"],5),"FPR95",round(m["FPR95"],5),"AUPR",round(m["AUPR"],5),flush=True)
        for typ in types:
            dat=result["scores"][typ]["per_ood"]
            result["scores"][typ]["macro"]={metric:float(np.mean([v[metric] for v in dat.values()])) for metric in ("AUROC","FPR95","AUPR")}
            print("MACRO",name,typ,"IDacc",round(acc,5),result["scores"][typ]["macro"],flush=True)
        f=output_dir/(name+"__"+a.oods.replace(",","-")+".json")
        if f.exists():raise FileExistsError("Refusing overwrite "+str(f))
        f.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        summary.append(result)
        del model
        torch.cuda.empty_cache()
    print("COMPLETE",len(summary),"models",flush=True)
if __name__=="__main__":main()
