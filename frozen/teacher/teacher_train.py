#!/usr/bin/env python3
"""B2 V1.2 PRE-REGISTERED clean-teacher trainer.
Only CIFAR train=True (45k train / 5k val); independent official test is NEVER loaded.
Primary teacher is FIXED epoch-200 FINAL, not best val. New stage only.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse, datetime, hashlib, json, random, subprocess, sys, time, socket, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from b2_protocol import Teacher,SEEDS,SPLIT_SEED,NUM_CLASSES,seed_everything,weights_hash,full_sha256,get_train_val,make_loader

STAGE=Path("/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020/B2_V1.2_teacher_20261020")
PROTOCOL=STAGE/"V1.2_teacher_protocol.md"
TR_SEED=20261020
TRAIN_N,VAL_N=45000,5000
BATCH=128
EPOCHS=200
MIN_FREE_GB=30
def save_torch(obj,p):
    tmp=p.with_name(p.name+".pending."+str(os.getpid()))
    if tmp.exists():raise FileExistsError(tmp)
    torch.save(obj,tmp)
    os.replace(tmp,p)
def save_json(obj,p):
    tmp=p.with_name(p.name+".pending."+str(os.getpid()))
    if tmp.exists():raise FileExistsError(tmp)
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    os.replace(tmp,p)
def rng_save():
    st=np.random.get_state()
    return {"python":random.getstate(),"numpy":[st[0],st[1].tolist(),st[2],st[3],st[4]],
            "cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def rng_load(st):
    random.setstate(st["python"])
    x=st["numpy"]; np.random.set_state((x[0],np.asarray(x[1],dtype="uint32"),x[2],x[3],x[4]))
    torch.set_rng_state(st["cpu"])
    torch.cuda.set_rng_state_all(st["cuda"])
def gpu_idle():
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True,capture_output=True,timeout=15)
    if p.returncode!=0:raise RuntimeError("nvidia-smi compute query failed: "+p.stderr)
    values=[x.strip() for x in p.stdout.splitlines() if x.strip() and x.strip() not in ("No running processes found","[Not Supported]")]
    if values:raise RuntimeError("GPU ALREADY BUSY "+str(values))
def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,choices=("cifar10","cifar100"))
    p.add_argument("--mode",required=True,choices=("smoke","formal"))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--run-label",default="")
    p.add_argument("--smoke-epochs",type=int,default=2)
    p.add_argument("--smoke-train-batches",type=int,default=2)
    p.add_argument("--smoke-val-batches",type=int,default=2)
    p.add_argument("--stop-after-epoch",type=int,default=0)
    return p.parse_args()
@torch.inference_mode()
def validate(m,dl,dev,max_batches=None):
    m.eval();n=0;correct=0;loss=0.
    for k,(x,y) in enumerate(dl):
        if max_batches and k>=max_batches:break
        x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True)
        z=m(x);v=F.cross_entropy(z,y)
        n+=len(y);correct+=int((z.argmax(1)==y).sum());loss+=float(v)*len(y)
    if not n:raise ValueError("zero val examples")
    return {"val_n":n,"val_loss":loss/n,"val_acc":correct/n}
def main():
    a=parser()
    if a.mode=="formal":
        if a.run_label or a.stop_after_epoch:raise RuntimeError("Smoke parameters forbidden in formal")
    else:
        if not a.run_label or "/" in a.run_label or ".." in a.run_label:raise ValueError("Smoke label required")
        if not (1<=a.smoke_epochs<=3):raise ValueError('Smoke epochs must be 1..3')
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if a.workers<0 or a.workers>8:raise RuntimeError("Unfrozen number of workers")
    if not PROTOCOL.is_file():raise RuntimeError("Missing V1.2 amended protocol")
    if not (STAGE/"TEACHER_RETRAIN_AUTHORIZED").is_file():raise RuntimeError("Teacher re-training not authorized")
    if shutil.disk_usage(STAGE).free<MIN_FREE_GB*2**30:raise RuntimeError("Shared disk low")
    if not a.resume:gpu_idle()
    torch.use_deterministic_algorithms(True)
    seed_everything(TR_SEED)
    device=torch.device("cuda")
    epochs=EPOCHS if a.mode=="formal" else a.smoke_epochs
    if a.mode=="smoke" and not (1<=epochs<=3):raise RuntimeError("Bad smoke epochs")
    if a.mode=="smoke" and min(a.smoke_train_batches,a.smoke_val_batches)<1:raise RuntimeError("bad cap")
    name=a.dataset+("__SMOKE_"+a.run_label if a.mode=="smoke" else "__TEACHER200")
    jobdir=STAGE/("smoke_runs" if a.mode=="smoke" else "formal_runs")/name
    jobdir.parent.mkdir(exist_ok=True)
    if not a.resume:jobdir.mkdir(exist_ok=False)
    elif not jobdir.is_dir():raise RuntimeError("No run to resume")
    lease=jobdir/".active"
    lease.mkdir(exist_ok=False)
    start=time.monotonic()
    try:
        if (jobdir/"DONE.json").exists():raise RuntimeError("DONE immutable")
        train,val,tri,vai=get_train_val(a.dataset,"/root/private_data/datasets/public",TR_SEED,BATCH,a.workers)
        assert len(train)==TRAIN_N and len(val)==VAL_N
        idxhash=lambda idx:hashlib.sha256(json.dumps(idx,separators=(",",":")).encode()).hexdigest()
        studentprotocol=Path(__file__).with_name("b2_protocol.py")
        studentprotohash=full_sha256(studentprotocol)
        teacher=Teacher(NUM_CLASSES[a.dataset]).to(device)
        initsha=weights_hash(teacher)
        opt=torch.optim.SGD(teacher.parameters(),lr=.1,momentum=.9,weight_decay=5e-4,nesterov=False)
        warmup=LinearLR(opt,start_factor=.01,end_factor=1.,total_iters=5)
        cosine=CosineAnnealingLR(opt,T_max=195)
        sched=SequentialLR(opt,schedulers=[warmup,cosine],milestones=[5])
        manifest={
          "protocol":"CrossScore-B2-V1.2-cleanteacher-20261020",
          "dataset":a.dataset,"primary":"epoch200_final.pth","teacher_seed":TR_SEED,
          "train_n":len(train),"val_n":len(val),"split_seed":SPLIT_SEED,
          "train_index_sha256":idxhash(tri),"val_index_sha256":idxhash(vai),
          "data_root":"/root/private_data/datasets/public",
          "source_sha256":{"trainer":full_sha256(Path(__file__)),"b2_protocol":studentprotohash,
              "V1.2_teacher_protocol":full_sha256(PROTOCOL)},
          "arch":"CIFAR-ResNet50","init_sha256":initsha,
          "epochs":epochs,"mode":a.mode,"batch_size":BATCH,
          "optimizer":{"type":"SGD","lr":.1,"momentum":.9,"weight_decay":.0005,"nesterov":False},
          "schedule":{"warmup_epochs":5,"warmup_start_factor":.01,"cosine_epochs":195},
          "selection":"PREDECLARED EPOCH200 PRIMARY; best-ID-val only secondary",
          "no_official_test_access":True,"no_ood_access":True,
          "smoke_train_cap":a.smoke_train_batches if a.mode=="smoke" else None,
          "smoke_val_cap":a.smoke_val_batches if a.mode=="smoke" else None,
          "run_label":a.run_label,
          "host":socket.gethostname(),"python":sys.version,"pytorch":torch.__version__,
          "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        if a.resume:
            saved=json.loads((jobdir/"manifest.json").read_text())
            # Runtime host is not part of frozen scientific definition, all other fields must match
            for k in manifest:
                if k in ("host","start_utc","python","pytorch"):continue
                if saved[k]!=manifest[k]:raise RuntimeError("RESUME_MANIFEST_MISMATCH "+k)
            ck=torch.load(jobdir/"latest.pt",map_location="cpu",weights_only=True)
            if ck["manifest_sha256"]!=full_sha256(jobdir/"manifest.json"):raise RuntimeError("manifest altered")
            teacher.load_state_dict(ck["model"],strict=True)
            opt.load_state_dict(ck["optimizer"]);sched.load_state_dict(ck["scheduler"])
            for state in opt.state.values():
                for key,item in state.items():
                    if torch.is_tensor(item):state[key]=item.to(device)
            rng_load(ck["rng"])
            first=ck["next_epoch"];best=ck["best_val"]
        else:
            save_json(manifest,jobdir/"manifest.json")
            first=1;best=-1
        with (jobdir/"history.jsonl").open("a") as hf:
            for epoch in range(first,epochs+1):
                epstart=time.monotonic()
                teacher.train()
                total=0;traincorrect=0;loss_sum=0.
                trloader=make_loader(train,TR_SEED,epoch,True,batch_size=BATCH,workers=a.workers)
                for bi,(x,y) in enumerate(trloader):
                    if a.mode=="smoke" and bi>=a.smoke_train_batches:break
                    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                    z=teacher(x)
                    loss=F.cross_entropy(z,y)
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    total+=len(y);traincorrect+=int((z.detach().argmax(1)==y).sum())
                    loss_sum+=float(loss.detach())*len(y)
                vloader=make_loader(val,TR_SEED,epoch,False,batch_size=BATCH,workers=a.workers)
                ev=validate(teacher,vloader,device,
                    max_batches=(a.smoke_val_batches if a.mode=="smoke" else None))
                sched.step()
                if ev["val_acc"]>best:
                    best=ev["val_acc"]
                    save_torch(teacher.state_dict(),jobdir/"best_id_val_SECONDARY.pth")
                row={"epoch":epoch,"train_n":total,"train_loss":loss_sum/max(total,1),
                     "train_acc":traincorrect/max(total,1),
                     "lr_post":sched.get_last_lr()[0],"elapsed_sec":round(time.monotonic()-epstart,3),**ev}
                hf.write(json.dumps(row,ensure_ascii=False)+"\n");hf.flush();os.fsync(hf.fileno())
                state={"model":teacher.state_dict(),"optimizer":opt.state_dict(),
                       "scheduler":sched.state_dict(),"rng":rng_save(),
                       "next_epoch":epoch+1,"best_val":best,
                       "manifest_sha256":full_sha256(jobdir/"manifest.json")}
                save_torch(state,jobdir/"latest.pt")
                print("EPOCH",json.dumps(row),flush=True)
                if a.mode=="smoke" and a.stop_after_epoch and epoch>=a.stop_after_epoch:
                    print("INTENTIONAL_SMOKE_STOP",flush=True);return
        if a.mode=="formal" and epochs!=200:raise RuntimeError("bad final epoch")
        name="epoch200_final.pth" if a.mode=="formal" else "smoke_final.pth"
        save_torch(teacher.state_dict(),jobdir/name)
        finsha=full_sha256(jobdir/name)
        done={"dataset":a.dataset,"mode":a.mode,"epoch":epochs,"selection":"EPOCH200_FINAL_ONLY",
              "weights_sha256":finsha,"parameter_hash":weights_hash(teacher),
              "best_id_val_secondary":best,"elapsed_sec":round(time.monotonic()-start,2),
              "finish_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save_json(done,jobdir/"DONE.json")
        print("COMPLETE",json.dumps(done,ensure_ascii=False),flush=True)
    finally:
        lease.rmdir()
if __name__=="__main__":main()
