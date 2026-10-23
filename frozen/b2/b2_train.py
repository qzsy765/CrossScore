"""CrossScore-SR B2 V1.0 production-ready *candidate* trainer.
Production MUST be launched only after engineering gates sign LAUNCH_READY.
Smoke testing under LAUNCH_HOLD is allowed and is isolated from formal runs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,datetime,hashlib,json,random,sys,time,socket,subprocess
from pathlib import Path
import numpy as np
import torch
from b2_protocol import (PROTOCOL_NAME,SEEDS,METHODS,NUM_CLASSES,Student,Teacher,
    seed_everything,weights_hash,full_sha256,get_train_val,make_loader,objective)

PROTOCOL_NAME='CrossScore-SR-AB-B2-V1.2-student-20261020'
PROJ=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
FREEZE=PROJ/'B2_V1.2_student_20261020'
DEFAULT_WORK=PROJ/'B2_V1.2_student_20261020'
UPLOAD=PROJ/'B2_V1.2_teacher_20261020/formal_runs'
KNOWN_TEACHERS={'cifar10':'80ea46669bb0eb8582746315358e97b0a404960363e86fe90ab64bb8ba26a41b',
                'cifar100':'8ffbd65a1f3457abfaf46d548ccb582ad9f0d3c83cc296807f9dc7e06667f8c6'}

def atomic_save(obj,path):
    tmp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    if tmp.exists():raise FileExistsError(tmp)
    torch.save(obj,tmp)
    os.replace(tmp,path)

def atomic_json(obj,path):
    tmp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    if tmp.exists():raise FileExistsError(tmp)
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,path)

def rng_export():
    npst=np.random.get_state()
    return {'python':random.getstate(),'numpy':[npst[0],npst[1].tolist(),npst[2],npst[3],npst[4]],
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def rng_restore(d):
    random.setstate(d['python'])
    st=d['numpy'];np.random.set_state((st[0],np.asarray(st[1],dtype='uint32'),st[2],st[3],st[4]))
    torch.set_rng_state(d['torch_cpu'])
    if torch.cuda.is_available() and d['torch_cuda']:
        torch.cuda.set_rng_state_all(d['torch_cuda'])

def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=sorted(NUM_CLASSES),required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--work-root',type=Path,default=DEFAULT_WORK)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Only smoke is allowed while LAUNCH_HOLD exists')
    p.add_argument('--smoke-epochs',type=int,default=2)
    p.add_argument('--smoke-max-train-batches',type=int,default=1)
    p.add_argument('--smoke-max-val-batches',type=int,default=1)
    p.add_argument('--stop-after-epoch',type=int,default=0,help='Smoke only; leave resumable checkpoint')
    p.add_argument('--run-label',default='',help='Smoke only unique test label')
    return p.parse_args()

def assert_gpu_free_for_formal_run():
    """Reject formal runs if any compute process already owns this GPU.

    Cross-server per-job mkdir locks prevent duplicate work; this additional
    guard prevents different jobs from contending for the same physical 4090.
    """
    x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,timeout=12)
    if x.returncode!=0:
        raise RuntimeError('Cannot verify GPU compute processes: '+x.stderr[:500])
    lines=[z.strip() for z in x.stdout.splitlines() if z.strip() and z.strip() not in ('[Not Supported]','No running processes found')]
    if lines:
        raise RuntimeError('Formal GPU reservation blocked, compute processes exist: '+repr(lines))

def validate_args(a):
    if not a.smoke:
        if not (FREEZE/'LAUNCH_READY').is_file() or (FREEZE/'LAUNCH_HOLD_NOT_READY').exists():
            raise RuntimeError('LAUNCH_HOLD remains; no formal GPU training allowed')
        if a.run_label or a.stop_after_epoch:raise ValueError('Smoke-only arguments forbidden in formal training')
        total_epochs=200
    else:
        total_epochs=a.smoke_epochs
        if total_epochs<1 or total_epochs>3:raise ValueError('Smoke epochs 1..3')
        if a.smoke_max_train_batches<1 or a.smoke_max_val_batches<1:raise ValueError('Bad cap')
        if not a.run_label:raise ValueError('Explicit smoke run-label required')
        if '/' in a.run_label or '..' in a.run_label:raise ValueError('Unsafe label')
    return total_epochs

@torch.inference_mode()
def validate(model,dl,dev,limit=None):
    model.eval();correct=0;loss_sum=0;n=0
    for i,(x,y) in enumerate(dl):
        if limit and i>=limit:break
        x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True)
        logits,_=model(x)
        loss=torch.nn.functional.cross_entropy(logits,y)
        loss_sum+=float(loss)*len(y);correct+=int((logits.argmax(dim=1)==y).sum());n+=len(y)
    return {'val_acc':correct/max(1,n),'val_loss':loss_sum/max(1,n),'val_n':n}

def run():
    a=args_parser(); epochs=validate_args(a)
    if not a.smoke: assert_gpu_free_for_formal_run()
    if not torch.cuda.is_available():raise RuntimeError('Require CUDA 4090 for GPU preflight')
    torch.use_deterministic_algorithms(True)
    dev=torch.device('cuda:0')
    tag=f'{a.dataset}__seed{a.seed}__{a.method}'
    if a.smoke:tag+='__SMOKE_'+a.run_label
    basedir=a.work_root/('smoke_runs' if a.smoke else 'formal_runs')
    basedir.mkdir(parents=True,exist_ok=True)
    job=basedir/tag
    # Cross-host atomic claim of a unique dataset/seed/method job directory.
    if not a.resume:job.mkdir(exist_ok=False)
    elif not job.is_dir():raise RuntimeError('Cannot resume missing task '+str(job))
    lease=job/'.active'
    lease.mkdir(exist_ok=False)
    started=time.time()
    try:
        if (job/'DONE.json').exists():raise RuntimeError('Completed task immutable; refuse overwrite')
        seed_everything(a.seed)
        train,val,train_idx,val_idx=get_train_val(a.dataset,'/root/private_data/datasets/public',a.seed,128,a.num_workers)
        dataset_fp={'split_seed':20261020,'train_indices_sha256':hashlib.sha256(json.dumps(train_idx,separators=(',',':')).encode()).hexdigest(),
            'val_indices_sha256':hashlib.sha256(json.dumps(val_idx,separators=(',',':')).encode()).hexdigest()}
        student=Student(NUM_CLASSES[a.dataset]).to(dev)
        init_hash=weights_hash(student)
        if a.method=='CE':teacher=None;teacher_sha=None
        else:
            teacher=Teacher(NUM_CLASSES[a.dataset]).to(dev)
            ckpt=UPLOAD/(a.dataset+'__TEACHER200')/'epoch200_final.pth'
            teacher_sha=full_sha256(ckpt)
            if teacher_sha!=KNOWN_TEACHERS[a.dataset]:raise RuntimeError('Teacher checkpoint fingerprint mismatch')
            teacher.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True)
            teacher.eval().requires_grad_(False)
        optimizer=torch.optim.SGD(student.parameters(),lr=.1,momentum=.9,weight_decay=5e-4,nesterov=False)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)
        manifest={'protocol':PROTOCOL_NAME,'dataset':a.dataset,'seed':a.seed,'method':a.method,
            'source':'b2_protocol.py and b2_train.py',
            'source_sha256':{'trainer':full_sha256(Path(__file__)),
                'protocol':full_sha256(Path(__file__).with_name('b2_protocol.py'))},'run_type':'SMOKE' if a.smoke else 'FORMAL',
            'total_epochs':epochs,'batch_size':128,'lr':.1,'momentum':.9,'weight_decay':5e-4,
            'seed_split':20261020,'dataset_fp':dataset_fp,
            'teacher_sha256':teacher_sha,'init_weights_sha256':init_hash,
            'smoke_train_cap':a.smoke_max_train_batches if a.smoke else None,
            'smoke_val_cap':a.smoke_max_val_batches if a.smoke else None,'host':socket.gethostname(),
            'utc_created':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if a.resume:
            saved=json.loads((job/'manifest.json').read_text())
            for k in ('protocol','dataset','seed','method','total_epochs','run_type','batch_size','dataset_fp',
                      'teacher_sha256','init_weights_sha256','source_sha256','smoke_train_cap','smoke_val_cap'):
                if saved[k]!=manifest[k]:raise RuntimeError(f'Resume manifest mismatch: {k}')
            ck=torch.load(job/'latest.pt',map_location='cpu',weights_only=True)
            if ck['epoch_next']>epochs:raise RuntimeError('Attempting to resume already complete job')
            student.load_state_dict(ck['student'],strict=True)
            optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):state[k]=v.to(dev)
            rng_restore(ck['rng'])
            first=ck['epoch_next'];best=ck['best_acc']
        else:
            atomic_json(manifest,job/'manifest.json')
            first=1;best=-1.
        with (job/'history.jsonl').open('a') as hist:
            for epoch in range(first,epochs+1):
                model_loss=0.; n=0;train_correct=0
                dl=make_loader(train,a.seed,epoch,True,workers=a.num_workers)
                student.train()
                for bi,(x,y) in enumerate(dl):
                    if a.smoke and bi>=a.smoke_max_train_batches:break
                    x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True)
                    slog,shead=student(x)
                    with torch.no_grad():tlog=teacher(x) if teacher is not None else None
                    loss,aux=objective(a.method,slog,shead,y,tlog)
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    n+=len(y);model_loss+=float(loss.detach())*len(y)
                    train_correct+=int((slog.detach().argmax(dim=1)==y).sum())
                vdl=make_loader(val,a.seed,epoch,False,workers=a.num_workers)
                vd=validate(student,vdl,dev,a.smoke_max_val_batches if a.smoke else None)
                scheduler.step()
                if vd['val_acc']>best:
                    best=vd['val_acc']
                    atomic_save(student.state_dict(),job/'best_id_val.pt')
                row={'epoch':epoch,'train_n':n,'train_loss':model_loss/max(1,n),
                    'train_acc':train_correct/max(1,n),'lr':scheduler.get_last_lr()[0],**vd}
                hist.write(json.dumps(row)+'\n');hist.flush();os.fsync(hist.fileno())
                state={'student':student.state_dict(),'optimizer':optimizer.state_dict(),
                       'scheduler':scheduler.state_dict(),'rng':rng_export(),
                       'epoch_next':epoch+1,'best_acc':best,'manifest_sha256':full_sha256(job/'manifest.json')}
                atomic_save(state,job/'latest.pt')
                print('EPOCH',json.dumps(row),flush=True)
                if a.smoke and a.stop_after_epoch and epoch>=a.stop_after_epoch:
                    print('INTENTIONAL_SMOKE_STOP_RESUMABLE',flush=True)
                    return
        if (not a.smoke and epochs!=200):raise AssertionError('Formal incomplete')
        atomic_save(student.state_dict(),job/('epoch200_final.pt' if not a.smoke else 'smoke_final.pt'))
        complete={'protocol':PROTOCOL_NAME,'job':tag,'epoch':epochs,'best_val_accuracy':best,
                  'final_weights_sha256':weights_hash(student),
                  'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'elapsed_sec':time.time()-started,'run_type':manifest['run_type']}
        atomic_json(complete,job/'DONE.json')
        print('COMPLETE',json.dumps(complete),flush=True)
    finally:
        # Remove only this process's active lease; permanent job directory remains reserved.
        lease.rmdir()
if __name__=='__main__':run()
