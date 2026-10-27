#!/usr/bin/env python3
"""P2-A frozen ID-val local gradient and geometry audit; never trains or evaluates OOD."""
import argparse,hashlib,json,math,os,socket,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))
EXP=ROOT/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
PHASE=EXP/'P2_C0_gradient_temperature_decoupling_20261027'
sys.path.insert(0,str(EXP/'source'))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_im200_same_method_recovery import Student
from im200_data_eval_recovery import dataset
from im200_models import teacher_restricted

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rank_hinge(h,u,margin=.5):
    td=u[:,None]-u[None,:]; sd=h[:,None]-h[None,:]
    mask=~torch.eye(len(u),dtype=torch.bool,device=u.device)
    return F.relu(margin-td.sign()*sd)[mask].mean()

def summaries(params,grads):
    groups={
        'shared_backbone':[i for i,(n,p) in enumerate(params) if n.startswith('backbone.') and not n.startswith('backbone.fc.')],
        'layer4':[i for i,(n,p) in enumerate(params) if n.startswith('backbone.layer4.')],
        'classifier_fc':[i for i,(n,p) in enumerate(params) if n.startswith('backbone.fc.')],
        'aux_head':[i for i,(n,p) in enumerate(params) if n.startswith('ood_head.')]
    }
    names=list(grads.keys())
    result={}
    for group,ids in groups.items():
        nn={k:float(math.sqrt(sum(float(g[i].float().square().sum().item()) for i in ids if g[i] is not None))) for k,g in grads.items()}
        dots={}
        for i,a in enumerate(names):
            for b in names[i+1:]:
                dot=sum(float((grads[a][t].float()*grads[b][t].float()).sum().item()) for t in ids if grads[a][t] is not None and grads[b][t] is not None)
                dots[a+'_vs_'+b] = float(dot/max(1e-15,nn[a]*nn[b])) if nn[a]>1e-12 and nn[b]>1e-12 else None
        result[group]={'norms':nn,'cosines':dots}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',required=True,type=int,choices=[44,45,46])
    args=ap.parse_args()
    seed=args.seed
    protocol_path=PHASE/'FROZEN_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text())
    codehash=sha(Path(__file__))
    assert codehash==protocol['source_sha256'], 'source mutated after P2-C0 freeze'
    assert protocol['frozen'] and protocol['new_training']==False
    assert protocol['seeds']==[44,45,46]
    assert not (PHASE/f'seed{seed}').exists()
    # Dedicated run directory is a strict create-only lock. Never modifies checkpoints.
    out=PHASE/f'seed{seed}'
    out.mkdir(exist_ok=False)
    (out/'RUNNING').write_text(str(os.getpid())+'\n')
    started=time.time()
    try:
        assert torch.cuda.is_available()
        gpu=subprocess_gpu_status()
        assert gpu['memory_mib']<=300 and gpu['utilization_pct']<=40, f'GPU currently occupied: {gpu}'
        torch.set_num_threads(4)
        torch.manual_seed(271026)
        torch.backends.cudnn.benchmark=False
        ds=dataset('val')
        rng=np.random.default_rng(271026)
        indices=rng.choice(len(ds),size=96,replace=False).tolist()
        indexhash=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()
        loader=DataLoader(Subset(ds,indices),batch_size=24,shuffle=False,num_workers=0)
        cached_batches=list(loader)
        assert len(cached_batches)==4
        device=torch.device('cuda:0')
        teacher,tidx=teacher_restricted()
        teacher=teacher.to(device).eval().requires_grad_(False)
        tidx=tidx.to(device)
        teacher_logits=[]
        with torch.no_grad():
            for x,y in cached_batches:
                zz=teacher(x.to(device)).float().index_select(1,tidx).cpu()
                teacher_logits.append(zz)
        del teacher
        torch.cuda.empty_cache()
        results=[]
        paths={
            'T1_E1_GRADDOWN':EXP/'P2_B_gradient_weight_intervention_20261026'/f'T1_E1_GRADDOWN_seed{seed}'/'best.pt',
            'T6_E1_GRADUP':EXP/'P2_B_gradient_weight_intervention_20261026'/f'T6_E1_GRADUP_seed{seed}'/'best.pt',
            'T6_E1_ORIGINAL':EXP/'P1_7E_heldout_seed_replication_20261026'/f'T6_E1_seed{seed}'/'best.pt',
            'T1_E1_ORIGINAL':EXP/'runs'/f'CrossScore_T6_Corrected_seed{seed}'/'best.pt',
            'CE':EXP/'runs'/f'CE_seed{seed}'/'best.pt'
        }
        frozen=json.loads((EXP/'P2_B_gradient_weight_intervention_20261026/P2B_ID_ONLY_FROZEN_GATE.json').read_text())
        assert frozen['status']=='ID_ONLY_FROZEN_PASS' and frozen['new_models']==6
        for a in ('T1_E1_GRADDOWN','T6_E1_GRADUP'):
            recorded=next(k for k in frozen['data'] if k['arm']==a and k['seed']==seed)
            assert sha(paths[a])==recorded['checkpoint_sha256']
        weights={
          'T1_E1_GRADDOWN':(.2,.25,1.),
          'T6_E1_GRADUP':(5.,4.,1.),
          'T6_E1_ORIGINAL':(1.,1.,1.),
          'T1_E1_ORIGINAL':(1.,1.,1.),
          'CE':(0.,0.,0.)
        }

        for arm,ckpath in paths.items():
            assert ckpath.is_file()
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            model=Student().to(device)
            model.load_state_dict(ck['model'],strict=True)
            model.eval()
            params=list(model.named_parameters())
            print('ARM_BEGIN',arm,'seed',seed,'epoch',ck.get('epoch'),flush=True)
            for bi,((xc,yc),t_cpu) in enumerate(zip(cached_batches,teacher_logits)):
                x=xc.to(device);y=yc.to(device);tz=t_cpu.to(device)
                z,h=model(x)
                z=z.float();h=h.float()
                ce=F.cross_entropy(z,y)
                with torch.no_grad():
                    tp=F.softmax(tz/6.0,dim=1)
                    tc=tp.amax(1)
                    te=-(tp*tp.clamp_min(1e-8).log()).sum(1)
                    tu=1.0-tc
                    raw_prob=z.softmax(1)
                    score_energy=-z.logsumexp(1)
                rank=rank_hinge(h,tu,.5)
                for temp in [1.0,6.0]:
                    sp=F.softmax(z/temp,dim=1)
                    sc=sp.amax(1)
                    se=-(sp*sp.clamp_min(1e-8).log()).sum(1)
                    conf=F.mse_loss(sc,tc)
                    ent=F.mse_loss(se,te)
                    losses={'ce':ce,'conf':conf,'rank':rank,'entropy':ent}
                    grads={}
                    for k,loss in losses.items():
                        grads[k]=[gi.detach() if gi is not None else None for gi in torch.autograd.grad(loss,[pp for nn,pp in params],retain_graph=True,allow_unused=True)]
                    wconf,went,wrank=weights[arm]
                    reg=[None if all(grads[k][ix] is None for k in ('conf','entropy','rank')) else sum((scale*grads[k][ix] for k,scale in [('conf',wconf),('entropy',went),('rank',wrank)] if grads[k][ix] is not None),torch.zeros_like(pp)) for ix,(nn,pp) in enumerate(params)]
                    total=[(grads['ce'][ix] if grads['ce'][ix] is not None else torch.zeros_like(pp))+(reg[ix] if reg[ix] is not None else torch.zeros_like(pp)) for ix,(nn,pp) in enumerate(params)]
                    grads['weighted_aux_total']=reg
                    grads['objective_total']=total
                    geom=summaries(params,grads)
                    entry={
                        'arm':arm,'seed':seed,'batch_index':bi,'n':len(y),'student_T_diagnostic':temp,
                        'losses':{k:float(v.detach()) for k,v in losses.items()},'nominal_training_weights':dict(conf=wconf,entropy=went,rank=wrank),
                        'geometry':{
                            'teacher_entropy_mean':float(te.mean()),
                            'teacher_conf_mean':float(tc.mean()),
                            'student_entropy_at_T_mean':float(se.mean()),
                            'student_conf_at_T_mean':float(sc.mean()),
                            'entropy_target_minus_student_mean':float((te-se).mean()),
                            'student_logit_mean':float(z.mean()),
                            'student_logit_std':float(z.std(unbiased=False)),
                            'student_energy_mean':float(score_energy.mean()),
                            'student_energy_std':float(score_energy.std(unbiased=False)),
                            'student_raw_predictive_entropy_mean':float((-(raw_prob*raw_prob.clamp_min(1e-8).log()).sum(1)).mean()),
                            'batch_ID_val_acc':float((z.argmax(1)==y).float().mean())
                        },
                        'gradients':geom
                    }
                    results.append(entry)
                    del grads
                print('BATCH_DONE',arm,'seed',seed,'batch',bi+1,'of',len(cached_batches),'elapsed',round(time.time()-started,1),flush=True)
                del z,h,ce,rank
            del model
            torch.cuda.empty_cache()
            print('ARM_DONE',arm,'seed',seed,flush=True)
        record={
            'study':'P2-C0 temperature-gradient decoupling on post-trained frozen five-model arms, diagnosis-only',
            'seed':seed,'host':socket.gethostname(),'checkpoint_sha256':{k:sha(v) for k,v in paths.items()},
            'protocol_sha256':sha(protocol_path),'source_sha256':codehash,
            'val_n':len(ds),'subset_indices_sha256':indexhash,'subset_indices':indices,
            'n_subset':len(indices),'batch_size':24,'batches':4,'teacher_T':6.0,'model_arms':list(paths),'counterfactual_student_temperatures':[1.,6.],
            'parameter_update_count':0,'OOD_used':False,
            'results':results,'elapsed_seconds':time.time()-started
        }
        tmp=out/'P2C0_RESULT.json.tmp'
        tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        tmp.rename(out/'P2C0_RESULT.json')
        (out/'COMPLETE.json').write_text(json.dumps({'seed':seed,'status':'PASS','result_sha256':sha(out/'P2C0_RESULT.json'),'source_sha256':codehash,'n_batch_records':len(results)},indent=2)+'\n')
        print('P2C0_COMPLETE',seed,'records',len(results),'seconds',round(time.time()-started,1),flush=True)
    finally:
        (out/'RUNNING').unlink(missing_ok=True)

def subprocess_gpu_status():
    import subprocess
    a=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(a)==1
    mem,util=[int(x.strip()) for x in a[0].split(',')]
    return {'memory_mib':mem,'utilization_pct':util}

if __name__=='__main__':
    main()
