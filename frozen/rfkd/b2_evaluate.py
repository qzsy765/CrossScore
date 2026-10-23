"""CrossScore B2 V1.0 six-OOD evaluation; OOD-positive + ID-positive FPR95, per-image scores.
Never selects checkpoint based on test/OOD; refuses incomplete six-domain outputs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from b2_protocol import (NUM_CLASSES,METHODS,OOD,PROTOCOL_NAME,Student,eval_ood_scores,full_sha256)
PROTOCOL_NAME='CrossScore-SR-P1-5-RFKD-Matched-20261022'
ROOT=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020')
sys.path.insert(0,str(ROOT/'audit_scripts'))
from eval_uploaded_checkpoints import loader_for,id_loader

class LogitsOnly(torch.nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,x):return self.student(x)[0]

@torch.inference_mode()
def scores(m,dl,limit=None):
    out={k:[] for k in ('energy_neg','msp_neg','entropy')}
    for i,(x,_) in enumerate(dl):
        if limit is not None and i>=limit:break
        z=m(x.cuda(non_blocking=True)).float()
        p=z.softmax(dim=1)
        out['energy_neg'].append((-torch.logsumexp(z,dim=1)).cpu().numpy())
        out['msp_neg'].append((1-p.amax(dim=1)).cpu().numpy())
        out['entropy'].append((-(p*p.clamp_min(1e-8).log()).sum(dim=1)).cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def write_json_atomic(p,obj):
    tmp=p.with_name(p.name+'.pending.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    os.replace(tmp,p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=NUM_CLASSES,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--checkpoint',choices=['primary','secondary'],default='primary')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-run-label',default='')
    p.add_argument('--smoke-batches',type=int,default=1)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    base=ROOT/'P1_SR_ReviewerEvidence_20261022/P1_5_RFKD_Mechanism_Comparator_20261022'/('smoke_runs' if a.smoke else 'formal_runs')
    tag=f'{a.dataset}__seed{a.seed}__{a.method}'
    if a.smoke:
        if not a.smoke_run_label:raise ValueError('smoke-run-label required')
        tag+='__SMOKE_'+a.smoke_run_label
    job=base/tag
    done=job/'DONE.json'
    if not done.is_file():raise RuntimeError('Model not complete: '+str(job))
    state_file=job/('best_id_val.pt' if a.checkpoint=='secondary' else ('smoke_final.pt' if a.smoke else 'epoch200_final.pt'))
    if not state_file.is_file():raise FileNotFoundError(state_file)
    arch=Student(NUM_CLASSES[a.dataset])
    arch.load_state_dict(torch.load(state_file,map_location='cpu',weights_only=True),strict=True)
    model=LogitsOnly(arch.cuda().eval())
    output=(ROOT/'P1_SR_ReviewerEvidence_20261022/P1_5_RFKD_Mechanism_Comparator_20261022'/('eval_smoke' if a.smoke else 'eval_formal')/tag/a.checkpoint)
    output.mkdir(parents=True,exist_ok=False)
    meta={'protocol':PROTOCOL_NAME,'dataset':a.dataset,'method':a.method,'seed':a.seed,
          'model_sha256':full_sha256(state_file),'checkpoint_selection':a.checkpoint,
          'smoke':a.smoke,'ood_positive':True,'ood_order':'SVHN,Textures,LSUN-Crop,LSUN-Resize,Places365,iSUN',
          'score_sign':'higher score = more OOD','fpr95_method':'full unique score threshold ties included',
          'metrics':{},'per_image_files':{},'ood_counts':{}}
    cap=a.smoke_batches if a.smoke else None
    iddl,_=id_loader(a.dataset,a.batch_size,a.workers)
    sid=scores(model,iddl,cap)
    for od in OOD:
        dl,_=loader_for(od,a.batch_size,a.workers)
        so=scores(model,dl,cap)
        meta['ood_counts'][od]=len(so['energy_neg'])
        meta['metrics'][od]={}
        for kind in sid:
            met=eval_ood_scores(sid[kind],so[kind])
            meta['metrics'][od][kind]=met
            score_file=f'{od}__{kind}.npz'
            np.savez_compressed(output/score_file,id=sid[kind],ood=so[kind])
            meta['per_image_files'][od+'__'+kind]=score_file
            print('OOD',od,kind,met['OOD_positive'],flush=True)
    if set(meta['metrics'])!=set(OOD):raise RuntimeError('Incomplete OOD; refuse DONE')
    meta['macro']={kind:{cls:{metric:float(np.mean([meta['metrics'][od][kind][cls][metric] for od in OOD]))
            for metric in ('AUROC','AUPR','FPR95_complete_ties')}
           for cls in ('OOD_positive','ID_positive')}
          for kind in sid}
    meta['utc_completed']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json_atomic(output/'results.json',meta)
    write_json_atomic(output/'DONE.json',{'protocol':PROTOCOL_NAME,'tag':tag,'six_ood_present':True,
        'model_sha256':meta['model_sha256'],'eval_sha256':full_sha256(Path(__file__))})
    print('COMPLETE',json.dumps(meta['macro']['energy_neg'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
