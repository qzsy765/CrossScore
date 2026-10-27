#!/usr/bin/env python3
"""P1-7B read-only multi-score diagnostics on frozen P1-6 ImageNet-200 checkpoints.
All outputs in separate P1_7B directory. No refitting, calibration or OOD selection.
"""
import argparse,sys,json,hashlib,time
from pathlib import Path
import os
import numpy as np,torch
from torch.utils.data import DataLoader
EXP=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
sys.path.insert(0,str(EXP/'source'))
from eval_im200_same_method_recovery import Student
from im200_data_eval_recovery import dataset
from b2_protocol import eval_ood_scores
DOMAINS=['ssb_hard','ninco','inaturalist','texture','openimage_o']
LENGTHS={'test':9000,'ssb_hard':49000,'ninco':5879,'inaturalist':10000,'texture':5160,'openimage_o':15869}
SCORES=['energy','neg_msp','neg_maxlogit','entropy','aux_uncert','centered_energy']
@torch.inference_mode()
def extract(model,kind,batch,workers):
    ds=dataset(kind)
    assert len(ds)==LENGTHS[kind],(kind,len(ds),LENGTHS[kind])
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=False)
    out={k:[] for k in SCORES}
    t=time.time()
    for bi,(x,_) in enumerate(dl):
        logits,h=model(x.to('cuda:0',non_blocking=True))
        z=logits.float()
        p=z.softmax(1)
        vmax=z.amax(1)
        msp=p.amax(1)
        entropy=-(p*p.clamp_min(1e-12).log()).sum(1)
        values={'energy':-z.logsumexp(1),'neg_msp':-msp,'neg_maxlogit':-vmax,'entropy':entropy,'aux_uncert':h.float(),'centered_energy':-(z-z.mean(1,keepdim=True)).logsumexp(1)}
        for k,tensor in values.items():out[k].append(tensor.detach().cpu().numpy().astype('float32'))
        if bi%45==0:print(f'PASS {kind} batch={bi}/{len(dl)} sec={time.time()-t:.1f}',flush=True)
    return {k:np.concatenate(vals) for k,vals in out.items()}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    if a.run not in {f'{m}_seed{s}' for m in ['CE','CrossScore_T6_Corrected'] for s in range(42,47)}: raise ValueError('unexpected run')
    run=EXP/'runs'/a.run
    assert (run/'COMPLETE.json').exists()
    out=EXP/'P1_7B_score_family_20261025'/a.run
    out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model=Student();model.load_state_dict(ck['model'],strict=True);model=model.to('cuda:0').eval()
    raw={};details={}
    for d in ['test']+DOMAINS:
        row=extract(model,d,a.batch_size,a.workers)
        for score,v in row.items():
            assert len(v)==LENGTHS[d] and np.isfinite(v).all()
            raw[f'{d}__{score}']=v
        details[d]={k:{'mean':float(v.mean()),'sd':float(v.std())} for k,v in row.items()}
    old=EXP/'evaluations'/a.run/'energy_scores_per_image.npz'
    with np.load(old) as frozen:
        diffs={d:float(np.max(np.abs(raw[f'{d}__energy']-frozen[('id_test' if d=='test' else d)+'_energy']))) for d in ['test']+DOMAINS}
    assert max(diffs.values())<1e-4,('original_energy_mismatch',diffs)
    print('FROZEN_ENERGY_CROSSCHECK_PASS',json.dumps(diffs),flush=True)
    metrics={}
    for score in SCORES:
        idsc=raw[f'test__{score}']
        metrics[score]={}
        for domain in DOMAINS:
            oodsc=raw[f'{domain}__{score}']
            metrics[score][domain]=eval_ood_scores(idsc,oodsc)['OOD_positive']
    rawpath=out/'score_arrays.npz'
    np.savez_compressed(rawpath,**raw)
    result={'run':a.run,'score_family':SCORES,'run_type':'EXPLORATORY_READ_ONLY_DIAGNOSTIC',
       'purpose':'Compare rank/scale behavior across frozen checkpoints; auxiliary head is NOT a deployed method',
       'score_ood_positive':True,'score_orientation':{'energy':'-logsumexp(logits)','neg_msp':'-max(softmax(logits))','neg_maxlogit':'-max(logits)','entropy':'predictive_entropy(logits)','aux_uncert':'auxiliary scalar predicted training uncertainty, higher interpreted as OOD','centered_energy':'-logsumexp(logits - mean(logits)), removes sample-dependent logit offset'},
       'selected_epoch':ck.get('epoch'),'model_hash_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
       'frozen_energy_exact_score_max_abs_diff':diffs,'sample_counts':LENGTHS,'distribution_statistics':details,'ood_metrics':metrics,
       'raw_scores_path':str(rawpath)}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps({'run':a.run,'finished':True,'raw_score_file':str(rawpath),'metrics_file':str(out/'metrics.json')},indent=2)+'\n')
    print('DIAGNOSTIC_COMPLETE',a.run,flush=True)
if __name__=='__main__':main()

