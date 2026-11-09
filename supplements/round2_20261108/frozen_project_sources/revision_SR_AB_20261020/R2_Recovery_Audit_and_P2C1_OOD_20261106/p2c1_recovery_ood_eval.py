#!/usr/bin/env python3
"""Evaluate frozen P2-C1 weights; no training or OOD-driven selection."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path('/root/private_data/projects/OOD/cross_score_v2')
EXP=ROOT/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
C1=EXP/'P2_C1_gradient_norm_matched_IDonly_20261027'
STUDY=ROOT/'revision_SR_AB_20261020/R2_Recovery_Audit_and_P2C1_OOD_20261106'
sys.path.insert(0,str(EXP/'source'))
from eval_im200_same_method_recovery import Student
from im200_data_eval_recovery import dataset
from b2_protocol import eval_ood_scores
DOMAINS=['ssb_hard','ninco','inaturalist','texture','openimage_o']
LENGTHS={'test':9000,'ssb_hard':49000,'ninco':5879,'inaturalist':10000,'texture':5160,'openimage_o':15869}
SCORES=['energy','neg_msp','neg_maxlogit','entropy','aux_uncert','centered_energy']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

@torch.inference_mode()
def extract(model,kind,batch,workers):
    ds=dataset(kind)
    assert len(ds)==LENGTHS[kind],(kind,len(ds))
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=False)
    values={k:[] for k in SCORES};correct=0;seen=0;started=time.time()
    for bi,(x,y) in enumerate(loader):
        z,h=model(x.to('cuda:0',non_blocking=True));z=z.float();p=z.softmax(1)
        batch_values={'energy':-z.logsumexp(1),'neg_msp':-p.amax(1),'neg_maxlogit':-z.amax(1),
            'entropy':-(p*p.clamp_min(1e-12).log()).sum(1),'aux_uncert':h.float(),
            'centered_energy':-(z-z.mean(1,keepdim=True)).logsumexp(1)}
        for k,tensor in batch_values.items():values[k].append(tensor.cpu().numpy().astype('float32'))
        if kind=='test':correct+=(z.argmax(1).cpu()==y).sum().item();seen+=len(y)
        if bi%40==0:print('EVAL_PROGRESS',kind,bi,len(loader),round(time.time()-started,2),flush=True)
    arrays={k:np.concatenate(v) for k,v in values.items()}
    assert all(len(v)==LENGTHS[kind] and np.isfinite(v).all() for v in arrays.values())
    return arrays,(correct/seen if seen else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--batch-size',type=int,default=256);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    plan=json.loads((STUDY/'FROZEN_OOD_PROTOCOL.json').read_text())
    assert sha(Path(__file__))==plan['evaluation_source_sha256']
    gate_path=C1/'P2C1_ID_ONLY_FROZEN_GATE.json'
    assert sha(gate_path)==plan['id_gate_sha256']
    gate=json.loads(gate_path.read_text())
    assert gate['status']=='ID_ONLY_FROZEN_PASS' and gate['n_models']==6
    assert a.run in plan['runs']
    expected=next(r for r in gate['records'] if r['arm']+'_seed'+str(r['seed'])==a.run)
    run=C1/a.run;checkpoint=run/'best.pt';out=STUDY/'OOD_evaluations'/a.run
    assert sha(checkpoint)==expected['checkpoint_sha256']
    for item in plan['source_dependencies']:
        assert sha(item['path'])==item['sha256'],item['path']
    for item in plan['dataset_lists'].values():
        assert sha(item['list_path'])==item['sha256'],item['list_path']
    out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Student();model.load_state_dict(ck['model'],strict=True);model.to('cuda:0').eval()
    reference=json.loads((run/'ID_ONLY_DIAGNOSTICS.json').read_text())
    arrays,idacc=extract(model,'test',a.batch_size,a.workers)
    frozen=reference['score_stats']['energy']
    diffs={'energy_mean':abs(float(arrays['energy'].mean())-frozen['mean']),
           'energy_std':abs(float(arrays['energy'].std())-frozen['std']),
           'top1':abs(idacc-reference['id_test_top1'])}
    assert diffs['energy_mean']<1e-4 and diffs['energy_std']<1e-4 and diffs['top1']<1e-10,('ID_CROSSCHECK_FAILED',diffs)
    (out/'ID_REPLAY_PASS.json').write_text(json.dumps({'run':a.run,'differences':diffs,'before_any_OOD':True},indent=2)+'\n')
    print('ID_REPLAY_PASS_BEFORE_OOD',a.run,json.dumps(diffs),flush=True)
    raw={f'test__{k}':v for k,v in arrays.items()};stats={'test':{k:{'mean':float(v.mean()),'sd':float(v.std())} for k,v in arrays.items()}}
    for domain in DOMAINS:
        values,_=extract(model,domain,a.batch_size,a.workers)
        for k,v in values.items():raw[f'{domain}__{k}']=v
        stats[domain]={k:{'mean':float(v.mean()),'sd':float(v.std())} for k,v in values.items()}
    metrics={score:{domain:eval_ood_scores(raw[f'test__{score}'],raw[f'{domain}__{score}'])['OOD_positive'] for domain in DOMAINS} for score in SCORES}
    np.savez_compressed(out/'score_arrays.npz',**raw)
    result={'run':a.run,'seed':expected['seed'],'arm':expected['arm'],'selected_epoch':ck['epoch'],'selected_checkpoint_val_acc':ck['val_acc'],
        'id_test_top1':idacc,'checkpoint_sha256':sha(checkpoint),'protocol_sha256':sha(STUDY/'FROZEN_OOD_PROTOCOL.json'),
        'source_sha256':sha(Path(__file__)),'score_family':SCORES,'sample_counts':LENGTHS,'ood_metrics':metrics,'distribution_statistics':stats,
        'frozen_ID_replay_abs_differences':diffs,'scope':'Exploratory post-ID-freeze gradient magnitude intervention; same previously examined OOD domains; no retraining',
        'no_OOD_model_selection':True,'rank_and_backbone_gradient_norm_not_matched':True}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps({'run':a.run,'finished':True,'metrics_sha256':sha(out/'metrics.json'),'raw_scores_sha256':sha(out/'score_arrays.npz')},indent=2)+'\n')
    print('OOD_COMPLETE',a.run,flush=True)

if __name__=='__main__':main()
