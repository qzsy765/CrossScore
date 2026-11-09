#!/usr/bin/env python3
"""Uniform re-evaluation after all 40 training checkpoints have been frozen."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
STUDY=Path(__file__).resolve().parent
PLAN=json.loads((STUDY/'FROZEN_FACTORIAL_PROTOCOL.json').read_text())
sys.path.insert(0,PLAN['recovery_study'])
from p2c1_recovery_ood_eval import Student,extract,sha,SCORES,DOMAINS,LENGTHS,eval_ood_scores
def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args()
 for r in PLAN['source_dependencies']+PLAN['dataset_lists']:assert sha(r['path'])==r['sha256'],r['path']
 gatepath=STUDY/'ALL_40_ID_FROZEN_GATE.json';gate=json.loads(gatepath.read_text());assert gate['status']=='ALL_40_ID_TRAINING_FROZEN' and len(gate['records'])==40
 r=next(x for x in gate['records'] if x['label']==a.label and x['seed']==a.seed);checkpoint=Path(r['path'])/'best.pt';assert sha(checkpoint)==r['checkpoint_sha256']
 out=STUDY/'OOD_evaluations'/(a.label+'_seed'+str(a.seed));out.mkdir(parents=True,exist_ok=False)
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model=Student();model.load_state_dict(ck['model'],strict=True);model.to('cuda:0').eval()
 arrays,idacc=extract(model,'test',256,4);raw={f'test__{k}':v for k,v in arrays.items()};stats={'test':{k:{'mean':float(v.mean()),'sd':float(v.std())} for k,v in arrays.items()}}
 (out/'ID_EVALUATION_COMPLETE_BEFORE_OOD.json').write_text(json.dumps({'id_test_top1':idacc,'sample_n':9000,'all_40_gate_sha256':sha(gatepath)},indent=2)+'\n')
 for domain in DOMAINS:
  values,_=extract(model,domain,256,4)
  for k,v in values.items():raw[f'{domain}__{k}']=v
  stats[domain]={k:{'mean':float(v.mean()),'sd':float(v.std())} for k,v in values.items()}
 metrics={score:{domain:eval_ood_scores(raw[f'test__{score}'],raw[f'{domain}__{score}'])['OOD_positive'] for domain in DOMAINS} for score in SCORES}
 np.savez_compressed(out/'score_arrays.npz',**raw)
 result={'label':a.label,'components':r['components'],'seed':a.seed,'source':'historical_anchor' if 'frozen_hashes' in r else 'new_training','checkpoint_sha256':sha(checkpoint),'selected_epoch':ck['epoch'],'selected_val_acc':ck['val_acc'],'id_test_top1':idacc,'ood_metrics':metrics,'distribution_statistics':stats,'sample_counts':LENGTHS,'protocol_sha256':sha(STUDY/'FROZEN_FACTORIAL_PROTOCOL.json'),'all_40_gate_sha256':sha(gatepath),'no_OOD_selection':True,'scope':'Post-review component factorial using a mixture of historical anchors and newly trained missing cells on previously examined OOD domains.'}
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'COMPLETE.json').write_text(json.dumps({'metrics_sha256':sha(out/'metrics.json'),'raw_sha256':sha(out/'score_arrays.npz')},indent=2)+'\n');print('OOD_COMPLETE',a.label,a.seed,flush=True)
if __name__=='__main__':main()
