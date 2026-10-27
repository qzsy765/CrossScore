#!/usr/bin/env python3
"""Exploratory 2x2 factorial ID-only test and score geometry; no OOD data."""
import argparse,json,time,sys
from pathlib import Path
import os
import torch,numpy as np
from torch.utils.data import DataLoader
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
sys.path.insert(0,str(ROOT/'source'))
from eval_im200_same_method_recovery import Student
from im200_data_eval_recovery import dataset
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 run=ROOT/'P2_B_gradient_weight_intervention_20261026'/a.run
 assert run.is_dir() and (run/'COMPLETE.json').is_file()
 out=run/'ID_ONLY_DIAGNOSTICS.json';assert not out.exists()
 c=torch.load(run/'best.pt',map_location='cpu',weights_only=False);m=Student();m.load_state_dict(c['model'],strict=True);m.to('cuda:0').eval()
 d=dataset('test');assert len(d)==9000
 dl=DataLoader(d,batch_size=256,shuffle=False,num_workers=4,pin_memory=True)
 sums={k:[] for k in ['energy','neg_msp','entropy','aux']}
 correct=0;num=0;t=time.time()
 with torch.inference_mode():
  for i,(x,y) in enumerate(dl):
   z,h=m(x.to('cuda:0',non_blocking=True))
   z=z.float();p=z.softmax(dim=1)
   score={'energy':-z.logsumexp(dim=1),'neg_msp':-p.amax(dim=1),'entropy':-(p*p.clamp_min(1e-12).log()).sum(dim=1),'aux':h.float()}
   for k,v in score.items():sums[k].append(v.cpu().numpy())
   correct+=(z.argmax(1)==y.to('cuda:0')).sum().item();num+=len(y)
   if i%20==0:print('ID_ONLY_BATCH',a.run,i,len(dl),'sec',round(time.time()-t,1),flush=True)
 assert num==9000
 result={'run':a.run,'no_ood_evaluated':True,'checkpoint_epoch':c['epoch'],'checkpoint_val_acc':c['val_acc'],
 'id_test_top1':correct/num,'ID_test_N':num,'score_stats':{}}
 for k,parts in sums.items():
  v=np.concatenate(parts); assert len(v)==num and np.isfinite(v).all()
  result['score_stats'][k]={'mean':float(v.mean()),'std':float(v.std()),'q05':float(np.quantile(v,.05)),'q50':float(np.median(v)),'q95':float(np.quantile(v,.95))}
 out.write_text(json.dumps(result,indent=2)+'\n')
 print('ID_ONLY_COMPLETE',json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__':main()

