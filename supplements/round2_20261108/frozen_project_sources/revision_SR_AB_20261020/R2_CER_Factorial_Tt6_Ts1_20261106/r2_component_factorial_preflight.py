#!/usr/bin/env python3
"""CPU-only audit of component wiring and historical anchor provenance."""
import ast, csv, hashlib, importlib.util, itertools, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

STUDY=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def audit_run(r):
 p=Path(r['path']);m=json.loads((p/'manifest.json').read_text());done=json.loads((p/'COMPLETE.json').read_text())
 hist=list(csv.DictReader((p/'history.csv').open()))
 assert len(hist)==60 and [int(x['epoch']) for x in hist]==list(range(1,61)),p
 assert done['complete_epochs']==60 and m['seed']==r['seed'],p
 for k,v in {'epochs':60,'batch_size':512,'num_workers':16,'teacher_temperature':6,'student_probability_temperature':1,'lr':.1,'weight_decay':.0005,'rank_margin':.5,'id_train_n':258951,'id_val_n':1000,'max_train_batches':0,'max_val_batches':0,'ood_data_used_for_training':False,'selection':'ID validation accuracy only; epoch-0 eligible'}.items():assert m[k]==v,(p,k,m[k],v)
 c,e,rank=r['components']
 actual=[0,0,0] if m['mode']=='CE' else [m['lambda_conf'],m['lambda_uncert'],m['lambda_order']]
 assert actual==[c,e,rank],(p,actual,r)
 assert all(math.isfinite(float(v)) for row in hist for v in row.values()),p
 hashes={f:sha(p/f) for f in ['best.pt','manifest.json','COMPLETE.json','history.csv']}
 if 'frozen_hashes' in r:assert hashes==r['frozen_hashes'],p
 return {**r,'frozen_hashes':hashes,'amp_skipped_steps':sum(int(x['skipped_amp_steps']) for x in hist),'best_val_acc':done['best_val_acc']}
def main():
 plan=json.loads((STUDY/'FROZEN_FACTORIAL_PROTOCOL.json').read_text())
 for r in plan['source_dependencies']+plan['dataset_lists']:
  assert sha(r['path'])==r['sha256'],r['path']
 assert sha(STUDY/'r2_component_factorial_train.py')==plan['trainer_sha256']
 a=ast.parse(Path(plan['historical_trainer']).read_text());b=ast.parse((STUDY/'r2_component_factorial_train.py').read_text())
 defs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 da,db=defs(a),defs(b);identical=['TrainSet','AuxHead','Student','seedall','eval_acc','teacher_signals','rank_hinge']
 for name in identical:assert da[name]==db[name],name
 spec=importlib.util.spec_from_file_location('new_trainer',STUDY/'r2_component_factorial_train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 torch.manual_seed(813);z0=torch.randn(9,200);h0=torch.randn(9);tz=torch.randn(9,200);labels=torch.arange(9)
 tc,te,tu=mod.teacher_signals(tz,6.)
 tests=[]
 for c,e,r in itertools.product([0,1],repeat=3):
  grads=[];losses=[]
  for branch in ['implemented','reference']:
   z=z0.clone().requires_grad_(True);h=h0.clone().requires_grad_(True);sp=z.softmax(1)
   ce=F.cross_entropy(z,labels);conf=F.mse_loss(sp.amax(1),tc);ent=F.mse_loss(-(sp*sp.clamp_min(1e-8).log()).sum(1),te);rank=mod.rank_hinge(h,tu,.5)
   actual=mod.combine_losses(ce,conf,rank,ent,c,e,r) if branch=='implemented' else ce+(conf if c else z.sum()*0)+(rank if r else h.sum()*0)+(ent if e else z.sum()*0)
   g=torch.autograd.grad(actual,(z,h),allow_unused=True);grads.append([x if x is not None else torch.zeros_like(y) for x,y in zip(g,(z,h))]);losses.append(actual.detach())
  diff=max((x-y).abs().max().item() for x,y in zip(*grads));ld=abs(float(losses[0]-losses[1]));assert diff<1e-7 and ld<1e-7
  tests.append({'C':c,'E':e,'R':r,'max_gradient_abs_difference':diff,'loss_abs_difference':ld})
 from r2_component_factorial_analyze import contrast
 fixture={(x,42):1+2*x[0]+3*x[1]+4*x[2]+5*x[0]*x[1]+6*x[0]*x[2]+7*x[1]*x[2]+8*x[0]*x[1]*x[2] for x in itertools.product([0,1],repeat=3)}
 for axes,expected in [((0,),9.5),((1,),11),((2,),12.5),((0,1),9),((0,2),10),((1,2),11),((0,1,2),8)]:assert contrast(fixture,42,axes)==expected
 anchors=[audit_run(r) for r in plan['anchors']];assert len(anchors)==12
 out={'status':'PASS_CPU_COMPONENT_AND_ANCHOR_AUDIT','no_GPU_used':True,'identical_historical_definitions':identical,'all_eight_component_loss_and_gradient_tests':tests,'seven_factorial_contrast_fixture_tests':'PASS','anchors':anchors,'n_anchor_hash_checks':48,'scope':'Initial historical RNG states were not archived; protocol-matched reuse does not establish bitwise rerun equivalence.'}
 (STUDY/'CPU_PREFLIGHT_PASS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'n_anchors':12,'n_hash_checks':48,'n_gradient_tests':8}),flush=True)
if __name__=='__main__':main()
